/* Raspberry Pi Pico 2 (RP2350A) - Complete Seismic Detection System
 * 
 * This is a STANDALONE earthquake detection system that runs entirely on RP2350A:
 * - SM-24 Geophone analog signal acquisition via ADC
 * - Edge Impulse CNN-LSTM inference for seismic classification
 * - Real-time event detection and alerting
 * - Serial output for monitoring
 * - LED and buzzer alerts
 * 
 * Hardware: Raspberry Pi Pico 2 (RP2350A)
 * Sensor: SM-24 Geophone (analog output)
 * No RTOS - Simple main loop architecture
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/gpio.h"


/* ========================================================================= */
/* HARDWARE CONFIGURATION                                                    */
/* ========================================================================= */

// SM-24 Geophone ADC Configuration
#define SM24_ADC_CHANNEL    0           // ADC0 = GPIO 26
#define SM24_ADC_PIN        26          // GPIO 26 for analog input
#define ADC_SAMPLES         64          // Averaging samples for noise reduction
#define ADC_VREF            3.3f        // ADC reference voltage

// SM-24 Geophone Specifications
#define SM24_SENSITIVITY_V_MS   28.8f   // Sensitivity: 28.8 V/m/s
#define SM24_SENSITIVITY_V_MMS  0.0288f // Sensitivity: 0.0288 V/mm/s
#define SM24_FREQ_MIN_HZ        10      // Minimum frequency: 10 Hz
#define SM24_FREQ_MAX_HZ        240     // Maximum frequency: 240 Hz

// Pin Definitions
#define LED_BUILTIN         25          // Built-in LED
#define LED_STATUS          15          // External status LED
#define LED_ALERT           14          // Alert LED (red)
#define BUZZER_PIN          16          // Buzzer for audio alerts
#define BUTTON_PIN          17          // User button for silence/reset

// Sampling Configuration
#define SAMPLE_RATE_HZ      100         // 100 Hz sampling rate
#define SAMPLE_PERIOD_MS    10          // 10ms between samples
#define WINDOW_SIZE         256         // Sample window for inference


/* ========================================================================= */
/* EDGE IMPULSE MODEL CONFIGURATION                                         */
/* ========================================================================= */

// These will be replaced with your Edge Impulse exported model
#define EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE  256
#define EI_CLASSIFIER_LABEL_COUNT           3
#define EI_CLASSIFIER_HAS_ANOMALY           0

// Placeholder for Edge Impulse categories
const char *ei_classifier_inferencing_categories[] = {
    "noise",
    "earthquake",
    "tremor"
};


/* ========================================================================= */
/* DATA STRUCTURES                                                           */
/* ========================================================================= */

typedef struct {
    float velocity_m_s;
    float velocity_mm_s;
    float raw_voltage;
    uint32_t raw_adc;
    uint64_t timestamp_ms;
} geophone_sample_t;

typedef struct {
    char label[64];
    float confidence;
    uint32_t inference_time_ms;
    uint64_t timestamp_ms;
} inference_result_t;

typedef struct {
    float buffer[WINDOW_SIZE];
    uint16_t index;
    bool filled;
} sample_buffer_t;


/* ========================================================================= */
/* GLOBAL VARIABLES                                                          */
/* ========================================================================= */

static sample_buffer_t geophone_buffer = {0};
static uint32_t total_events = 0;
static uint32_t high_confidence_events = 0;
static uint32_t critical_events = 0;
static bool system_ready = false;
static bool alert_silenced = false;


/* ========================================================================= */
/* HARDWARE INITIALIZATION                                                   */
/* ========================================================================= */

void adc_init_sm24(void) {
    // Initialize ADC
    adc_init();

    // Configure GPIO for analog input
    adc_gpio_init(SM24_ADC_PIN);

    // Select ADC channel
    adc_select_input(SM24_ADC_CHANNEL);

    printf("[ADC] SM-24 Geophone initialized on GPIO %d\n", SM24_ADC_PIN);
    printf("[ADC] Sensitivity: %.2f V/m/s\n", SM24_SENSITIVITY_V_MS);
    printf("[ADC] Frequency range: %d - %d Hz\n", SM24_FREQ_MIN_HZ, SM24_FREQ_MAX_HZ);
}

void gpio_init_all(void) {
    // LEDs
    gpio_init(LED_BUILTIN);
    gpio_set_dir(LED_BUILTIN, GPIO_OUT);
    gpio_put(LED_BUILTIN, 0);

    gpio_init(LED_STATUS);
    gpio_set_dir(LED_STATUS, GPIO_OUT);
    gpio_put(LED_STATUS, 0);

    gpio_init(LED_ALERT);
    gpio_set_dir(LED_ALERT, GPIO_OUT);
    gpio_put(LED_ALERT, 0);

    // Buzzer
    gpio_init(BUZZER_PIN);
    gpio_set_dir(BUZZER_PIN, GPIO_OUT);
    gpio_put(BUZZER_PIN, 0);

    // Button (with pull-up)
    gpio_init(BUTTON_PIN);
    gpio_set_dir(BUTTON_PIN, GPIO_IN);
    gpio_pull_up(BUTTON_PIN);

    printf("[GPIO] All pins initialized\n");
}


/* ========================================================================= */
/* LED & BUZZER CONTROL                                                      */
/* ========================================================================= */

void led_blink(uint8_t pin, int count, int duration_ms) {
    for (int i = 0; i < count; i++) {
        gpio_put(pin, 1);
        sleep_ms(duration_ms);
        gpio_put(pin, 0);
        sleep_ms(duration_ms);
    }
}

void buzzer_beep(int count, int duration_ms) {
    if (alert_silenced) return;

    for (int i = 0; i < count; i++) {
        gpio_put(BUZZER_PIN, 1);
        sleep_ms(duration_ms);
        gpio_put(BUZZER_PIN, 0);
        sleep_ms(duration_ms);
    }
}

void alert_low_confidence(void) {
    led_blink(LED_STATUS, 2, 200);
}

void alert_high_confidence(void) {
    led_blink(LED_ALERT, 5, 100);
    buzzer_beep(3, 150);
}

void alert_critical(void) {
    led_blink(LED_ALERT, 10, 50);
    buzzer_beep(5, 100);
    gpio_put(LED_ALERT, 1);  // Keep alert LED on
}


/* ========================================================================= */
/* SM-24 GEOPHONE DATA ACQUISITION                                          */
/* ========================================================================= */

uint32_t read_adc_averaged(void) {
    uint32_t sum = 0;

    for (int i = 0; i < ADC_SAMPLES; i++) {
        sum += adc_read();
    }

    return sum / ADC_SAMPLES;
}

float adc_to_voltage(uint32_t adc_value) {
    // RP2350 ADC is 12-bit (0-4095)
    return (adc_value * ADC_VREF) / 4095.0f;
}

float voltage_to_velocity_ms(float voltage) {
    // SM-24 outputs bipolar signal centered at Vcc/2 (1.65V)
    // Velocity = (Voltage - V_center) / Sensitivity
    float v_center = ADC_VREF / 2.0f;
    float v_signal = voltage - v_center;
    return v_signal / SM24_SENSITIVITY_V_MS;
}

void acquire_geophone_sample(geophone_sample_t *sample) {
    // Read ADC with averaging
    uint32_t adc_raw = read_adc_averaged();

    // Convert to voltage
    float voltage = adc_to_voltage(adc_raw);

    // Convert to velocity
    float velocity_m_s = voltage_to_velocity_ms(voltage);

    // Fill sample structure
    sample->raw_adc = adc_raw;
    sample->raw_voltage = voltage;
    sample->velocity_m_s = velocity_m_s;
    sample->velocity_mm_s = velocity_m_s * 1000.0f;
    sample->timestamp_ms = to_ms_since_boot(get_absolute_time());
}

void buffer_add_sample(float value) {
    geophone_buffer.buffer[geophone_buffer.index] = value;
    geophone_buffer.index++;

    if (geophone_buffer.index >= WINDOW_SIZE) {
        geophone_buffer.index = 0;
        geophone_buffer.filled = true;
    }
}


/* ========================================================================= */
/* EDGE IMPULSE INFERENCE (PLACEHOLDER)                                     */
/* ========================================================================= */

// This is a placeholder inference function.
// Replace this with your actual Edge Impulse exported model functions:
// - run_classifier()
// - ei_impulse_result_t
// - EI_IMPULSE_ERROR
//
// To integrate Edge Impulse:
// 1. Export your model as C++ library
// 2. Copy model files to project
// 3. Include edge-impulse-sdk headers
// 4. Replace this function with actual run_classifier() call

void run_inference(inference_result_t *result) {
    if (!geophone_buffer.filled) {
        strcpy(result->label, "insufficient_data");
        result->confidence = 0.0f;
        result->inference_time_ms = 0;
        return;
    }

    uint32_t start_time = to_ms_since_boot(get_absolute_time());

    // PLACEHOLDER: Simulate inference
    // In production, this would call your Edge Impulse model:
    //
    // signal_t signal;
    // signal.total_length = WINDOW_SIZE;
    // signal.get_data = &get_signal_data;
    // ei_impulse_result_t ei_result = {0};
    // EI_IMPULSE_ERROR res = run_classifier(&signal, &ei_result, false);

    // For demonstration: simple threshold-based classification
    float avg_amplitude = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        avg_amplitude += fabsf(geophone_buffer.buffer[i]);
    }
    avg_amplitude /= WINDOW_SIZE;

    // Simulate classification based on amplitude
    if (avg_amplitude > 0.05f) {
        strcpy(result->label, "earthquake");
        result->confidence = 0.85f + (avg_amplitude * 2.0f);  // Simulated
        if (result->confidence > 1.0f) result->confidence = 0.98f;
    } else if (avg_amplitude > 0.02f) {
        strcpy(result->label, "tremor");
        result->confidence = 0.70f + (avg_amplitude * 5.0f);
        if (result->confidence > 1.0f) result->confidence = 0.85f;
    } else {
        strcpy(result->label, "noise");
        result->confidence = 0.95f;
    }

    uint32_t end_time = to_ms_since_boot(get_absolute_time());
    result->inference_time_ms = end_time - start_time;
    result->timestamp_ms = end_time;
}


/* ========================================================================= */
/* EVENT PROCESSING                                                          */
/* ========================================================================= */

void process_inference_result(const inference_result_t *result) {
    // Skip noise detections
    if (strcmp(result->label, "noise") == 0) {
        return;
    }

    printf("\n╔═══════════════════════════════════════════════╗\n");
    printf("║     SEISMIC EVENT DETECTED                    ║\n");
    printf("╚═══════════════════════════════════════════════╝\n");
    printf("  Event Type: %s\n", result->label);
    printf("  Confidence: %.2f%%\n", result->confidence * 100.0f);
    printf("  Inference Time: %u ms\n", result->inference_time_ms);
    printf("  Timestamp: %llu ms\n", result->timestamp_ms);

    total_events++;

    // Determine alert level
    if (result->confidence >= 0.95f) {
        printf("\n  *** CRITICAL ALERT - VERY HIGH CONFIDENCE ***\n");
        alert_critical();
        critical_events++;

    } else if (result->confidence >= 0.85f) {
        printf("\n  *** HIGH CONFIDENCE ALERT ***\n");
        alert_high_confidence();
        high_confidence_events++;

    } else {
        printf("\n  [Low confidence detection]\n");
        alert_low_confidence();
    }

    printf("\n  Total Events: %u | High Conf: %u | Critical: %u\n",
           total_events, high_confidence_events, critical_events);
    printf("═══════════════════════════════════════════════\n\n");
}


/* ========================================================================= */
/* SYSTEM STATUS & MONITORING                                               */
/* ========================================================================= */

void print_system_status(void) {
    printf("\n┌───────────────────────────────────────────────┐\n");
    printf("│         SYSTEM STATUS                         │\n");
    printf("├───────────────────────────────────────────────┤\n");
    printf("│ Geophone: %s                          │\n", system_ready ? "✓ Active    " : "✗ Inactive  ");
    printf("│ Buffer: %s                              │\n", geophone_buffer.filled ? "✓ Ready     " : "○ Filling   ");
    printf("│ Events Detected: %-5u                      │\n", total_events);
    printf("│ High Confidence: %-5u                      │\n", high_confidence_events);
    printf("│ Critical Events: %-5u                      │\n", critical_events);
    printf("│ Alert: %s                               │\n", alert_silenced ? "Silenced    " : "Enabled     ");
    printf("└───────────────────────────────────────────────┘\n");
}

void check_button(void) {
    static uint32_t last_press = 0;
    uint32_t now = to_ms_since_boot(get_absolute_time());

    // Check button with debounce
    if (!gpio_get(BUTTON_PIN) && (now - last_press > 500)) {
        alert_silenced = !alert_silenced;
        gpio_put(LED_ALERT, 0);  // Turn off alert LED

        printf("\n[BUTTON] Alert %s\n", alert_silenced ? "SILENCED" : "ENABLED");
        led_blink(LED_BUILTIN, alert_silenced ? 1 : 3, 100);

        last_press = now;
    }
}


/* ========================================================================= */
/* MAIN APPLICATION                                                         */
/* ========================================================================= */

int main(void) {
    // Initialize standard I/O
    stdio_init_all();
    sleep_ms(2000);  // Wait for USB serial

    printf("\n\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║                                                       ║\n");
    printf("║    Raspberry Pi Pico 2 (RP2350A)                     ║\n");
    printf("║    Standalone Seismic Detection System               ║\n");
    printf("║    with SM-24 Geophone & Edge Impulse                ║\n");
    printf("║                                                       ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Initialize hardware
    printf("[System] Initializing hardware...\n");
    gpio_init_all();
    adc_init_sm24();

    printf("[System] Hardware initialization complete\n");
    led_blink(LED_BUILTIN, 3, 200);  // Startup blink pattern

    printf("\n[System] Starting data acquisition...\n");
    printf("[System] Sample Rate: %d Hz\n", SAMPLE_RATE_HZ);
    printf("[System] Window Size: %d samples\n", WINDOW_SIZE);
    printf("\n[System] Waiting for buffer to fill...\n");

    system_ready = true;

    uint32_t last_sample_time = 0;
    uint32_t last_inference_time = 0;
    uint32_t last_status_time = 0;
    uint32_t heartbeat_counter = 0;

    geophone_sample_t current_sample;
    inference_result_t inference;

    // Main loop
    while (1) {
        uint32_t now = to_ms_since_boot(get_absolute_time());

        // Data acquisition (every 10ms = 100 Hz)
        if (now - last_sample_time >= SAMPLE_PERIOD_MS) {
            acquire_geophone_sample(&current_sample);
            buffer_add_sample(current_sample.velocity_m_s);

            last_sample_time = now;
        }

        // Run inference every 2.56 seconds (256 samples at 100 Hz)
        if (geophone_buffer.filled && (now - last_inference_time >= 2560)) {
            run_inference(&inference);
            process_inference_result(&inference);

            last_inference_time = now;
        }

        // Print system status every 30 seconds
        if (now - last_status_time >= 30000) {
            print_system_status();
            last_status_time = now;
        }

        // Heartbeat LED (blink every 1 second when running)
        if (heartbeat_counter % 1000 == 0) {
            gpio_put(LED_BUILTIN, 1);
        }
        if (heartbeat_counter % 1000 == 100) {
            gpio_put(LED_BUILTIN, 0);
        }
        heartbeat_counter++;

        // Check user button
        check_button();

        sleep_ms(1);
    }

    return 0;
}
