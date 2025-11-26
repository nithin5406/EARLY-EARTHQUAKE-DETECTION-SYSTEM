## SM-24 Geophone Earthquake Detection on Raspberry Pi Pico 2 (RP2350)

## Project overview
This project turns a Raspberry Pi Pico 2 (RP2350) and an SM-24 geophone into a standalone, low-power seismic node that can detect earthquakes and distinguish them from everyday vibrations using Edge Impulse. The model is trained in Edge Impulse and deployed as a C++ library running fully on-device, with real-time inference on streaming geophone data.

The system is designed for:

* Early detection of seismic activity using a professional geophone (SM-24).

* Fully offline inference on the RP2350 microcontroller.

* Clear, multi-level alerts via LEDs and a buzzer for different confidence levels.
  
## Requirements

### Hardware

* [Raspberry Pi Pico](https://www.raspberrypi.org/products/raspberry-pi-pico/).
* [Geophone- SM-24, with insulating disc](https://www.sparkfun.com/geophone-sm-24-with-insulating-disc.html).
* [INA333 Low-Power, Zero-Drift, Precision Instrumentation Amplifier](https://www.ti.com/product/INA333).

### Software

* [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/cli-installation).
* [GNU ARM Embedded Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads).
* [CMake](https://cmake.org/install/).
* Rasperry Pi Pico SDK:
   ```bash
   git clone -b master https://github.com/raspberrypi/pico-sdk.git
   cd pico-sdk
   git submodule update --init
   export PATH="<Path to Pico SDK>"
   ```

