# Graphics accelerated packet processing pipeline

Inspired by [PacketShader](https://dl.acm.org/doi/10.1145/1851275.1851207), this project aims to create a software router with GPU assist, where the entire pipeline should rest in the userspace.

# Assumptions

* NICs supported by [DPDK](https://github.com/DPDK/dpdk) userspace polling mode driver
* CUDA capable Nvidia GPU
* C++ 20 / GNU++ 20 compatible compiler (i.e. gcc 11)
* `fmtlib` (might gets embedded as a submodule)
* Relatively modern CMake (3.16+), ninja, meson (required by dpdk)

# Components

* DPDK middleware - the portion that interfaces with the NIC
* CUDA kernels - where the main packet processing take place on GPU

# Goals?

* A ethernet switch?
* A IPv6 router?
* A IPSec tunneling endpoint?

# Building

* On older Linux distros, you may want to define compiler version by exporting `CC=/path/to/gcc` and `CXX=/path/to/g++` before proceeding.
* Make sure all submodules are up-to-date:
  - `git submodule update --init --recursive`
* Check where `cuda.h` and `cudaTypedefs.h` - on my computer this is in `/usr/lib/cuda/targets/x86_64-linux/include/`.
* Build DPDK - get into `dpdk` first:
  - `meson <-DFLAG=value> -Dc_args=-I${CUDA_HEADER_DIR} build` (put values there if you need to)
  - `ninja -C build`
* Return to root dir:
  - `mkdir build`
  - `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja ..`
  - `ninja`
