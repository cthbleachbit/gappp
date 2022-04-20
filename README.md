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
* Return to root dir:
  - `mkdir build`
  - `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja ..`
  - `ninja`
