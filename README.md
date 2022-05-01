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
  - `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja -DCUDA_TOOLKIT_ROOT_DIR=<path_to_cuda_toolkit>..`
  - `ninja`

### Private Testbed Notes

Use the following cmake parameters:

```
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.6 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.6/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  ..
```

# Command line arguments

### Parameters accepted
- EAL parameter: `-w bus:dev.func` PCIe device address of the NIC to take over
- Use `--` to separate DPDK EAL options from program specific options
- `--ip a.b.c.d/cidr` Assign this IP address to the card
- `--route routing_table_file` Load routing table from file

### Routing table format:
- `a.b.c.d/cidr (via <gateway>) dev <port_id>`

Looks like this:

```
0.0.0.0/0 via 24.178.40.1 dev 0
24.178.40.0/22 dev 0
172.16.0.0/24 dev 1
192.168.1.0/24 dev 1
```

To test on testbeds, use something like `sudo ./gappp -w 41:00.0 -- --ip 10.0.0.1/24 --route simple-routes`

To inject traffic, use a virtual device: `--vdev 'net_pcap0,rx_pcap=input.pcap,tx_pcap=output.pcap'` as part of DPDK EAL parameters.