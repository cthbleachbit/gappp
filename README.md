# Graphics accelerated packet processing pipeline

Inspired by [PacketShader](https://dl.acm.org/doi/10.1145/1851275.1851207), this project aims to create a software router with GPU assist, where the entire pipeline should rest in the userspace.

# Assumptions

* NICs supported by [DPDK](https://github.com/DPDK/dpdk) userspace polling mode driver
* CUDA capable Nvidia GPU
* C++ 20 / GNU++ 20 compatible compiler (i.e. gcc 11)
* `fmtlib` (might gets embedded as a submodule)
* Relatively modern CMake (3.16+), ninja, meson (required by dpdk)

# Components

* `src` DPDK middleware - the portion that interfaces with the NIC
* `cuda-src` CUDA kernels - where the main packet processing take place on GPU

# Goals?

* A ethernet switch?
* A IPv6 router?
* A in-network firewall?

# Building

* On older Linux distros, you may want to define compiler version by exporting `CC=/path/to/gcc` and `CXX=/path/to/g++` before proceeding.
* Make sure all submodules are up-to-date:
  - `git submodule update --init --recursive`
* Return to root dir:
  - `mkdir build`
  - `cd build`
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
  -DCMAKE_PREFIX_PATH=/opt/gappp/dpdk/ \
  ..
```

# Command line arguments

### Parameters accepted
- EAL parameter: `-w bus:dev.func` PCIe device address of the NIC/GPU to take over
- Use `--` to separate DPDK EAL options from program specific options
- `-m|--module [l3fwd|dummy]` Use this CUDA module
- `-r|--route routing_table_file` Load routing table from file
- `-n|--num-ports X` Declare number of ports available
- `-p|--port-queue id:nq` Allocate `nq` TX and `nq` RX queues on port `id` (optional, default to 4. Note that some port drivers don't support MQ)

### Routing table format:
- `a.b.c.d/cidr (via <gateway>) dev <port_id>`

Looks like this:

```
0.0.0.0/0 via 24.178.40.1 dev 0
24.178.40.0/22 dev 0
172.16.0.0/24 dev 1
192.168.1.0/24 dev 1
```

To test on testbeds, use something like `sudo ./gappp -w 41:00.0 -- --module l3fwd --route /path/to/simple-routes`

To inject traffic, use a virtual device: `--vdev 'net_pcap0,rx_pcap=input.pcap,tx_pcap=output.pcap'` as part of DPDK EAL parameters.

To test with routing table in `test-inputs/test-routes`, start main program with the following arguments and send traffic to 192.168.0.{10,20,30,40}.

```
sudo ./gappp -w 41:00.0 \
  --vdev 'net_pcap1,tx_pcap=net_pcap1.pcap' \
  --vdev 'net_pcap2,tx_pcap=net_pcap2.pcap' \
  --vdev 'net_pcap3,tx_pcap=net_pcap3.pcap' \
  --vdev 'net_pcap4,tx_pcap=net_pcap4.pcap' \
  -- \
  --module l3fwd --route /path/to/test-routes \
  -n 5 -p 0:4 -p 1:1 -p 2:1 -p 3:1 -p 4:1
```

# Building DPDK on testbed for GPU direct

Assuming `v22.03`, export the following environment variable:

- `CFLAGS=-I/usr/local/cuda-11.6/include`
- `CC=/usr/bin/gcc-11`
- `CXX=/usr/bin/g++-11`

For meson configuration, use

- `-Dprefix=/opt/gappp/dpdk`
- `-Db_lto=true`