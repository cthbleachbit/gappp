# Router abstractions

- Each router has a list of ports and auto-negotiate speed upon router activation or port state change.
- Each port will have a number of TX/RX queues assigned to multiple CPU workers. Each worker gets notified when incoming traffic needs to be handled, preprocess the packets and notifies the GPU helm to launch kernels.
- Each worker also gets notified when GPU computation has returned. Packets will be delivered to NIC output queue by each individual worker threads.
- Ideally each CPU worker needs to be pinned to a CPU core to reduce cache misses.

# Interaction with GPU kernels

### Calling into GPU

A CPU worker needs to provide the following to the GPU helm:

- Memory location of preprocessed packets + metadata
- A callback entry point that interrupts current execution flow and hands results into NIC TX queue

### Returning from GPU

GPU helm provides memory location of processed packets + metadata, and instructs the CPU worker to send packets off.

### Memory passing between CPU worker and GPU

As is discussed in the PacketShader, GPU helm should avoid touching memory region containing packets to avoid cache pollution.
Current plan is to use dpdk provided lock-free ring buffer implementation to pass contents in to / out of GPU.