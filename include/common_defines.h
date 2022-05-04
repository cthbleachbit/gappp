//
// Created by cth451 on 22-5-4.
//

#ifndef GAPPP_COMMON_DEFINES_H
#define GAPPP_COMMON_DEFINES_H

// Number of transmit descriptors
#define GAPPP_DEFAULT_TX_DESC (1 << 7)
// Number of receive descriptors
#define GAPPP_DEFAULT_RX_DESC (1 << 4)
// Router worker threads count
#define GAPPP_ROUTER_THREADS_PER_PORT 1
// Number of transmit queue
#define GAPPP_DEFAULT_TX_QUEUE GAPPP_ROUTER_THREADS_PER_PORT
// Number of receive queue
#define GAPPP_DEFAULT_RX_QUEUE GAPPP_ROUTER_THREADS_PER_PORT
// Socket ID
#define GAPPP_DEFAULT_SOCKET_ID 0
// Maximum number of cores i.e. threads
#define GAPPP_MAX_CPU 64
// Burst - batching packet TX
#define GAPPP_BURST_MAX_PACKET 32
// Burst - total ring size
#define GAPPP_BURST_RING_SIZE 1024
// Burst - drain period in units of microseconds
#define GAPPP_BURST_TX_DRAIN_US 100
// ???
#define GAPPP_MEMPOOL_CACHE_SIZE 256
// Memory pool size
#define GAPPP_MEMPOOL_PACKETS ((1 << 16) - 1)
// Router Logging identifier
#define GAPPP_LOG_ROUTER "Router"


#define GAPPP_DIRECT_MBUF_DATAROOM 2048
#define GAPPP_GPU_PAGE_SIZE 4096

#define GAPPP_GPU_ID 0
// Slots to reserve in the ring_tasks buffer
#define GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT 4096
// Number of tasks to dequeue in one shot
#define GAPPP_GPU_HELM_TASK_BURST 64U
// Preallocate minion asynchronous results
#define GAPPP_GPU_FUTURE_PREALLOCATE 10
// GPU logging identifier
#define GAPPP_LOG_GPU_HELM "GPU Helm"
// GPU logging identifier
#define GAPPP_LOG_GPU_DIRECT_HELM "GPU Direct Helm"

#endif //COMMON_DEFINES_H
