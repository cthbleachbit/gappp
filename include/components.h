//
// Created by cth451 on 22-4-30.
//

#ifndef GAPPP_COMPONENTS
#define GAPPP_COMPONENTS

#include <rte_dev.h>
#include <rte_ethdev.h>
#include <future>
#include <random>

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <fmt/format.h>
#include "gappp_types.h"

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
#define GAPPP_BURST_RING_SIZE 128
// Burst - drain period in units of microseconds
#define GAPPP_BURST_TX_DRAIN_US 100
// ???
#define GAPPP_MEMPOOL_CACHE_SIZE 256
// Memory pool size
#define GAPPP_MEMPOOL_PACKETS ((1 << 16) - 1)
// Router Logging identifier
#define GAPPP_LOG_ROUTER "Router"

// Slots to reserve in the ring_tasks buffer
#define GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT 64
// Number of tasks to dequeue in one shot
#define GAPPP_GPU_HELM_TASK_BURST 4U
// Preallocate minion asynchronous results
#define GAPPP_GPU_FUTURE_PREALLOCATE 10
// GPU logging identifier
#define GAPPP_LOG_GPU_HELM "GPU Helm"

// FORWARD DECLARATION

namespace GAPPP {
	class Router;
	class GPUHelm;
}

namespace GAPPP {

	// DATA STRUCTURES
	struct router_thread_ident {
		uint16_t port;
		// the same queue id is used for both TX and RX queues
		uint16_t queue;
		struct hash {
			std::size_t operator()(const router_thread_ident &id) const {
				std::size_t h1 = std::hash<uint16_t>()(id.port);
				std::size_t h2 = std::hash<uint16_t>()(id.queue);
				return h1 + h2;
			}
		};

		inline std::strong_ordering operator<=>(const struct router_thread_ident &rhs) const {
			if (auto cmp = this->port <=> rhs.port; cmp != 0) {
				return cmp;
			}
			return this->queue <=> rhs.queue;
		};

		bool operator==(const struct router_thread_ident &rhs) const = default;
	} __rte_cache_aligned;

	// Per thread RX buffer
	struct router_thread_local_mbuf {
		uint16_t len;
		struct rte_mbuf *m_table[GAPPP_BURST_MAX_PACKET];
	};

	/**
	 * Router instance
	 *
	 * A router takes control over one or more ports, creates a handful of queues and spawn threads to poll the queues.
	 * When RX occur on a queue:
	 *    - DPDK allocates packet in local buffers and place pointers to packets in RX queue.
	 *    - workers dequeue and send pointers to GPUHelm
	 *    - GPUHelm spawns asynchronous minions, DMA packets to GPU memory, free the packets, launches CUDA kernel
	 * When GPU minions return:
	 *    - Minions call submit_tx() to schedule packet for delivery.
	 *    - FIXME: memory allocation?
	 */
	class Router {
	private:
		std::default_random_engine &rng_engine;
		GPUHelm *g = nullptr;
	public:
		// Set of ports. Use rte_eth_dev_info_get to obtain rte_eth_dev_info
		std::unordered_set<uint16_t> ports{};
		// Maps <port number, queue_id> to worker watching on
		std::unordered_map<router_thread_ident, decltype(rte_lcore_id()), router_thread_ident::hash> workers;
		// Allocate workers to CPUs as we go
		std::array<router_thread_ident, GAPPP_MAX_CPU> workers_affinity{};
		// Pointers to per-NIC packet buffers, can be made NUMA aware but assuming 1 socket here.
		// This should be allocated during router construction
		std::array<struct rte_mempool *, RTE_MAX_ETHPORTS> packet_memory_pool{};
		// Ring buffers per worker
		std::unordered_map<router_thread_ident, struct rte_ring *, router_thread_ident::hash> ring_tx;

		explicit Router(std::default_random_engine &rng_engine) noexcept;
		~Router() noexcept;

		/**
		 * initialize an ethernet device
		 * @param port_id       Port ID to set up
		 * @param mem_buf_pool  memory buffer to setup
		 * @return true if the device is initialized successfully and registered in the ports array
		 */
		bool dev_probe(uint16_t port_id) noexcept;

		/**
		 * Main thread event loop
		 * @param ident  thread identification
		 * @param stop   set to false to jump out of the loop
		 */
		void port_queue_event_loop(struct router_thread_ident ident, volatile bool *stop);

		/**
		 * Launch main event loop. This function will keep until "stop" is set to true
		 */
		void launch_threads(volatile bool *stop);

		/**
		 * Submit processed packets for transmission
		 * @param port_id  port to transmit on
		 * @param buf      contents to transmit
		 */
		unsigned int submit_tx(uint16_t port_id, struct router_thread_local_mbuf *buf);

		/**
		 * Submit processed packets for transmission
		 * @param port_id  port to transmit on
		 * @param len      number of packets
		 * @param packets  pointers to packet data to transmit
		 */
		unsigned int submit_tx(uint16_t port_id, size_t len, struct rte_mbuf *const *packets);

		/**
		 * Assign GPU helm to router - helm will be used to submit RX'd packets
		 * @param helm
		 */
		inline void assign_gpu_helm(GPUHelm *helm) noexcept {
			this->g = helm;
		}

	protected:
		/**
		 * Allocate packet memory buffer for a single port
		 * @param n_packet Number of packets to allocate for each port
		 * @param port     Port (NIC) to allocate memory for
		 */
		void allocate_packet_memory_buffer(unsigned int n_packet, uint16_t port);
	};

} // namespace GAPPP

// Formatter for thread identifier
template<>
struct fmt::formatter<GAPPP::router_thread_ident> {
	constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
		auto it = ctx.begin(), end = ctx.end();
		if (it != end && *it != '}') throw format_error("invalid format");
		return it;
	}

	template<typename FormatContext>
	auto format(const GAPPP::router_thread_ident &id, FormatContext &ctx) -> decltype(ctx.out()) {
		// ctx.out() is an output iterator to write to.
		return format_to(ctx.out(), "eth{}/worker{}", id.port, id.queue);
	}
};

namespace GAPPP {

	class GPUHelm {
		// Incoming buffers - CPU workers will submit_rx tasks to this ring buffer
		// Note that this buffer is multi-producer/single-consumer.
		struct rte_ring *ring_tasks = nullptr;
		// GPU threads will place finished data here
		// Note tha this buffer is multi-producer(running)/multi-consumer(passed directly to NIC workers)
		struct rte_ring *ring_completion = nullptr;
		// Outstanding GPU threads
		std::vector<std::shared_future<int>> running;

		// Pointer to initialized router instance - ownership is borrowed (i.e. not to be freed)
		Router *r = nullptr;

		// Routing table for sanity check
		routing_table routes{};

		// Entry point to module invocation
		GAPPP::cuda_module_t &module_invoke;

	public:
		/**
		 * Construct a GPU helm and allocate associated message ring buffers
		 */
		GPUHelm(GAPPP::cuda_module_t &module_invoke, const std::string &path_route_table);

		/**
		 * Free the associated data structures
		 */
		~GPUHelm();

		/**
		 * Submit tasks to the GPU Helm
		 * @param thread_id identity of running thread
		 * @param task      incoming packet to process
		 *                  the packet buffer are allocated from within CPU worker from the memory pool
		 *                  ownership is transferred to GPU-helm - will be freed after DMA to GPU.
		 * @return    0 if submission was successful
		 */
		unsigned int submit_rx(router_thread_ident thread_id, router_thread_local_mbuf *task);

		/**
		 * Submit tasks to the GPU Helm
		 * @param thread_id identity of running thread
		 * @param len       number of packets to enqueue
		 * @param task      incoming packet to process
		 *                  the packet buffer are allocated from within CPU worker from the memory pool
		 *                  ownership is transferred to GPU-helm - will be freed after DMA to GPU.
		 * @return    0   if submission was successful
		 *            NUM if NUM packets were left out due to space issues
		 */
		unsigned int submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task);

		/**
		 * GPU main event loop
		 * @param stop     terminate when stop is true
		 */
		void gpu_helm_event_loop(const volatile bool *stop);

		void inline assign_router(Router *router) {
			this->r = router;
		}

	private:
		/**
		 * Launch a GPU asynchronous task to process nbr_tasks packets, referred as pointers in packets.
		 * @param nbr_tasks
		 * @param packets     Note: needs to be freed after use [transfer full]
		 * @return
		 */
		int
		gpu_minion_thread(unsigned int nbr_tasks, std::array<struct rte_mbuf *, GAPPP_GPU_HELM_TASK_BURST> *packets);
	};

}

#endif //COMPONENTS_H
