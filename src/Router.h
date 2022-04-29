//
// Created by cth451 on 22-4-17.
//

/**
 * Abstraction for a router node.
 *
 * Useful links to dpdk: https://doc.dpdk.org/api/rte__ethdev_8h.html
 */

#ifndef ROUTER_H
#define ROUTER_H

#include <rte_dev.h>
#include <rte_ethdev.h>
#include <thread>
#include <random>

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <fmt/format.h>

// Number of transmit descriptors
#define GAPPP_DEFAULT_TX_DESC (1 << 4)
// Number of receive descriptors
#define GAPPP_DEFAULT_RX_DESC (1 << 4)
// Number of transmit queue
#define GAPPP_DEFAULT_TX_QUEUE 1
// Number of receive queue
#define GAPPP_DEFAULT_RX_QUEUE 1
// Socket ID
#define GAPPP_DEFAULT_SOCKET_ID 0
// Maximum number of cores i.e. threads
#define GAPPP_MAX_CPU 64
// Burst - batching packet TX
#define GAPPP_BURST_MAX_PACKET 32
// Burst - drain period in units of microseconds
#define GAPPP_BURST_TX_DRAIN_US 100
// ???
#define GAPPP_MEMPOOL_CACHE_SIZE 256
// Memory pool size
#define GAPPP_MEMPOOL_PACKETS ((1 << 16) - 1)
// Logging identifier
#define GAPPP_LOG_ROUTER "Router"

namespace GAPPP {

	// Forward declaration
	class GPUHelm;

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
			if (auto cmp = this->port <=> rhs.port; cmp !=0) {
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
		std::default_random_engine& rng_engine;
		GPUHelm *g;
	public:
		// Set of ports. Use rte_eth_dev_info_get to obtain rte_eth_dev_info
		std::unordered_set<uint16_t> ports{};
		// Maps <port number, queue_id> to worker watching on
		std::unordered_map<router_thread_ident, std::shared_ptr<std::thread>, router_thread_ident::hash> workers;
		// Allocate workers to CPUs as we go
		std::array<router_thread_ident, GAPPP_MAX_CPU> workers_affinity{};
		// Pointers to per-NIC packet buffers, can be made NUMA aware but assuming 1 socket here.
		// This should be allocated during router construction
		std::array<struct rte_mempool *, RTE_MAX_ETHPORTS> packet_memory_pool{};
		// Ring buffers per worker
		std::unordered_map<router_thread_ident, struct rte_ring*, router_thread_ident::hash> ring_tx;

		explicit Router(std::default_random_engine& rng_engine) noexcept;
		~Router() = default;

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
		 * @param buf    a pointer array to track pending TX
		 * @param stop   set to false to jump out of the loop
		 */
		void port_queue_event_loop(struct router_thread_ident ident,
		                           struct router_thread_local_mbuf *buf,
		                           volatile bool *stop);

		/**
		 * Launch main event loop. This function will keep until "stop" is set to true
		 */
		void launch_threads(volatile bool *stop);

		/**
		 * Submit processed packets for transmission
		 * @param port_id
		 * @param buf
		 */
		void submit_tx(uint16_t port_id, struct router_thread_local_mbuf *buf);

		/**
		 * Assign GPU helm to router
		 * @param helm
		 */
		void set_gpu_helm(GPUHelm *helm) noexcept;

	protected:
		/**
		 * Allocate packet memory buffer for port
		 * @param n_packet Number of packets to allocate for each port
		 * @param port     Port (NIC) to allocate memory for
		 */
		void allocate_packet_memory_buffer(unsigned int n_packet, uint16_t port);
	};

} // GAPPP

template <> struct fmt::formatter<GAPPP::router_thread_ident> {
	constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
		auto it = ctx.begin(), end = ctx.end();
		if (it != end && *it != '}') throw format_error("invalid format");
		return it;
	}

	template <typename FormatContext>
	auto format(const GAPPP::router_thread_ident& id, FormatContext& ctx) -> decltype(ctx.out()) {
		// ctx.out() is an output iterator to write to.
		return format_to(ctx.out(), "eth{}/worker{}", id.port, id.queue);
	}
};

#endif //ROUTER_H
