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
#ifdef GAPPP_GPU_DIRECT
#include <rte_gpudev.h>
#include <cuda_runtime.h>
#endif
#include "gappp_types.h"
#include "common_defines.h"

// FORWARD DECLARATION

namespace GAPPP {
	class Router;
	class GPUHelmBase;
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
		GPUHelmBase *g = nullptr;
		bool use_gpu_direct = false;
		// Only used for GPU direct: External memory zone
		std::unordered_map<uint16_t, struct rte_pktmbuf_extmem> external_mem{};
	public:
		// Set of ports. Use rte_eth_dev_info_get to obtain rte_eth_dev_info, maps to number of queues
		std::unordered_map<uint16_t, uint16_t> ports{};
		// Maps <port number, queue_id> to worker watching on
		std::unordered_map<router_thread_ident, decltype(rte_lcore_id()), router_thread_ident::hash> tx_workers;
		std::unordered_map<router_thread_ident, decltype(rte_lcore_id()), router_thread_ident::hash> rx_workers;
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
		bool dev_probe(uint16_t port_id, uint16_t n_queue) noexcept;

		/**
		 * Stop and remove a port from the router
		 */
		bool dev_stop(uint16_t port_id);

		/** Main thread event loop - RX only
		 * @param ident
		 * @param stop
		 */
		void port_queue_rx_loop(struct router_thread_ident ident, volatile bool *stop);

		/** Main thread event loop - TX only
		 * @param ident
		 * @param stop
		 */
		void port_queue_tx_loop(struct router_thread_ident ident, volatile bool *stop);

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
		void assign_gpu_helm(GPUHelmBase *helm) noexcept;

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
	/**
	 * GPU Helm base class for all the API GPU helm implementations should provide
	 */
	class GPUHelmBase {
	public:
		/**
		 * Construct a GPU helm and allocate associated message ring buffers
		 */
		GPUHelmBase() = default;

		/**
		 * Free the associated data structures
		 */
		virtual ~GPUHelmBase() = default;

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
		virtual unsigned int submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task) = 0;

		/**
		 * GPU main event loop
		 * @param stop     terminate when stop is true
		 */
		virtual void gpu_helm_event_loop(const volatile bool *stop) {};

		virtual void assign_router(Router *router) {};

		/**
		 * @return whether the instance is using gpu direct
		 */
		virtual bool is_direct() const noexcept = 0;
	};

	/**
	 * Original GPU Helm - very inefficient
	 */
	class GPUHelm: public GPUHelmBase {
		// Incoming buffers - CPU workers will submit_rx tasks to this ring buffer
		// Note that this buffer is multi-producer/single-consumer.
		struct rte_ring *ring_tasks = nullptr;
		// Outstanding GPU threads
		std::vector<std::shared_future<int>> running;

		// Pointer to initialized router instance - ownership is borrowed (i.e. not to be freed)
		Router *r = nullptr;

		// Routing table for sanity check
		routing_table routes{};

		// Entry point to module invocation
		GAPPP::cuda_module_invoke_t &module_invoke;

	public:
		/**
		 * Construct a GPU helm and allocate associated message ring buffers
		 */
		GPUHelm(GAPPP::cuda_module_invoke_t &module_invoke,
		        GAPPP::cuda_module_init_t &module_init,
		        const std::string &path_route_table);

		/**
		 * Free the associated data structures
		 */
		~GPUHelm() override;

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
		unsigned int submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task) override;

		/**
		 * GPU main event loop
		 * @param stop     terminate when stop is true
		 */
		void gpu_helm_event_loop(const volatile bool *stop) override;

		void inline assign_router(Router *router) override {
			this->r = router;
		}

		bool is_direct() const noexcept override {
			return false;
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


#ifdef GAPPP_GPU_DIRECT
	/**
	 * A GPU helm using GPU direct
	 */
	class GPUDirectHelm: public GPUHelmBase {
		// Outstanding GPU threads
		std::vector<std::shared_future<int>> running;

		// Pointer to initialized router instance - ownership is borrowed (i.e. not to be freed)
		Router *r = nullptr;

		// Routing table for sanity check
		routing_table routes{};

		// Entry point to module invocation
		GAPPP::cuda_module_invoke_t &module_invoke;

		// reference: rte_gpu_comm_list is a list of pointer of tasks - why this instead of pktmbuf?
		std::array<struct rte_gpu_comm_list *, GAPPP_DIRECT_MAX_PERSISTENT_KERNEL> gpu_comm_lists{};

		std::array<cudaStream_t, GAPPP_DIRECT_MAX_PERSISTENT_KERNEL> gpu_streams{};

	public:
		/**
		 * Construct a GPU helm and allocate associated message ring buffers
		 */
		GPUDirectHelm(GAPPP::cuda_module_invoke_t &module_invoke,
		              GAPPP::cuda_module_init_t &module_init,
		              const std::string &path_route_table);

		/**
		 * Free the associated data structures
		 */
		~GPUDirectHelm() override;

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
		unsigned int submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task) override;

		/**
		 * GPU main event loop
		 * @param stop     terminate when stop is true
		 */
		void gpu_helm_event_loop(const volatile bool *stop) override;

		void inline assign_router(Router *router) override {
			this->r = router;
		}

		bool is_direct() const noexcept override {
			return true;
		}

		/**
		 * Register Allocated external memory pool with GPU for DMA
		 * @param external_mem
		 * @return
		 */
		int register_ext_mem(const struct rte_pktmbuf_extmem &external_mem);

		/**
		 * Unregister Allocated external memory pool with GPU for DMA
		 * @param external_mem
		 * @return
		 */
		int unregister_ext_mem(const struct rte_pktmbuf_extmem &external_mem);
	};
#endif
}

#endif //COMPONENTS_H
