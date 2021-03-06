//
// Created by cth451 on 22-4-17.
//

#include "Logging.h"
#include "components.h"

#include <fmt/format.h>
#include <sys/prctl.h>
#include <rte_eal.h>
#include <random>
#include <rte_ethdev.h>
#include <rte_malloc.h>

#ifndef ETH_MQ_RX_NONE
#define RTE_ETH_MQ_RX_NONE ETH_MQ_RX_NONE
#endif
#ifndef ETH_MQ_TX_NONE
#define RTE_ETH_MQ_TX_NONE ETH_MQ_TX_NONE
#endif
#ifndef ETH_MQ_RX_DCB_RSS
#define RTE_ETH_MQ_RX_DCB_RSS ETH_MQ_RX_DCB_RSS
#endif
#ifndef ETH_MQ_TX_DCB
#define RTE_ETH_MQ_TX_DCB ETH_MQ_TX_DCB
#endif

#ifndef RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE
#define RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE DEV_TX_OFFLOAD_MBUF_FAST_FREE
#endif

namespace GAPPP {
	static struct rte_eth_conf port_conf = {
		.rxmode = {
			.mq_mode = RTE_ETH_MQ_RX_NONE,
			.split_hdr_size = 0,
		},
		.txmode = {
			.mq_mode = RTE_ETH_MQ_TX_NONE,
		},
		.rx_adv_conf = {
			.rss_conf = {
				.rss_key = nullptr,
				.rss_hf = 0,
			},
		},
	};

	bool Router::dev_probe(uint16_t port_id, uint16_t n_queue) noexcept {
		int ret_val;
		uint16_t nb_rxd = GAPPP_DEFAULT_RX_DESC;
		uint16_t nb_txd = GAPPP_DEFAULT_TX_DESC;
		struct rte_eth_dev_info dev_info{};
		struct rte_eth_rxconf rxq_conf{};
		struct rte_eth_txconf txq_conf{};
		auto local_port_conf = port_conf;
		if (n_queue > 1) {
			local_port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_DCB_RSS;
			local_port_conf.txmode.mq_mode = RTE_ETH_MQ_TX_DCB;
		}

		if (!rte_eth_dev_is_valid_port(port_id))
			whine(Severity::CRIT, fmt::format("Invalid port ID {}", port_id), GAPPP_LOG_ROUTER);

		ret_val = rte_eth_dev_info_get(port_id, &dev_info);
		if (ret_val != 0)
			whine(Severity::CRIT, fmt::format("Error during getting device (port {}) info: {}\n",
			                                  port_id, strerror(-ret_val)), GAPPP_LOG_ROUTER);

		// Turn on fast free if supported
		if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE)
			local_port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;

		local_port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;

		ret_val = rte_eth_dev_configure(port_id, n_queue, n_queue, &local_port_conf);
		if (ret_val != 0)
			whine(Severity::CRIT, fmt::format("port {}: configuration failed (res={})\n",
			                                  port_id, ret_val), GAPPP_LOG_ROUTER);

		ret_val = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rxd, &nb_txd);
		if (ret_val != 0)
			whine(Severity::CRIT, fmt::format("port (): rte_eth_dev_adjust_nb_rx_tx_desc failed (res={})\n",
			                                  port_id,
			                                  ret_val), GAPPP_LOG_ROUTER);

		// Allocate packet buffer for this NIC
		allocate_packet_memory_buffer(GAPPP_MEMPOOL_PACKETS, port_id);

		// RX Queue setup
		rxq_conf = dev_info.default_rxconf;
		rxq_conf.offloads = local_port_conf.rxmode.offloads;
		for (int i = 0; i < n_queue; i++) {
			ret_val = rte_eth_rx_queue_setup(port_id, i, nb_rxd,
			                                 rte_eth_dev_socket_id(port_id),
			                                 &rxq_conf,
			                                 this->packet_memory_pool[port_id]);
			if (ret_val < 0)
				whine(Severity::CRIT,
				      fmt::format("port {}: RX queue {} setup failed : {}", port_id, i, ret_val, rte_strerror(ret_val)),
				      GAPPP_LOG_ROUTER);
		}


		// TX queue setup
		txq_conf = dev_info.default_txconf;
		txq_conf.offloads = local_port_conf.txmode.offloads;
		for (int i = 0; i < n_queue; i++) {
			ret_val = rte_eth_tx_queue_setup(port_id, i, nb_txd,
			                                 rte_eth_dev_socket_id(port_id),
			                                 &txq_conf);
			if (ret_val < 0)
				whine(Severity::CRIT,
				      fmt::format("port {}: TX queue {} setup failed: {}", port_id, i, rte_strerror(ret_val)),
				      GAPPP_LOG_ROUTER);
		}

		// Allocate TX ring buffers for workers
		for (uint16_t i = 0; i < n_queue; i++) {
			struct router_thread_ident ident{port_id, i};
			if (likely(!this->ring_tx.contains(ident))) {
				std::string name = fmt::format("TX ring {}", ident);
				this->ring_tx[ident] = rte_ring_create(name.c_str(), GAPPP_BURST_RING_SIZE, 0, RING_F_SC_DEQ);
				if (unlikely(!this->ring_tx[ident])) {
					whine(Severity::CRIT,
					      fmt::format("TX ring for worker {} allocation failed with {} - {}",
					                  ident,
					                  rte_errno,
					                  rte_strerror(rte_errno)),
					      GAPPP_LOG_ROUTER);
				} else {
					whine(Severity::INFO,
					      fmt::format("TX ring for worker {} allocated", ident),
					      GAPPP_LOG_ROUTER);
				}
			}
		}

		// Start the port
		ret_val = rte_eth_dev_start(port_id);
		if (ret_val < 0)
			whine(Severity::CRIT, fmt::format("Start port {} failed (res={})", port_id, ret_val), GAPPP_LOG_ROUTER);

		// Go into promiscuous mode to catch all incoming packets
		rte_eth_promiscuous_enable(port_id);

		struct rte_ether_addr addr{};
		ret_val = rte_eth_macaddr_get(port_id, &addr);
		if (ret_val != 0)
			whine(Severity::CRIT,
			      fmt::format("Mac address get port {} failed (res={})", port_id, ret_val),
			      GAPPP_LOG_ROUTER);

		whine(Severity::INFO,
		      fmt::format("Port {} MAC: {}", port_id, mac_addr_to_string(addr.addr_bytes)),
		      GAPPP_LOG_ROUTER);
		this->ports[port_id] = n_queue;
		return true;
	}

	bool Router::dev_stop(uint16_t port_id) {
		whine(Severity::INFO, fmt::format("Stopping port {}", port_id), GAPPP_LOG_ROUTER);
		if (!this->ports.contains(port_id)) {
			whine(Severity::WARN, fmt::format("Attempting to stop nonexistent port {}", port_id), GAPPP_LOG_ROUTER);
			return false;
		}
		rte_eth_dev_stop(port_id);
		for (uint16_t i = 0; i < this->ports[port_id]; i++) {
			rte_ring_free(this->ring_tx[{port_id, i}]);
		}
		rte_eth_dev_close(port_id);

		whine(Severity::INFO, fmt::format("Releasing memory pool for {}", port_id), GAPPP_LOG_ROUTER);
#ifdef GAPPP_GPU_DIRECT
		if (use_gpu_direct) {
			struct rte_eth_dev_info eth_info{};
			struct rte_pktmbuf_extmem &local_external_mem = this->external_mem[port_id];
			rte_eth_dev_info_get(port_id, &eth_info);
			rte_dev_dma_unmap(eth_info.device,
			                  local_external_mem.buf_ptr,
			                  local_external_mem.buf_iova,
			                  local_external_mem.buf_len);
			((GPUDirectHelm *) g)->unregister_ext_mem(local_external_mem);
			this->external_mem.erase(port_id);
		} else {
			rte_mempool_free(packet_memory_pool[port_id]);
		}
#else
		rte_mempool_free(packet_memory_pool[port_id]);
#endif
		packet_memory_pool[port_id] = nullptr;
		this->ports.erase(port_id);
		return true;
	}

	struct router_worker_launch_parameter {
		Router *r;
		volatile bool *stop;
		struct router_thread_ident id;
	};

	static int router_rx_worker_launch(void *ptr) {
		auto parameters = (router_worker_launch_parameter *) (ptr);
		parameters->r->port_queue_rx_loop(parameters->id, parameters->stop);
		return 0;
	}

	static int router_tx_worker_launch(void *ptr) {
		auto parameters = (router_worker_launch_parameter *) (ptr);
		parameters->r->port_queue_tx_loop(parameters->id, parameters->stop);
		return 0;
	}

	void Router::launch_threads(volatile bool *stop) {
		static unsigned int worker_id_base = 0;
		// Spawn workers
		prctl(PR_SET_NAME, "Router Master");
		for (auto port: this->ports) {
			for (uint16_t i = 0; i < port.second; i++) {
				struct router_worker_launch_parameter param{.r = this, .stop = stop, .id = {port.first, i}};
				worker_id_base = rte_get_next_lcore(worker_id_base, 1, 0);
				unsigned int rx_worker_id = worker_id_base;
				worker_id_base = rte_get_next_lcore(worker_id_base, 1, 0);
				unsigned int tx_worker_id = worker_id_base;
				rte_eal_remote_launch(router_rx_worker_launch, &param, rx_worker_id);
				this->rx_workers[param.id] = rx_worker_id;
				rte_eal_remote_launch(router_tx_worker_launch, &param, tx_worker_id);
				this->tx_workers[param.id] = tx_worker_id;
			}
		}

		// Wait for workers to exit
		for (auto worker: this->rx_workers) {
			rte_eal_wait_lcore(worker.second);
		}
		for (auto worker: this->tx_workers) {
			rte_eal_wait_lcore(worker.second);
		}

		this->tx_workers.clear();
		this->rx_workers.clear();
	}

	void Router::port_queue_tx_loop(struct router_thread_ident ident, volatile bool *stop) {
		std::array<struct rte_mbuf *, GAPPP_BURST_MAX_PACKET> tx_burst{};

		unsigned int lcore_id;
		uint64_t prev_tsc, diff_tsc, cur_tsc;
		unsigned int nb_tx;
		unsigned int ret;
		uint8_t portid = ident.port;
		uint8_t queueid = ident.queue;
		const uint64_t drain_tsc = (rte_get_tsc_hz() + US_PER_S - 1) /
			US_PER_S * GAPPP_BURST_TX_DRAIN_US;

		prev_tsc = 0;
		lcore_id = rte_lcore_id();

		// Check if queues are ready
		if (!this->ring_tx.contains(ident)) {
			whine(Severity::CRIT, fmt::format("TX Worker {} - TX queue isn't initialized", ident), GAPPP_LOG_ROUTER);
		}

		whine(Severity::INFO,
		      fmt::format("TX Worker {} - Entering main loop @ lcore={}", ident, lcore_id), GAPPP_LOG_ROUTER);

		prctl(PR_SET_NAME, fmt::format("Router TX Worker {}", ident).c_str());

		while (!*stop) {
			/*
			 * TX burst queue drain
			 */
			cur_tsc = rte_rdtsc();
			diff_tsc = cur_tsc - prev_tsc;
			if (unlikely(diff_tsc > drain_tsc)) {
				nb_tx = rte_ring_dequeue_burst(this->ring_tx.at(ident),
				                               reinterpret_cast<void **>(tx_burst.data()),
				                               GAPPP_BURST_MAX_PACKET,
				                               nullptr);
				if (nb_tx > 0) {
					ret = rte_eth_tx_burst(portid, queueid, tx_burst.data(), nb_tx);
					if (unlikely(ret < nb_tx)) {
						whine(Severity::WARN,
						      fmt::format("TX Worker {} submitted {} packets for TX but only {} were sent",
						                  ident,
						                  nb_tx,
						                  ret),
						      GAPPP_LOG_ROUTER);
						do {
							rte_pktmbuf_free(tx_burst[ret]);
						}
						while (++ret < nb_tx);
					}
				}
				prev_tsc = cur_tsc;
			}
		}

		whine(Severity::INFO, fmt::format("TX Worker {} - Terminating", ident), GAPPP_LOG_ROUTER);
	}

	void Router::port_queue_rx_loop(struct router_thread_ident ident, volatile bool *stop) {
		std::array<struct rte_mbuf *, GAPPP_BURST_MAX_PACKET> rx_burst{};

		unsigned int lcore_id;
		int nb_rx;
		uint8_t portid = ident.port;
		uint8_t queueid = ident.queue;

		lcore_id = rte_lcore_id();

		whine(Severity::INFO,
		      fmt::format("RX Worker {} - Entering main loop @ lcore={}", ident, lcore_id), GAPPP_LOG_ROUTER);

		prctl(PR_SET_NAME, fmt::format("Router RX Worker {}", ident).c_str());

		while (!*stop) {
			nb_rx = rte_eth_rx_burst(portid, queueid, rx_burst.data(),
			                         GAPPP_BURST_MAX_PACKET);
			if (nb_rx != 0) {
				this->g->submit_rx(ident, nb_rx, rx_burst.data());
			}
		}

		whine(Severity::INFO, fmt::format("RX Worker {} - Terminating", ident), GAPPP_LOG_ROUTER);
	}

	void Router::allocate_packet_memory_buffer(unsigned int n_packet, uint16_t port) {
		if (this->packet_memory_pool[port] == nullptr) {
			struct rte_eth_dev_info eth_info{};
			std::string pool_name = fmt::format("Packet memory pool/{}", port);
			if (use_gpu_direct) {
#ifdef GAPPP_GPU_DIRECT
				// Allocate external memory for gpu direct
				std::string zone_name = fmt::format("Shared DMA zone/{}", port);
				struct rte_pktmbuf_extmem local_external_mem{};
				local_external_mem.elt_size = GAPPP_DIRECT_MBUF_DATAROOM + RTE_PKTMBUF_HEADROOM;
				local_external_mem.buf_len =
					RTE_ALIGN_CEIL(n_packet * local_external_mem.elt_size, GAPPP_GPU_PAGE_SIZE);
				local_external_mem.buf_ptr =
					rte_malloc(zone_name.c_str(), local_external_mem.buf_len, GAPPP_GPU_PAGE_SIZE);
				if (local_external_mem.buf_ptr == nullptr) {
					whine(Severity::CRIT,
					      fmt::format("External allocation for {} failed", pool_name),
					      GAPPP_LOG_ROUTER);
				}

				// Register external buffer memory region for GPU DMA
				rte_eth_dev_info_get(port, &eth_info);
				((GPUDirectHelm *) g)->register_ext_mem(local_external_mem);

				// Register external buffer memory region for NIC DMA
				int ret = rte_dev_dma_map(eth_info.device,
				                          local_external_mem.buf_ptr,
				                          local_external_mem.buf_iova,
				                          local_external_mem.buf_len);
				if (ret < 0) {
					whine(Severity::CRIT, "Failed to register DMA zone with NIC", GAPPP_LOG_ROUTER);
				} else {
					whine(Severity::INFO, "Registered DMA zone with NIC", GAPPP_LOG_ROUTER);
				}

				this->external_mem.emplace(port, local_external_mem);

				// Allocate pktmbuf wrapper
				this->packet_memory_pool[port] = rte_pktmbuf_pool_create_extbuf(pool_name.c_str(),
				                                                                n_packet,
				                                                                0,
				                                                                0,
				                                                                local_external_mem.elt_size,
				                                                                GAPPP_DEFAULT_SOCKET_ID,
				                                                                &local_external_mem,
				                                                                1);
#else
				whine(Severity::CRIT, "Unreachable code path", GAPPP_LOG_ROUTER);
#endif
			} else {
				// Allocate normal memory pool
				this->packet_memory_pool[port] =
					rte_pktmbuf_pool_create(pool_name.c_str(),
					                        n_packet,
					                        GAPPP_MEMPOOL_CACHE_SIZE,
					                        0,
					                        RTE_MBUF_DEFAULT_BUF_SIZE,
					                        0);
			}
			if (this->packet_memory_pool[port] == nullptr) {
				whine(Severity::CRIT, fmt::format("Allocation for {} failed", pool_name), GAPPP_LOG_ROUTER);
			} else {
				whine(Severity::INFO, fmt::format("Allocation for {} completed", pool_name), GAPPP_LOG_ROUTER);
			}
		}
	}

	unsigned int Router::submit_tx(uint16_t port_id, struct router_thread_local_mbuf *buf) {
		return submit_tx(port_id, buf->len, buf->m_table);
	}

	unsigned int Router::submit_tx(uint16_t port_id, size_t len, struct rte_mbuf *const *packets) {
		std::uniform_int_distribution<uint16_t> dist(0, this->ports[port_id] - 1);
		uint16_t queue = dist(this->rng_engine);
		struct rte_ring *tx_ring;
		struct router_thread_ident id{port_id, queue};
		try {
			tx_ring = this->ring_tx.at(id);
		}
		catch (std::out_of_range &e) {
			for (int i = 0; i < len; i++) {
				rte_pktmbuf_free(packets[i]);
			}
			whine(Severity::WARN, fmt::format("No TX buffer allocated for {}", id), GAPPP_LOG_ROUTER);
			return 0;
		};
		unsigned int ret = rte_ring_enqueue_burst(tx_ring, reinterpret_cast<void *const *>(packets), len, nullptr);
		if (ret < len) {
			whine(Severity::WARN,
			      fmt::format("TX buffer {}: {} enqueue requested > {} enqueued", id, len, ret),
			      GAPPP_LOG_ROUTER);
		} else {
			whine(Severity::INFO,
			      fmt::format("TX buffer {}: {} enqueued", id, len),
			      GAPPP_LOG_ROUTER);
		}
		return len - ret;
	}

	Router::Router(std::default_random_engine &rng_engine) noexcept
		:
		rng_engine(rng_engine) {}

	Router::~Router() noexcept {

	}

	void Router::assign_gpu_helm(GPUHelmBase *helm) noexcept {
		this->g = helm;
		use_gpu_direct = helm->is_direct();
	}
} // GAPPP