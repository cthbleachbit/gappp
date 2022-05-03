//
// Created by cth451 on 22-4-17.
//

#include "Logging.h"
#include "components.h"

#include <fmt/format.h>
#include <sys/prctl.h>
#include <random>
#include <rte_ethdev.h>

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

	struct router_worker_launch_parameter {
		Router *r;
		volatile bool *stop;
		struct router_thread_ident id;
	};

	static int router_worker_launch(void *ptr) {
		auto parameters = (router_worker_launch_parameter *) (ptr);
		parameters->r->port_queue_event_loop(parameters->id, parameters->stop);
		return 0;
	}

	void Router::launch_threads(volatile bool *stop) {
		static int worker_id_offset = 6;
		// Spawn workers
		prctl(PR_SET_NAME, "Router Master");
		for (auto port: this->ports) {
			for (uint16_t i = 0; i < port.second; i++) {
				struct router_worker_launch_parameter param{.r = this, .stop = stop, .id = {port.first, i}};
				int worker_number = worker_id_offset;
				worker_id_offset++;
				rte_eal_remote_launch(router_worker_launch, &param, worker_number);
				this->workers[param.id] = worker_number;
			}
		}

		// Wait for workers to exit
		for (auto worker: this->workers) {
			rte_eal_wait_lcore(worker.second);
			this->workers.erase(worker.first);
		}
	}

	static inline int
	send_burst(struct router_thread_local_mbuf *buf, uint16_t port, uint16_t queueid) {
		int ret;

		ret = rte_eth_tx_burst(port, queueid, buf->m_table, buf->len);
		if (unlikely(ret < buf->len)) {
			do {
				rte_pktmbuf_free(buf->m_table[ret]);
			} while (++ret < buf->len);
		}

		return 0;
	}

	void Router::port_queue_event_loop(struct router_thread_ident ident, volatile bool *stop) {
		std::array<struct rte_mbuf *, GAPPP_BURST_MAX_PACKET> tx_burst{};
		std::array<struct rte_mbuf *, GAPPP_BURST_MAX_PACKET> rx_burst{};

		unsigned int lcore_id;
		uint64_t prev_tsc, diff_tsc, cur_tsc;
		int i, nb_rx;
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
			whine(Severity::CRIT, fmt::format("Worker {} - TX queue isn't initialized", ident), GAPPP_LOG_ROUTER);
		}

		whine(Severity::INFO,
		      fmt::format("Worker {} - Entering main loop @ lcore={}", ident, lcore_id), GAPPP_LOG_ROUTER);

		prctl(PR_SET_NAME, fmt::format("Router Worker {}", ident).c_str());

		while (!*stop) {
			/*
			 * Read packet from RX queues
			 */
			nb_rx = rte_eth_rx_burst(portid, queueid, rx_burst.data(),
			                         GAPPP_BURST_MAX_PACKET);
			if (nb_rx != 0) {
				this->g->submit_rx(ident, nb_rx, rx_burst.data());
				// CPU only implementation at https://github.com/ceph/dpdk/blob/master/examples/l3fwd/l3fwd_lpm_sse.h
			}

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
						      fmt::format("Worker {} submitted {} packets for TX but only {} were sent",
						                  ident,
						                  nb_tx,
						                  ret),
						      GAPPP_LOG_ROUTER);
						do {
							rte_pktmbuf_free(tx_burst[ret]);
						} while (++ret < nb_tx);
					}
				}
				prev_tsc = cur_tsc;
			}
		}

		whine(Severity::INFO, fmt::format("Worker {} - Terminating", ident), GAPPP_LOG_ROUTER);
	}

	void Router::allocate_packet_memory_buffer(unsigned int n_packet, uint16_t port) {
		if (this->packet_memory_pool[port] == nullptr) {
			std::string pool_name = fmt::format("Packet memory pool/{}", port);
			this->packet_memory_pool[port] =
					rte_pktmbuf_pool_create(pool_name.c_str(),
					                        n_packet,
					                        GAPPP_MEMPOOL_CACHE_SIZE,
					                        0,
					                        RTE_MBUF_DEFAULT_BUF_SIZE,
					                        0);
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
		struct rte_ring *tx_ring = nullptr;
		struct router_thread_ident id{port_id, queue};
		try {
			tx_ring = this->ring_tx.at(id);
		}
		catch (std::out_of_range &e) {
			for (int i = 0; i < len; i++) {
				rte_pktmbuf_free(packets[i]);
			}
			whine(Severity::WARN, fmt::format("No TX buffer allocated for {}", id), GAPPP_LOG_ROUTER);
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
		int ret;
		// Deallocate ring buffers
		for (auto port: this->ports) {
			whine(Severity::INFO, fmt::format("Stopping port {}", port.first), GAPPP_LOG_ROUTER);
			rte_eth_dev_stop(port.first);
			for (uint16_t i = 0; i < port.second; i++) {
				rte_ring_free(this->ring_tx[{port.first, i}]);
			}
		}

		// Clean up all devices and memory pools
		for (auto port: this->ports) {
			whine(Severity::INFO, fmt::format("Releasing memory pool for {}", port.first), GAPPP_LOG_ROUTER);
			rte_mempool_free(packet_memory_pool[port.first]);
		}
	}
} // GAPPP