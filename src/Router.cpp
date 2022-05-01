//
// Created by cth451 on 22-4-17.
//

#include "Logging.h"
#include "components.h"

#include <fmt/format.h>
#include <random>

namespace GAPPP {
	static struct rte_eth_conf port_conf = {
		.rxmode = {
			.mq_mode = ETH_MQ_RX_NONE,
			.split_hdr_size = 0,
		},
		.txmode = {
			.mq_mode = ETH_MQ_TX_NONE,
		},
		.rx_adv_conf = {
			.rss_conf = {
				.rss_key = nullptr,
				.rss_hf = 0,
			},
		},
	};

	bool Router::dev_probe(uint16_t port_id) noexcept {
		int ret_val;
		uint16_t nb_rxd = GAPPP_DEFAULT_RX_DESC;
		uint16_t nb_txd = GAPPP_DEFAULT_TX_DESC;
		struct rte_eth_dev_info dev_info{};
		struct rte_eth_rxconf rxq_conf{};
		struct rte_eth_txconf txq_conf{};
		auto local_port_conf = port_conf;

		if (!rte_eth_dev_is_valid_port(port_id))
			whine(Severity::CRIT, fmt::format("Invalid port ID {}", port_id), GAPPP_LOG_ROUTER);

		ret_val = rte_eth_dev_info_get(port_id, &dev_info);
		if (ret_val != 0)
			whine(Severity::CRIT, fmt::format("Error during getting device (port {}) info: {}\n",
			                                  port_id, strerror(-ret_val)), GAPPP_LOG_ROUTER);

		// Turn on fast free if supported
		if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
			local_port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;

		local_port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;

		ret_val = rte_eth_dev_configure(port_id, GAPPP_DEFAULT_RX_QUEUE, GAPPP_DEFAULT_TX_QUEUE, &local_port_conf);
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
		// FIXME: initialize multiple RX queue if needed
		rxq_conf = dev_info.default_rxconf;
		rxq_conf.offloads = local_port_conf.rxmode.offloads;
		ret_val = rte_eth_rx_queue_setup(port_id, 0, nb_rxd,
		                                 rte_eth_dev_socket_id(port_id),
		                                 &rxq_conf,
		                                 this->packet_memory_pool[port_id]);
		if (ret_val < 0)
			whine(Severity::CRIT,
			      fmt::format("port {}: RX queue 0 setup failed (res={})", port_id, ret_val),
			      GAPPP_LOG_ROUTER);

		// TX queue setup
		// FIXME: initialize multiple TX queue if needed
		txq_conf = dev_info.default_txconf;
		txq_conf.offloads = local_port_conf.txmode.offloads;
		ret_val = rte_eth_tx_queue_setup(port_id, 0, nb_txd,
		                                 rte_eth_dev_socket_id(port_id),
		                                 &txq_conf);
		if (ret_val < 0)
			whine(Severity::CRIT,
			      fmt::format("port {}: TX queue 0 setup failed (res={})", port_id, ret_val),
			      GAPPP_LOG_ROUTER);

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
		this->ports.emplace(port_id);
		return true;
	}

	void Router::launch_threads(volatile bool *stop) {
		// TODO: create and start threads
		// TODO: bind threads to cores
		// TODO: block on keyboard input
		// TODO: request threads to quit
	}

	static inline int
	send_burst(struct router_thread_local_mbuf *buf, uint16_t port, uint16_t queueid) {
		int ret;

		ret = rte_eth_tx_burst(port, queueid, buf->m_table, buf->len);
		if (unlikely(ret < buf->len)) {
			do {
				rte_pktmbuf_free(buf->m_table[ret]);
			}
			while (++ret < buf->len);
		}

		return 0;
	}

	void Router::port_queue_event_loop(struct router_thread_ident ident,
	                                   struct router_thread_local_mbuf *buf,
	                                   volatile bool *stop) {
		std::array<struct rte_mbuf *, GAPPP_BURST_MAX_PACKET> tx_burst{};
		std::array<struct rte_mbuf *, GAPPP_BURST_MAX_PACKET> rx_burst{};

		unsigned int lcore_id;
		uint64_t prev_tsc, diff_tsc, cur_tsc;
		int i, nb_rx;
		unsigned int nb_tx;
		unsigned int ret;
		uint8_t portid, queueid;
		const uint64_t drain_tsc = (rte_get_tsc_hz() + US_PER_S - 1) /
			US_PER_S * GAPPP_BURST_TX_DRAIN_US;

		prev_tsc = 0;
		lcore_id = rte_lcore_id();

		whine(Severity::INFO,
		      fmt::format("Entering main loop: lcore={}, port={}, rx queue={}", lcore_id, ident.port, ident.queue));

		while (!*stop) {
			cur_tsc = rte_rdtsc();

			/*
			 * TX burst queue drain
			 */
			diff_tsc = cur_tsc - prev_tsc;
			if (unlikely(diff_tsc > drain_tsc)) {
				send_burst(buf, buf->len, portid);
				buf->len = 0;
				prev_tsc = cur_tsc;
			}

			/*
			 * Read packet from RX queues
			 */
			nb_rx = rte_eth_rx_burst(portid, queueid, rx_burst.data(),
			                         GAPPP_BURST_MAX_PACKET);
			if (nb_rx != 0) {
				this->g->submit_rx(ident, nb_rx, rx_burst.data());
				// CPU only implementation at https://github.com/ceph/dpdk/blob/master/examples/l3fwd/l3fwd_lpm_sse.h
			}

			// Handle TX
			nb_tx = rte_ring_dequeue_burst(this->ring_tx.at(ident),
			                               reinterpret_cast<void **>(tx_burst.data()),
			                               GAPPP_BURST_MAX_PACKET,
			                               nullptr);
			if (nb_tx > 0) {
				ret = rte_eth_tx_burst(portid, queueid, tx_burst.data(), nb_tx);
				if (ret < nb_tx) {
					whine(Severity::WARN,
					      fmt::format("Worker {} submitted {} packets for TX but only {} were sent", ident, nb_tx, ret),
					      GAPPP_LOG_ROUTER);
				}
			}
		}

		whine(Severity::INFO, "Terminating", GAPPP_LOG_ROUTER);
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
		std::uniform_int_distribution<uint16_t> dist(1, GAPPP_DEFAULT_TX_QUEUE);
		uint16_t queue = dist(this->rng_engine);
		struct rte_ring *tx_ring = nullptr;
		struct router_thread_ident id{port_id, queue};
		try {
			tx_ring = this->ring_tx.at(id);
		}
		catch (std::out_of_range &e) {
			whine(Severity::CRIT, fmt::format("No TX buffer allocated for {}", id), GAPPP_LOG_ROUTER);
		};
		unsigned int ret = rte_ring_enqueue_burst(tx_ring, reinterpret_cast<void *const *>(packets), len, nullptr);
		if (ret < len) {
			whine(Severity::WARN,
			      fmt::format("TX buffer {}: {} enqueue requested > {} enqueued", id, len, ret),
			      GAPPP_LOG_ROUTER);
		}
		return len - ret;
	}

	Router::Router(std::default_random_engine &rng_engine) noexcept
		:
		rng_engine(rng_engine) {}

} // GAPPP