//
// Created by cth451 on 22-4-17.
//

#include "Router.h"
#include "Logging.h"

#include <fmt/format.h>

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

	bool Router::dev_probe(uint16_t port_id, struct rte_mempool &mem_buf_pool) noexcept {
		int ret_val;
		uint16_t nb_rxd = GAPPP_DEFAULT_RX_DESC;
		uint16_t nb_txd = GAPPP_DEFAULT_TX_DESC;
		struct rte_eth_dev_info dev_info;
		struct rte_eth_rxconf rxq_conf;
		struct rte_eth_txconf txq_conf;
		auto local_port_conf = port_conf;

		if (!rte_eth_dev_is_valid_port(port_id))
			throw std::runtime_error(fmt::format("Invalid port ID {}", port_id));

		ret_val = rte_eth_dev_info_get(port_id, &dev_info);
		if (ret_val != 0)
			throw std::runtime_error(fmt::format("Error during getting device (port {}) info: {}\n",
			                                     port_id, strerror(-ret_val)));

		// Turn on fast free if supported
		if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
			local_port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;

		local_port_conf.rx_adv_conf.rss_conf.rss_hf &= dev_info.flow_type_rss_offloads;

		ret_val = rte_eth_dev_configure(port_id, GAPPP_DEFAULT_RX_QUEUE, GAPPP_DEFAULT_TX_QUEUE, &local_port_conf);
		if (ret_val != 0)
			throw std::runtime_error(fmt::format("port {}: configuration failed (res={})\n",
			                                     port_id, ret_val));

		ret_val = rte_eth_dev_adjust_nb_rx_tx_desc(port_id, &nb_rxd, &nb_txd);
		if (ret_val != 0)
			throw std::runtime_error(fmt::format("port (): rte_eth_dev_adjust_nb_rx_tx_desc failed (res={})\n",
			                                     port_id,
			                                     ret_val));

		// RX Queue setup
		// FIXME: initialize multiple RX queue if needed
		rxq_conf = dev_info.default_rxconf;
		rxq_conf.offloads = local_port_conf.rxmode.offloads;
		ret_val = rte_eth_rx_queue_setup(port_id, 0, nb_rxd,
		                                 rte_eth_dev_socket_id(port_id),
		                                 &rxq_conf,
		                                 &mem_buf_pool);
		if (ret_val < 0)
			throw std::runtime_error(fmt::format("port {}: RX queue 0 setup failed (res={})", port_id, ret_val));

		// TX queue setup
		// FIXME: initialize multiple TX queue if needed
		txq_conf = dev_info.default_txconf;
		txq_conf.offloads = local_port_conf.txmode.offloads;
		ret_val = rte_eth_tx_queue_setup(port_id, 0, nb_txd,
		                                 rte_eth_dev_socket_id(port_id),
		                                 &txq_conf);
		if (ret_val < 0)
			throw std::runtime_error(fmt::format("port {}: TX queue 0 setup failed (res={})", port_id, ret_val));

		// Start the port
		ret_val = rte_eth_dev_start(port_id);
		if (ret_val < 0)
			throw std::runtime_error(fmt::format("Start port {} failed (res={})", port_id, ret_val));

		struct rte_ether_addr addr{};
		ret_val = rte_eth_macaddr_get(port_id, &addr);
		if (ret_val != 0)
			throw std::runtime_error(fmt::format("Mac address get port {} failed (res={})", port_id, ret_val));

		whine(Severity::INFO, fmt::format("Port {} MAC: {}", port_id, mac_addr_to_string(addr.addr_bytes)));
		this->ports.emplace(port_id);
		return true;
	}

	void Router::launch_threads() {
		// TODO: create and start threads
		// TODO: bind threads to cores
		// TODO: block on keyboard input
		// TODO: request threads to quit
	}

	static inline int
	send_burst(struct Router::thread_local_mbuf *buf, uint16_t port, uint16_t queueid) {
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

	void Router::port_queue_event_loop(Router::thread_ident ident, struct Router::thread_local_mbuf *buf, bool *stop) {
		struct rte_mbuf *pkts_burst[GAPPP_BURST_MAX_PACKET];
		unsigned int lcore_id;
		uint64_t prev_tsc, diff_tsc, cur_tsc;
		int i, nb_rx;
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
			nb_rx = rte_eth_rx_burst(portid, queueid, pkts_burst,
			                         GAPPP_BURST_MAX_PACKET);
			if (nb_rx == 0)
				continue;

			// FIXME: this doesn't exist yet
			// FIXME: tell GPU we have something to do
			// CPU only implementation at https://github.com/ceph/dpdk/blob/master/examples/l3fwd/l3fwd_lpm_sse.h
		}
	}
} // GAPPP