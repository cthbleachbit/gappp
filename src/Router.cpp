//
// Created by cth451 on 22-4-17.
//

#include "Router.h"
#include "Logging.h"

#include <fmt/format.h>

// Number of transmit descriptors
#define GAPPP_DEFAULT_TX_DESC 10
// Number of receive descriptors
#define GAPPP_DEFAULT_RX_DESC 10
// Socket ID
#define GAPPP_DEFAULT_SOCKET_ID 0

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

		ret_val = rte_eth_dev_configure(port_id, 1, 1, &local_port_conf);
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
		return true;
	}
} // GAPPP