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

#include <memory>

namespace GAPPP {

	class Router {
	protected:
		std::unique_ptr<struct rte_eth_dev[]> nic_list;
	public:
		Router() = default;
		/**
		 * initialize an ethernet device
		 * @param port_id       Port ID to set up
		 * @param mem_buf_pool  memory buffer to setup
		 * @return
		 */
		bool dev_probe(uint16_t port_id, struct rte_mempool &mem_buf_pool) noexcept;
	};

} // GAPPP

#endif //ROUTER_H
