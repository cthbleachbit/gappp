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

#include <unordered_set>
#include <unordered_map>
#include <memory>

#define GAPPP_MAX_CPU 64

namespace GAPPP {

	class Router {
	public:
		// Workers are identified with the port and queue id they tend to
		struct thread_ident {
			uint16_t port;
			uint16_t rx_queue;
			struct thread_ident_hash
			{
				std::size_t operator() (const thread_ident &id) const
				{
					std::size_t h1 = std::hash<uint16_t>()(id.port);
					std::size_t h2 = std::hash<uint16_t>()(id.rx_queue);
					return h1 + h2;
				}
			};
		};
		// Set of ports. Use rte_eth_dev_info_get to obtain rte_eth_dev_info
		std::unordered_set<uint16_t> ports {};
		// Maps <port number, queue_id> to worker watching on
		std::unordered_map<thread_ident, std::shared_ptr<std::thread>, thread_ident::thread_ident_hash> workers;
		// Allocate workers to CPUs as we go
		std::array<thread_ident, GAPPP_MAX_CPU> workers_affinity;

		Router() = default;

		/**
		 * initialize an ethernet device
		 * @param port_id       Port ID to set up
		 * @param mem_buf_pool  memory buffer to setup
		 * @return true if the device is initialized successfully and registered in the ports array
		 */
		bool dev_probe(uint16_t port_id, struct rte_mempool &mem_buf_pool) noexcept;

		/**
		 * Launch main event loop. This function will keep until ENTER key is pressed
		 */
		void launch_threads();
	};


} // GAPPP

#endif //ROUTER_H
