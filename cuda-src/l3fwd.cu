//
// Created by cth451 on 4/19/22.
//

/*
 * Quick reminders:
 *
 * __global__ = may be called from host or gpu
 * __host__ = run on host, call from host
 * __device__ = run on gpu , call from gpu
 */

#ifndef __global__
#define __global__
#endif

#include <cstdio>
#include <rte_ip.h>
#include <rte_ether.h>
#include <rte_mbuf.h>

#include "gappp_types.h"
#include "l3fwd.h"
#include "Logging.h"

#define GAPPP_L3FWD_MAX_ROUTES 4096

namespace GAPPP {
	namespace l3fwd {


		__managed__ int numroutes = 0;
		__managed__ struct route routes[GAPPP_L3FWD_MAX_ROUTES];

		__device__ inline char *gappp_l3fwd_pktmbuf_adj(struct rte_mbuf *m, uint16_t len)
		{
			__rte_mbuf_sanity_check(m, 1);

			if (unlikely(len > m->data_len))
				return nullptr;

			/* NB: elaborating the addition like this instead of using
			 *     += allows us to ensure the result type is uint16_t
			 *     avoiding compiler warnings on gcc 8.1 at least */
			m->data_len = (uint16_t)(m->data_len - len);
			m->data_off = (uint16_t)(m->data_off + len);
			m->pkt_len  = (m->pkt_len - len);
			return (char *)m->buf_addr + m->data_off;
		}

		__global__ void prefixMatch(struct rte_mbuf **packets) {
			int i = threadIdx.x;
			rte_mbuf* packet = packets[i];

			uint32_t max_mask = 0;
			int max_ind = -1;
			gappp_l3fwd_pktmbuf_adj(packet, (uint16_t) sizeof(struct rte_ether_hdr));

			/* if this is an IPv4 packet */
			struct rte_ipv4_hdr *ip_hdr;
			uint32_t ip_dst;

			/* Read the lookup key (i.e. ip_dst) from the input packet */
			ip_hdr = rte_pktmbuf_mtod(packet, struct rte_ipv4_hdr *);
			ip_dst = ip_hdr->dst_addr;

			uint32_t b0, b1, b2, b3;

			b0 = (ip_dst & 0x000000ff) << 24u;
			b1 = (ip_dst & 0x0000ff00) << 8u;
			b2 = (ip_dst & 0x00ff0000) >> 8u;
			b3 = (ip_dst & 0xff000000) >> 24u;

			ip_dst = b0 | b1 | b2 | b3;

			for (int j = 0; j < numroutes; j++) {
				if ((ip_dst & routes[j].mask) == routes[j].network) {
					if (routes[j].mask > max_mask) {
						max_mask = routes[j].mask;
						max_ind = j;
					}
				}
			}

			packet->port = routes[max_ind].out_port;
		}


		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			whine(Severity::INFO, fmt::format("L3FWD kernel invocation with {}", nbr_tasks), "L3FWD");
			prefixMatch<<<1, nbr_tasks >>>(packets);
			return 0;
		}


		__global__ void readTable() {
			int i = threadIdx.x;
			printf("%u %u %u %u\n", routes[i].network, routes[i].mask, routes[i].gateway, routes[i].out_port);
		}

		int setTable(GAPPP::routing_table t) {


			numroutes = t.size();
			cudaMemcpy(routes, t.data(), sizeof(route) * t.size(), cudaMemcpyHostToDevice);

			readTable<<<1, numroutes>>>();


			return 0;
		}

	}
}
