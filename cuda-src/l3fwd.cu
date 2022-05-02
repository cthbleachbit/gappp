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

#include "gappp_types.h"
#include "l3fwd.h"
#include "../include/gappp_types.h"


namespace GAPPP {
	namespace l3fwd {



		__managed__ int numroutes = 0;
		__managed__ struct route routes[4096];
		__global__ void prefixMatchPacket(struct rte_mbuf *m,  uint32_t **masks){
			int i = threadIdx.x;
			
			route r = routes[i];
			/* Remove the Ethernet header and trailer from the input packet */
   			 rte_pktmbuf_adj(m, (uint16_t)sizeof(struct rte_ether_hdr));
    			
			/* if this is an IPv4 packet */
       			struct rte_ipv4_hdr *ip_hdr;
       			uint32_t ip_dst;
        		
			/* Read the lookup key (i.e. ip_dst) from the input packet */
        		ip_hdr = rte_pktmbuf_mtod(m, struct rte_ipv4_hdr *);
        		ip_dst = rte_be_to_cpu_32(ip_hdr->dst_addr);

			uint32_t b0,b1,b2,b3;
			uint32_t res;

			b0 = (ip_dst & 0x000000ff) << 24u;
			b1 = (ip_dst & 0x0000ff00) << 8u;
			b2 = (ip_dst & 0x00ff0000) >> 8u;
			b3 = (ip_dst & 0xff000000) >> 24u;

			res = b0 | b1 | b2 | b3;
			
			if(res & r.mask == r.network){
				masks[i] = r.mask;
			}

			
		}

		__global__ void prefixMatch(struct rte_mbuf **packets){
			int i = threadIdx.x;
			rte_mbuf packet = packets[i];
			uint32_t masks[numroutes];
			
			prefixMatchPacket<<<1,numroutes>>>(packet, masks);
			uint32_t max_mask = 0;
			int max_ind = -1;
			for(int j = 0; j<numroutes:j++){
				
				uint32_t cur = masks[j];

				if(cur > max_mask){
					max_mask = cur;
					max_ind = j;

			}	

			
			packets[i]->port = routes[max_ind].out_port;
		}



		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
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
