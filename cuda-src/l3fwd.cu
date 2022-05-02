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
		__global__ void prefixMatch(struct rte_mbuf **packets){
			int i = threadIdx.x;
			int out_port = -1;
			// TODO: processing here
			// TODO: lookup the table, find longest match
			packets[i]->port = out_port;
		}
		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			prefixMatch<<<1, nbr_tasks>>>(packets);
			return 0;
		}


		__global__ void readTable() {
			int i = threadIdx.x;
			printf("%u %u %u %u\n", routes[i].network, routes[i].mask, routes[i].gateway, routes[i].out_port);
		}

		int setTable(GAPPP::routing_table t) {

			//cudaMemcpyToSymbol(routetable, &t, sizeof(t), size_t(0),cudaMemcpyHostToDevice);

			numroutes = t.size();
			cudaMemcpy(routes, t.data(), sizeof(route) * t.size(), cudaMemcpyHostToDevice);

			readTable<<<1, numroutes>>>();

			//cudaMemcpy(&ret, gpu, sizeof(t), cudaMemcpyDeviceToHost);

			return 0;
		}

	}
}
