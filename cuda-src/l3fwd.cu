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

		__global__ void VecAdd(float *A, float *B, float *C) {
			int i = threadIdx.x;
			C[i] = A[i] + B[i];
		}

		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			float a[4] = {1.0f, 3.0f, 5.0f, 7.0f};
			float b[4] = {1.0f, 3.0f, 5.0f, 7.0f};
			float c[4];
			VecAdd<<<1, 4>>>(a, b, c);
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
