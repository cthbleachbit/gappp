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

#include "gappp_types.h"
#include "l3fwd.h"



namespace GAPPP {
	namespace l3fwd {
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


		__global__ void routing_table_modify(routing_table t){
				int i = threadIdx.x;
				t[i].network = table[i].network;
				t[i].mask = table[i].mask;
				t[i].gateway = table[i].gateway;
				t[i].out_port = table[i].out_port;
		}

		int setTable(routing_table t){

			cudaMemcpyToSymbol(routetable, &t, sizeof(t), size_t(0),cudaMemcpyHostToDevice);

			
			//cudaMalloc((void **)&gpu, sizeof(t));//for gpu struct
			
			//routing_table_modify<<<1, t.size()>>>(*gpu);

			//cudaMemcpy(&ret, gpu, sizeof(t), cudaMemcpyDeviceToHost);

			return 0;
		}

	}
}