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

#include "types.h"

namespace GAPPP {
	namespace l3fwd {
		__global__ void VecAdd(float *A, float *B, float *C) {
			int i = threadIdx.x;
			C[i] = A[i] + B[i];
		}

		int invoke() {
			float a[4] = {1.0f, 3.0f, 5.0f, 7.0f};
			float b[4] = {1.0f, 3.0f, 5.0f, 7.0f};;
			float c[4];
			VecAdd<<<1, 4>>>(a, b, c);
		}
	}
}