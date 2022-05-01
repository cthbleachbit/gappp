#ifndef __global__
#define __global__
#endif

#include "gappp_types.h"
#include "l3fwd.h"

namespace GAPPP {
	namespace selftest {
		__global__ void VecAdd(int32_t *A, int32_t *B, int32_t *C) {
			int i = threadIdx.x;
			C[i] = A[i] + B[i];
		}

		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			int32_t a[4] = {1, 4, 16, 64};
			int32_t b[4] = {2, 8, 32, 128};;
			int32_t c[4];
			VecAdd<<<1, 4>>>(a, b, c);
			return c[0] + c[1] + c[2] + c[3];
		}
	}
}