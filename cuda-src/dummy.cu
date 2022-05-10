#include "dummy.h"
#include "Logging.h"
#include <fmt/format.h>
#include <cstdio>

#ifndef __global__
#define __global__
#endif

#define GAPPP_LOG_DUMMY "GPU Dummy"

namespace GAPPP {
	namespace dummy {
		__global__ void debug_print(unsigned int nbr_tasks, struct rte_mbuf *packets) {
			printf("\n\n\nGPU/Dummy: Incoming port %u\n\n\n", packets[0].port);
		}

		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			whine(Severity::INFO,
			      fmt::format("Spawning minions for {} packets", nbr_tasks),
			      GAPPP_LOG_DUMMY);
			struct rte_mbuf *gpu;
			cudaMalloc(&gpu, nbr_tasks * sizeof(struct rte_mbuf));
			for (int i = 0; i < nbr_tasks; i++) {
				cudaMemcpy((void *) (gpu + i), packets[i], sizeof(struct rte_mbuf), cudaMemcpyHostToDevice);
			}
			cudaDeviceSynchronize();
			debug_print<<<1, 1>>>(nbr_tasks, gpu);
			cudaDeviceSynchronize();
			whine(Severity::INFO,
			      "Dummy returning"
			      GAPPP_LOG_DUMMY);
			return 0;
		}

		int invoke_nothing(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			return 0;
		}

		int init() { return 0; }
	}
}