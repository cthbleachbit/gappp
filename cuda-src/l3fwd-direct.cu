//
// Created by cth451 on 5/3/22.
//

#include "l3fwd-direct.h"
#include <rte_gpudev.h>

#define GAPPP_L3DIRECT_MAX_ROUTES 4096

namespace GAPPP {
	namespace l3direct {
		__managed__ int numroutes = 0;
		__managed__ struct route routes[GAPPP_L3DIRECT_MAX_ROUTES];

		int setTable(GAPPP::routing_table t) {
			numroutes = t.size();
			cudaMemcpy(routes, t.data(), sizeof(route) * t.size(), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			return 0;
		}

		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets) {
			// STUB
			return 0;
		}
	}
}