//
// Created by cth451 on 22-4-19.
//

#ifndef GAPPP_TYPES
#define GAPPP_TYPES

#include <vector>
#include <cstdint>
#include <functional>
#include <rte_mbuf.h>

/**
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html on efficient data structure copying
 */
namespace GAPPP {
	struct route {
		uint32_t network;
		uint32_t mask;
		uint32_t gateway;
		uint16_t out_port;
	};

	typedef std::vector<struct route> routing_table;

	/**
	 * CUDA module invocation routine - called from GPU helm at each trigger
	 */
	typedef std::function<int(unsigned int, struct rte_mbuf **)> cuda_module_invoke_t;

	/**
	 * CUDA module initialization routine - called from GPU helm constructor only once
	 */
	typedef std::function<int(void)> cuda_module_init_t;
}

#endif //GAPPP_TYPES
