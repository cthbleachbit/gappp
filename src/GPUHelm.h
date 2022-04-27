//
// Created by cth451 on 4/20/22.
//

#ifndef GAPPP_GPUHELM_H
#define GAPPP_GPUHELM_H

#include <unordered_set>
#include <future>
#include <rte_ring.h>

#define GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT 64

namespace GAPPP {

	class GPUHelm {
		struct rte_ring *ring_tasks;

	public:
		/**
		 * Construct a GPU helm and allocate associated message ring buffers
		 */
		GPUHelm();
		~GPUHelm();
	};

} // GAPPP

#endif //GAPPP_GPUHELM_H
