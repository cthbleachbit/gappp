//
// Created by cth451 on 4/20/22.
//

#ifndef GAPPP_GPUHELM_H
#define GAPPP_GPUHELM_H

#include <unordered_set>
#include <future>

namespace GAPPP {

	class GPUHelm {
		/**
		 * A set of outstanding GPU tasks waiting for returning
		 */
		std::unordered_set<std::promise<int>> outstanding;
	};

} // GAPPP

#endif //GAPPP_GPUHELM_H
