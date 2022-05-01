#ifndef GAPPP_CUDA_SELFTEST_H
#define GAPPP_CUDA_SELFTEST_H

#include <rte_mbuf.h>
#include "gappp_types.h"

namespace GAPPP::selftest {
	int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets);
	inline bool selftest() {
		return GAPPP::selftest::invoke(0, nullptr) == 255;
	}
}

#endif
