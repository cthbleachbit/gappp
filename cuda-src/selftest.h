#ifndef GAPPP_CUDA_SELFTEST
#define GAPPP_CUDA_SELFTEST

#include <rte_mbuf.h>
#include "gappp_types.h"

namespace GAPPP::selftest {
	/**
	 * Self test - fill a vector with GPU
	 * @param nbr_tasks   unused
	 * @param packets     unused
	 * @return              0   if test successful,
	 *                     -1   if results mismatch
	 *     std::runtime_error   on other CUDA errors
	 */
	int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets);
}

#endif
