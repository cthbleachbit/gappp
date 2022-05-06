#ifndef GAPPP_CUDA_SELFTEST
#define GAPPP_CUDA_SELFTEST

#include <rte_mbuf.h>
#include "gappp_types.h"

namespace GAPPP::selftest {
	int self_test(bool try_direct);
	int init();
}

#endif
