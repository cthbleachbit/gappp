//
// Created by cth451 on 4/30/22.
//

#ifndef GAPPP_CUDA_L3FWD_H
#define GAPPP_CUDA_L3FWD_H

#include <rte_mbuf.h>
#include "gappp_types.h"

namespace GAPPP::l3fwd {
	int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets);
}

#endif //GAPPP_L3FWD_H
