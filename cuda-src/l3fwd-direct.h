//
// Created by cth451 on 5/3/22.
//

#ifndef GAPPP_CUDA_L3FWD_DIRECT_H
#define GAPPP_CUDA_L3FWD_DIRECT_H

namespace GAPPP {
	namespace l3direct {
		int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets);
	}
}

#endif //GAPPP_L3FWD_DIRECT_H
