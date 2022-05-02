//
// Created by cth451 on 5/2/22.
//

#ifndef GAPPP_CUDA_DUMMY_H
#define GAPPP_CUDA_DUMMY_H

#include <rte_mbuf.h>

namespace GAPPP::dummy {
	/**
	 * Dummy processing module - map things into GPU memory and do nothing
	 * @param nbr_tasks
	 * @param packets
	 * @return
	 */
	int invoke(unsigned int nbr_tasks, struct rte_mbuf **packets);
}

#endif //GAPPP_DUMMY_H
