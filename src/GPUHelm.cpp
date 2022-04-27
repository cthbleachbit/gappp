//
// Created by cth451 on 4/20/22.
//

#include "GPUHelm.h"
#include "Logging.h"

namespace GAPPP {
	GPUHelm::GPUHelm() {
		// The helm is single
		ring_tasks = rte_ring_create("GPUHelmRingBuf", GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT, 0, RING_F_SC_DEQ);
		if (ring_tasks == nullptr) {
			whine(Severity::CRIT, "Cannot allocate message ring buffer", "GPU Helm");
		}
	}
	GPUHelm::~GPUHelm() {
		rte_ring_free(ring_tasks);
	}
} // GAPPP