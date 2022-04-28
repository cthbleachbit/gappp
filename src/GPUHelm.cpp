//
// Created by cth451 on 4/20/22.
//

#include "GPUHelm.h"
#include "Logging.h"

namespace GAPPP {
	GPUHelm::GPUHelm() {
		ring_tasks = rte_ring_create("GPUHelmRingTask", GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT, 0, RING_F_SC_DEQ);
		if (ring_tasks == nullptr) {
			whine(Severity::CRIT, "Cannot allocate Task Queue ring buffer", "GPU Helm");
		}
		ring_completion = rte_ring_create("GPUHelmRingCompletion", GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT, 0, 0);
		if (ring_completion == nullptr) {
			whine(Severity::CRIT, "Cannot allocate Completion Queue ring buffer", "GPU Helm");
		}
		running.reserve(GAPPP_GPU_FUTURE_PREALLOCATE);
		// TODO: Transfer routing table into GPU?
	}

	GPUHelm::~GPUHelm() {
		rte_ring_free(ring_tasks);
		ring_tasks = nullptr;
		rte_ring_free(ring_completion);
		ring_completion = nullptr;
	}

	int GPUHelm::submit_rx(Router::thread_ident thread_id, Router::thread_local_mbuf *task) {
		int ret;
		ret = rte_ring_enqueue(this->ring_tasks, task);
		whine(Severity::INFO,
		      fmt::format("Thread {} submitted {} packets for processing", thread_id, task->len),
		      GAPPP_LOG_GPU_HELM);
		return ret;
	}

	void GPUHelm::gpu_helm_event_loop(const volatile bool *stop, Router &r) {
		using namespace std::chrono_literals;

		if (!this->ring_tasks || !this->ring_completion) {
			whine(Severity::CRIT, "GPU Helm ring buffers are not initialized", GAPPP_LOG_GPU_HELM);
		}

		std::array<Router::thread_local_mbuf *, GAPPP_GPU_HELM_TASK_BURST> local_tasks{};
		std::array<Router::thread_local_mbuf *, GAPPP_GPU_HELM_TASK_BURST> local_completion{};
		int nbr_local_tasks = 0;

		while (!*stop) {
			nbr_local_tasks = rte_ring_dequeue_bulk(this->ring_tasks,
			                                        (void **) local_tasks.data(),
			                                        GAPPP_GPU_HELM_TASK_BURST,
			                                        nullptr);
			if (nbr_local_tasks > 0) {
				// TODO: spawn GPU minions that launches CUDA contexts
				// Launch with
				// this->running.emplace_back(std::async(std::launch::async, [<capture>] {thread_routine};));
			}

			for (auto future = this->running.begin(); future != this->running.end(); future++) {
				auto status = future->wait_for(0ms);

				if (status == std::future_status::ready) {
					whine(Severity::INFO,
					      fmt::format("GPU minion returned with {}", future->get()),
					      GAPPP_LOG_GPU_HELM);
				}

				this->running.erase(future);
			}
		}
	}


} // GAPPP