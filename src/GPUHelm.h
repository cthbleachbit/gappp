//
// Created by cth451 on 4/20/22.
//

#ifndef GAPPP_GPUHELM_H
#define GAPPP_GPUHELM_H

#include <unordered_set>
#include <future>
#include <tuple>
#include <rte_ring.h>

#include "Router.h"

// Slots to reserve in the ring_tasks buffer
#define GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT 64
// Number of tasks to dequeue in one shot
#define GAPPP_GPU_HELM_TASK_BURST 4U
// Preallocate minion asynchronous results
#define GAPPP_GPU_FUTURE_PREALLOCATE 10
#define GAPPP_LOG_GPU_HELM "GPU Helm"

namespace GAPPP {

	class GPUHelm {
		// Incoming buffers - CPU workers will submit_rx tasks to this ring buffer
		// Note that this buffer is multi-producer/single-consumer.
		struct rte_ring *ring_tasks = nullptr;
		// GPU threads will place finished data here
		// Note tha this buffer is multi-producer(running)/multi-consumer(passed directly to NIC workers)
		struct rte_ring *ring_completion = nullptr;
		// Outstanding GPU threads
		std::vector<std::shared_future<int>> running;

		// Pointer to initialized router instance - ownership is borrowed (i.e. not to be freed)
		Router *r;

	public:
		/**
		 * Construct a GPU helm and allocate associated message ring buffers
		 */
		GPUHelm();

		/**
		 * Free the associated data structures
		 */
		~GPUHelm();

		/**
		 * Submit tasks to the GPU Helm
		 * @param thread_id identity of running thread
		 * @param task      incoming packet to process - this pointer will be freed upon consumption by GPU helm
		 * @return    0 if submission was successful
		 */
		int submit_rx(Router::thread_ident thread_id, Router::thread_local_mbuf *task);

		/**
		 * GPU main event loop
		 * @param stop     terminate when stop is true
		 * @param r        the router where output should be delivered to
		 */
		void gpu_helm_event_loop(const volatile bool *stop, Router &r);

		void inline assign_router(Router *r) {
			this->r = r;
		}

	private:
		int gpu_minion_thread();
	};

} // GAPPP

#endif //GAPPP_GPUHELM_H
