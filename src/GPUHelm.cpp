//
// Created by cth451 on 4/20/22.
//

#include "Logging.h"
#include "components.h"

#include "l3fwd.h"
#include "selftest.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "RouteTable.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::ifstream;
using std::vector;


namespace GAPPP {
	GPUHelm::GPUHelm(GAPPP::cuda_module_t &module_invoke, const std::string &path_route_table)
			: module_invoke(module_invoke) {
		// Do a GPU quick self test
		if (GAPPP::selftest::invoke(0, nullptr) != 0) {
			whine(Severity::CRIT, "GPU Self test failed", GAPPP_LOG_GPU_HELM);
		} else {
			whine(Severity::INFO, "GPU Self test completed", GAPPP_LOG_GPU_HELM);
		}

		ring_tasks = rte_ring_create("GPUHelmRingTask", GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT, 0, RING_F_SC_DEQ);
		if (ring_tasks == nullptr) {
			whine(Severity::CRIT, "Cannot allocate Task Queue ring buffer", GAPPP_LOG_GPU_HELM);
		}
		ring_completion = rte_ring_create("GPUHelmRingCompletion", GAPPP_GPU_HELM_MESSAGE_SLOT_COUNT, 0, 0);
		if (ring_completion == nullptr) {
			whine(Severity::CRIT, "Cannot allocate Completion Queue ring buffer", GAPPP_LOG_GPU_HELM);
		}
		running.reserve(GAPPP_GPU_FUTURE_PREALLOCATE);

		// Transfer routing table into GPU?
		ifstream input_file(path_route_table);
		if (!input_file.is_open()) {
			whine(Severity::CRIT, fmt::format("Failed to open routing table at {}", path_route_table),
			      GAPPP_LOG_GPU_HELM);
		}

		routing_table table;
		table = GAPPP::parse_table(input_file);
		int x = GAPPP::l3fwd::setTable(table);
		whine(Severity::INFO, "Routing table loaded", GAPPP_LOG_GPU_HELM);
		GAPPP::printTablePrinter(table, cout);
	}

	GPUHelm::~GPUHelm() {
		rte_ring_free(ring_tasks);
		ring_tasks = nullptr;
		rte_ring_free(ring_completion);
		ring_completion = nullptr;
	}

	unsigned int GPUHelm::submit_rx(router_thread_ident thread_id, router_thread_local_mbuf *task) {
		return this->submit_rx(thread_id, task->len, task->m_table);
	}

	unsigned int GPUHelm::submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task) {
		unsigned int ret;
		ret = rte_ring_enqueue_burst(this->ring_tasks, (void *const *) task, len, nullptr);
		if (ret < len) {
			whine(Severity::WARN,
			      fmt::format("GPU task buffer: {} enqueue requested by thread {} > {} enqueued", len, thread_id, ret),
			      GAPPP_LOG_GPU_HELM);
		} else {
			whine(Severity::INFO,
			      fmt::format("Thread {} submitted {} packets for processing", thread_id, len),
			      GAPPP_LOG_GPU_HELM);
		}
		return (len - ret);
	}

	void GPUHelm::gpu_helm_event_loop(const volatile bool *stop) {
		using namespace std::chrono_literals;
		auto ring_completion = this->ring_completion;

		if (!this->ring_tasks || !this->ring_completion) {
			whine(Severity::CRIT, "GPU Helm ring buffers are not initialized", GAPPP_LOG_GPU_HELM);
		}

		std::array<struct rte_mbuf *, GAPPP_GPU_HELM_TASK_BURST> local_tasks{};
		std::array<struct rte_mbuf *, GAPPP_GPU_HELM_TASK_BURST> local_completion{};
		int nbr_local_tasks = 0;

		whine(Severity::INFO, "Entering event loop", GAPPP_LOG_GPU_HELM);

		while (!*stop) {
			// Note: ownership is transferred into minion - free after use
			auto local_tasks = new std::array<struct rte_mbuf *, GAPPP_GPU_HELM_TASK_BURST>();
			nbr_local_tasks = rte_ring_dequeue_bulk(this->ring_tasks,
			                                        (void **) local_tasks->data(),
			                                        GAPPP_GPU_HELM_TASK_BURST,
			                                        nullptr);
			if (nbr_local_tasks > 0) {
				// TODO: spawn GPU minions that launches CUDA contexts
				this->running.emplace_back(std::async(std::launch::async, [nbr_local_tasks, local_tasks, this] {
					return this->gpu_minion_thread(nbr_local_tasks, local_tasks);
				}));
			}

			for (auto future = this->running.begin(); future < this->running.end(); future++) {
				auto status = future->wait_for(0ms);

				if (status == std::future_status::ready) {
					whine(Severity::INFO,
					      fmt::format("GPU minion returned with {}", future->get()),
					      GAPPP_LOG_GPU_HELM);
				}

				this->running.erase(future);
			}
		}

		whine(Severity::INFO, "Terminating", GAPPP_LOG_GPU_HELM);
	}

	int GPUHelm::gpu_minion_thread(unsigned int nbr_tasks,
	                               std::array<struct rte_mbuf *, GAPPP_GPU_HELM_TASK_BURST> *packets) {
		int ret = this->module_invoke(nbr_tasks, packets->data());
		// FIXME: extremely slow - submit tx to workers
		for (int idx = 0; idx < nbr_tasks; idx++) {
			struct rte_mbuf *pkt = (*packets)[idx];
			r->submit_tx(pkt->port, 1, &pkt);
		}
		delete packets;
		return ret;
	}

} // GAPPP
