#include <fstream>
#include <sys/prctl.h>
#include <rte_gpudev.h>
#include <cuda_runtime_api.h>

#include "components.h"
#include "Logging.h"
#include "RouteTable.h"
#include "l3fwd-direct.h"
#include "selftest.h"

namespace GAPPP {
	GPUDirectHelm::GPUDirectHelm(GAPPP::cuda_module_invoke_t &module_invoke,
	                             GAPPP::cuda_module_init_t &module_init,
	                             const std::string &path_route_table)
		: module_invoke(module_invoke) {
		// Do a GPU quick self test
		if (GAPPP::selftest::self_test(true) != 0) {
			whine(Severity::CRIT, "GPU Self test failed", GAPPP_LOG_GPU_DIRECT_HELM);
		} else {
			whine(Severity::INFO, "GPU Self test completed", GAPPP_LOG_GPU_DIRECT_HELM);
		}

		if (module_init()) {
			whine(Severity::CRIT, "Module initialization failed", GAPPP_LOG_GPU_DIRECT_HELM);
		};

		// Transfer routing table into GPU?
		std::ifstream input_file(path_route_table);
		routes = GAPPP::parse_table(input_file);
		int x = GAPPP::l3direct::setTable(routes);
		whine(Severity::INFO, "Routing table loaded", GAPPP_LOG_GPU_DIRECT_HELM);
		GAPPP::printTablePrinter(routes, std::cout);

		// Setup gpu communication list
		for (int i = 0; i < GAPPP_DIRECT_MAX_PERSISTENT_KERNEL; i++) {
			gpu_comm_lists[i] = rte_gpu_comm_create_list(GAPPP_GPU_ID, GAPPP_GPU_HELM_TASK_BURST);
			if (gpu_comm_lists[i] == nullptr) {
				whine(Severity::CRIT, fmt::format("GPU comm list {} allocation failure", i), GAPPP_LOG_GPU_DIRECT_HELM);
			}
			cudaError_t cuda_ret = cudaStreamCreateWithFlags(&(gpu_streams[i]), cudaStreamNonBlocking);
			if (cuda_ret != cudaSuccess) {
				whine(Severity::CRIT,
				      fmt::format("GPU stream {} allocation failure: {}", i, cudaGetErrorName(cuda_ret)),
				      GAPPP_LOG_GPU_DIRECT_HELM);
			}
		}
	}

	GPUDirectHelm::~GPUDirectHelm() {
		for (int i = 0; i < GAPPP_DIRECT_MAX_PERSISTENT_KERNEL; i++) {
			if (gpu_streams[i] != nullptr) {
				cudaStreamDestroy(gpu_streams[i]);
			}
			if (gpu_comm_lists[i] != nullptr) {
				rte_gpu_comm_destroy_list(GAPPP_GPU_ID, GAPPP_GPU_HELM_TASK_BURST);
				gpu_comm_lists[i] = nullptr;
			}
		}
	}

	unsigned int GPUDirectHelm::submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task) {
		// TODO
		return 0;
	}

	void GPUDirectHelm::gpu_helm_event_loop(const volatile bool *stop) {
		using namespace std::chrono_literals;

		prctl(PR_SET_NAME, GAPPP_LOG_GPU_DIRECT_HELM);

		// Sanity check the routing table - port must exist
		for (const auto &route: this->routes) {
			if (unlikely(!r->ports.contains(route.out_port))) {
				whine(Severity::WARN,
				      fmt::format("Port {} specified in the routing table doesn't exist", route.out_port),
				      GAPPP_LOG_GPU_DIRECT_HELM);
			}
		}

		whine(Severity::INFO, "Starting persistent kernels", GAPPP_LOG_GPU_DIRECT_HELM);


		while (!*stop) {
			sleep(5);
		}

		for (int i = 0; i < GAPPP_GPU_HELM_TASK_BURST; i++) {
			if(rte_gpu_comm_set_status(this->gpu_comm_lists[i], RTE_GPU_COMM_LIST_ERROR)) {
				whine(Severity::WARN,
				      fmt::format("Failed to set stop condition on GPU comm list {}", i),
				      GAPPP_LOG_GPU_DIRECT_HELM);
			}
		}

		whine(Severity::INFO, "Terminating", GAPPP_LOG_GPU_DIRECT_HELM);
	}

	int GPUDirectHelm::register_ext_mem(const rte_pktmbuf_extmem &external_mem) {
		int ret;

		whine(Severity::INFO, "Registering DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		ret = rte_gpu_mem_register(GAPPP_GPU_ID, external_mem.buf_len, external_mem.buf_ptr);
		if (ret < 0) {
			whine(Severity::CRIT, "Failed to register DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		} else {
			whine(Severity::INFO, "Registered DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		}
		return 0;
	}

	int GPUDirectHelm::unregister_ext_mem(const rte_pktmbuf_extmem &external_mem) {
		int ret;

		whine(Severity::INFO, "Unregistering DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		ret = rte_gpu_mem_unregister(GAPPP_GPU_ID, external_mem.buf_ptr);
		if (ret < 0) {
			whine(Severity::CRIT, "Failed to unregister DMA zone from GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		} else {
			whine(Severity::INFO, "Unregistered DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		}
		return 0;
	}

} // GAPPP