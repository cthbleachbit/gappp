#include <fstream>
#include <sys/prctl.h>
#include <rte_gpudev.h>

#include "components.h"
#include "Logging.h"
#include "RouteTable.h"
#include "l3fwd-direct.h"
#include "selftest.h"

namespace GAPPP {
	GPUDirectHelm::GPUDirectHelm(GAPPP::cuda_module_t &module_invoke, const std::string &path_route_table)
		: module_invoke(module_invoke) {
		// Do a GPU quick self test
		if (GAPPP::selftest::self_test(true) != 0) {
			whine(Severity::CRIT, "GPU Self test failed", GAPPP_LOG_GPU_DIRECT_HELM);
		} else {
			whine(Severity::INFO, "GPU Self test completed", GAPPP_LOG_GPU_DIRECT_HELM);
		}

		// Transfer routing table into GPU?
		std::ifstream input_file(path_route_table);
		routes = GAPPP::parse_table(input_file);
		int x = GAPPP::l3direct::setTable(routes);
		whine(Severity::INFO, "Routing table loaded", GAPPP_LOG_GPU_HELM);
		GAPPP::printTablePrinter(routes, std::cout);
	}

	GPUDirectHelm::~GPUDirectHelm() {

	}

	unsigned int GPUDirectHelm::submit_rx(router_thread_ident thread_id, size_t len, struct rte_mbuf *const *task) {
		// TODO
		return 0;
	}

	void GPUDirectHelm::gpu_helm_event_loop(const volatile bool *stop) {
		using namespace std::chrono_literals;

		prctl(PR_SET_NAME, GAPPP_LOG_GPU_HELM);

		// Sanity check the routing table - port must exist
		for (const auto &route: this->routes) {
			if (unlikely(!r->ports.contains(route.out_port))) {
				whine(Severity::WARN,
				      fmt::format("Port {} specified in the routing table doesn't exist", route.out_port),
				      GAPPP_LOG_GPU_HELM);
			}
		}

		whine(Severity::INFO, "Entering event loop", GAPPP_LOG_GPU_HELM);


		while (!*stop) {
			// TODO: event loop
		}

		whine(Severity::INFO, "Terminating", GAPPP_LOG_GPU_HELM);
	}

	int GPUDirectHelm::register_ext_mem(const rte_pktmbuf_extmem &external_mem) {
		int ret;

		ret = rte_gpu_mem_register(GAPPP_GPU_ID, external_mem.buf_len, external_mem.buf_ptr);
		if (ret < 0) {
			whine(Severity::CRIT, "Failed to register DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		} else {
			whine(Severity::INFO, "Registered DMA zone with GPU", GAPPP_LOG_GPU_DIRECT_HELM);
		}
		return 0;
	}

} // GAPPP