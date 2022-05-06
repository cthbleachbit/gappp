//
// Created by cth451 on 2022/4/3.
//

#include <fmt/format.h>
#include <rte_eal.h>
#include <csignal>
#include <random>
#include <future>
#include <sys/prctl.h>
#include <getopt.h>

#include "Logging.h"
#include "components.h"
#include "dummy.h"
#include "l3fwd.h"
#include "l3fwd-direct.h"

/*
 * Notes: DPDK parameters must be placed before program parameters
 * Hugepages of size 1GiB must be allocated before starting the program
 *
 * sudo ./gappp -w 41:00.0
 * If testing on Ming's testbeds, use `-w 41:00.0` to take control of the card.
 */

namespace GAPPP {
	static std::function<void(int)> shutdown_handler;

	/**
	 * C-style signal handler wrapper
	 * @param signum
	 */
	static void signal_handler(int signum) {
		shutdown_handler(signum);
	}
};

int main(int argc, char **argv) {
	int ret;
	GAPPP::Router *router = nullptr;
	GAPPP::GPUHelmBase *helm = nullptr;
	volatile bool stop = false;
	std::future<void> r_thread;
	std::future<void> g_thread;
#ifdef GAPPP_GPU_DIRECT
	bool use_gpu_direct = false;
#endif

	/**
	 * Actual shutdown handler - issue stop
	 */
	GAPPP::shutdown_handler = [&stop](int signum) {
		using namespace std::chrono_literals;
		if (signum == SIGINT || signum == SIGTERM) {
			GAPPP::whine(GAPPP::Severity::INFO, fmt::format("Signal {} received, preparing to exit", signum), "Main");
			stop = true;
		}
	};

	// Initialize EAL / DPDK
	ret = rte_eal_init(argc, argv);
	if (ret < 0) {
		GAPPP::whine(GAPPP::Severity::CRIT, "Invalid EAL parameters");
	}
	argc -= ret;
	argv += ret;

	signal(SIGINT, GAPPP::signal_handler);
	signal(SIGTERM, GAPPP::signal_handler);

	// Seed the RNG - needed to randomly select TX queue
	std::random_device rng;
	std::default_random_engine rng_engine(rng());

	// TODO: Handle program options

	static struct option long_options[] = {
		{"module", required_argument, nullptr, 'm'},
		{"route", required_argument, nullptr, 'r'},
		{"num-ports", required_argument, nullptr, 'n'},
		{"port-queue", required_argument, nullptr, 'p'},
#ifdef GAPPP_GPU_DIRECT
		{"gpu-direct", no_argument, nullptr, 'g'},
#endif
		{nullptr, 0, nullptr, 0},
	};

	GAPPP::cuda_module_invoke_t module_invoke = GAPPP::dummy::invoke;
	GAPPP::cuda_module_init_t module_init = GAPPP::dummy::init;
	std::string option_route_table;
	std::string option_module;
	std::unordered_map<uint16_t, uint16_t> port_queues{};
	int num_ports = 1;
	while (true) {
		int option_index = 0;
		uint16_t p;
		uint16_t q;
		decltype(std::string("").begin()) off;
		std::string arg;
#ifdef GAPPP_GPU_DIRECT
		int c = getopt_long(argc, argv, "m:r:n:p:g", long_options, &option_index);
#else
		int c = getopt_long(argc, argv, "m:r:n:p:", long_options, &option_index);
#endif
		if (c == -1)
			break;

		switch (c) {
			case 0:
				break;
			case 'm':
				option_module = std::string(optarg);
				break;
			case 'r':
				option_route_table = std::string(optarg);
				break;
			case 'n':
				num_ports = std::atoi(optarg);
				break;
			case 'p':
				arg = std::string(optarg);
				off = std::find(arg.begin(), arg.end(), ':');
				p = std::stoi(std::string(arg.begin(), off));
				q = std::stoi(std::string(off + 1, arg.end()));
				port_queues[p] = q;
				break;
#ifdef GAPPP_GPU_DIRECT
			case 'g':
				use_gpu_direct = true;
				break;
#endif
			default:
				GAPPP::whine(GAPPP::Severity::WARN, fmt::format("Unknown argument {}", optarg), "Main");
		}
	}

	if (option_route_table.empty()) {
		GAPPP::whine(GAPPP::Severity::CRIT, "No routing table specified! Use -r <path/to/table>.", "Main");
	}

	if (option_module == "dummy") {
		module_invoke = GAPPP::dummy::invoke;
		module_init = GAPPP::dummy::init;
	} else if (option_module == "l3fwd") {
		module_invoke = use_gpu_direct ? GAPPP::l3direct::invoke : GAPPP::l3fwd::invoke;
		module_init = use_gpu_direct ? GAPPP::l3direct::init : GAPPP::l3fwd::init;
	} else {
		GAPPP::whine(GAPPP::Severity::CRIT, "No module specified! Use -m [l3fwd|dummy].", "Main");
	}

	// TODO: Create GPU Helm
#ifdef GAPPP_GPU_DIRECT
	if (use_gpu_direct) {
		helm = new GAPPP::GPUDirectHelm(module_invoke, module_init, option_route_table);
	} else {
		helm = new GAPPP::GPUHelm(module_invoke, module_init, option_route_table);
	}
#else
	helm = new GAPPP::GPUHelm(module, option_route_table);
#endif

	// TODO: Create router instance
	router = new GAPPP::Router(rng_engine);
	// Link router to GPU helm
	helm->assign_router(router);
	router->assign_gpu_helm(helm);
	for (int i = 0; i < num_ports; i++) {
		router->dev_probe(i, port_queues.contains(i) ? port_queues[i] : GAPPP_ROUTER_THREADS_PER_PORT);
	}

	// TODO: Start event loop
	prctl(PR_SET_NAME, "Main");
	r_thread = std::async(std::launch::async, [&stop, router] { router->launch_threads(&stop); });
	g_thread = std::async(std::launch::async, [&stop, helm] { helm->gpu_helm_event_loop(&stop); });

	{
		using namespace std::chrono_literals;
		while (r_thread.wait_for(5s) != std::future_status::ready) {
			if (stop) GAPPP::whine(GAPPP::Severity::INFO, "Waiting for router event loop to exit", "Main");
		}
		while (g_thread.wait_for(5s) != std::future_status::ready) {
			if (stop) GAPPP::whine(GAPPP::Severity::INFO, "Waiting for gpu event loop to exit", "Main");
		}
	}

	// END
	delete router;
	delete helm;

	return 0;
}