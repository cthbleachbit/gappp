//
// Created by cth451 on 2022/4/3.
//

#include <fmt/format.h>
#include <rte_eal.h>
#include <csignal>
#include <random>
#include <future>

#include "Logging.h"
#include "components.h"

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
	GAPPP::GPUHelm *helm = nullptr;
	volatile bool stop = false;
	std::future<void> r_thread;
	std::future<void> g_thread;

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

	// TODO: Create GPU Helm
	helm = new GAPPP::GPUHelm();

	// TODO: Create router instance
	router = new GAPPP::Router(rng_engine);
	// Link router to GPU helm
	helm->assign_router(router);
	router->assign_gpu_helm(helm);

	router->dev_probe(0);

	// TODO: Start event loop
	r_thread = std::async(std::launch::async, [&stop, router] { router->launch_threads(&stop); });
	g_thread = std::async(std::launch::async, [&stop, helm] { helm->gpu_helm_event_loop(&stop); });

	{
		using namespace std::chrono_literals;
		while (r_thread.wait_for(5s) != std::future_status::ready) {
			if (stop) GAPPP::whine(GAPPP::Severity::INFO, "Waiting for router event loop to exit", "Main");
		};
		while (g_thread.wait_for(5s) != std::future_status::ready) {
			if (stop) GAPPP::whine(GAPPP::Severity::INFO, "Waiting for gpu event loop to exit", "Main");
		};
	}

	// END
	delete router;
	delete helm;

	return 0;
}