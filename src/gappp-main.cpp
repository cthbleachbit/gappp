//
// Created by cth451 on 2022/4/3.
//

#include <cstdint>
#include <iostream>
#include <fmt/format.h>
#include <rte_eal.h>

#include "Logging.h"

/*
 * Notes: DPDK parameters must be placed before program parameters
 */

int main(int argc, char **argv) {
	int ret;

	// Initialize EAL
	ret = rte_eal_init(argc, argv);
	if (ret < 0) {
		GAPPP::whine(GAPPP::Severity::CRIT, "Invalid EAL parameters");
		return 1;
	}
	argc -= ret;
	argv += ret;

	// TODO: Handle program options

	// TODO: Create router instance

	// TODO: Start event loop
	return 0;
}