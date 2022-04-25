//
// Created by cth451 on 2022/4/3.
//

#include <cstdint>
#include <iostream>
#include <fmt/format.h>
#include <rte_eal.h>

#include "Logging.h"
#include "Router.h"

/*
 * Notes: DPDK parameters must be placed before program parameters
 * Hugepages of size 1GiB must be allocated before starting the program
 *
 * sudo ./gappp -w 41:00.0
 * If testing on Ming's testbeds, use `-w 41:00.0` to take control of the card.
 */


int main(int argc, char **argv) {
	int ret;

	// Initialize EAL / DPDK
	ret = rte_eal_init(argc, argv);
	if (ret < 0) {
		GAPPP::whine(GAPPP::Severity::CRIT, "Invalid EAL parameters");
		return 1;
	}
	argc -= ret;
	argv += ret;

	// TODO: Handle program options

	// TODO: Create router instance
	GAPPP::Router r;

	// TODO: Start event loop
	return 0;
}