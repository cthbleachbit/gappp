//
// Created by cth451 on 22-4-25.
//

#ifndef GAPPP_ROUTETABLE
#define GAPPP_ROUTETABLE

#include "gappp_types.h"
#include <iostream>
#include <optional>
#include <string>

namespace GAPPP {
	/**
	 * parse a single line of route info
	 * @param line
	 * @return
	 */
	std::optional<struct route> parse_route(const std::string &line);

	/**
	 * Parse a table of routes input from text
	 * @param input
	 * @return
	 */
	routing_table parse_table(std::istream &input);

	/**
	 * print routing table - a debug functionality
	 * @param table
	 * @param os
	 */
	inline void printTablePrinter(const routing_table& table, std::ostream &os = std::cout) {
		std::string header = "Network   Mask  Gate    Port";
		std::cout << header << std::endl;
		for (const auto &route: table) {
			os << route.network << " ";
			os << route.mask << " ";
			os << route.gateway << " ";
			os << route.out_port << std::endl;//<< mask << gate<< port << endl;
		}
	}
}

#endif //ROUTETABLE_H
