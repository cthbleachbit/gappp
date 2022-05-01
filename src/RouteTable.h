//
// Created by cth451 on 22-4-25.
//

#ifndef GAPPP_ROUTETABLE
#define GAPPP_ROUTETABLE

#include "gappp_types.h"
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
}

#endif //ROUTETABLE_H
