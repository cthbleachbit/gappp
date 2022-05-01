//
// Created by cth451 on 22-4-17.
//

#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <string>
#include <fmt/format.h>
#include <rte_branch_prediction.h>

namespace GAPPP {
	enum class Severity {
		INFO,
		WARN,
		CRIT
	};

	inline const char* sev_to_string(Severity sev) {
		switch (sev) {
			case Severity::INFO:
				return "INFO";
			case Severity::WARN:
				return "WARN";
			case Severity::CRIT:
				return "CRIT";
		}
	}

	inline std::string mac_addr_to_string(uint8_t mac[6]) {
		return fmt::format("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
	};

	inline void whine(Severity sev, const std::string &message, const std::string &component = "???") {
		std::string print_out = fmt::format("[{}/{}] {}\n", component, sev_to_string(sev), message);
		std::cout << print_out << std::flush;
		if (unlikely(sev == Severity::CRIT)) {
			throw std::runtime_error(print_out);
		}
	}
}

#endif //LOGGING_H
