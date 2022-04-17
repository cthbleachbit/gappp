//
// Created by cth451 on 22-4-17.
//

#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <string>
#include <fmt/format.h>

namespace GAPPP {
	enum class Severity {
		INFO,
		WARN,
		CRIT
	};

	inline std::string mac_addr_to_string(uint8_t mac[6]) {
		return fmt::format("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
	};

	inline void whine(Severity sev, const std::string &message) {
		switch (sev) {
			case Severity::INFO:
				std::cout << "INFO " << message << std::endl;
				return;
			case Severity::WARN:
				std::cerr << "INFO " << message << std::endl;
				return;
			case Severity::CRIT:
				std::cerr << "CRIT " << message << std::endl;
		}
	}
}

#endif //LOGGING_H
