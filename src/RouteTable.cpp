#include <iostream>
#include <fstream>
#include <string>
#include <optional>
#include <arpa/inet.h>
#include "gappp_types.h"

#define GAPPP_ROUTE_DELIMITER ' '

namespace GAPPP {
	static std::optional<struct sockaddr_in> parse_address(const std::string &input) {
		struct sockaddr_in serve{};
		int s = inet_pton(AF_INET, input.c_str(), &serve.sin_addr);
		if (s <= 0) {
			return std::nullopt;
		} else {
			return serve;
		}
	}

	static std::optional<struct route> parse_line_no_gateway(const std::string &input) {
		std::string text = input;
		struct route r{};
		size_t pos;
		std::optional<struct sockaddr_in> serve;

		//find address
		pos = text.find(GAPPP_ROUTE_DELIMITER);
		std::string addr = text.substr(0, pos);
		text.erase(0, pos + 1);

		//split address into ip and mask
		size_t slash = addr.find('/');
		std::string ipv4 = addr.substr(0, slash);

		//parse address
		serve = parse_address(ipv4);
		if (serve.has_value()) { r.network = serve->sin_addr.s_addr; }
		else { return std::nullopt; }

		//find the mask
		std::string mask = addr.substr(slash + 1, addr.length());
		r.mask = 0xffffffffu << (32 - stoi(mask));

		r.gateway = 0;

		//find port
		pos = text.find(GAPPP_ROUTE_DELIMITER);
		text.erase(0, pos + 1);
		r.out_port = stoi(text);

		return r;
	}

	static std::optional<struct route> parse_line_gateway(const std::string &input) {
		std::string text = input;
		struct route r{};
		size_t pos;
		std::optional<struct sockaddr_in> serve;

		//find address
		pos = text.find(GAPPP_ROUTE_DELIMITER);
		std::string addr = text.substr(0, pos);
		text.erase(0, pos + 1);


		//split address into ip
		size_t slash = addr.find('/');
		std::string ipv4 = addr.substr(0, slash);


		//parse address
		serve = parse_address(ipv4);
		if (serve.has_value()) { r.network = serve->sin_addr.s_addr; }
		else { return std::nullopt; }

		//find the mask
		std::string mask = addr.substr(slash + 1, addr.length());
		r.mask = 0xffffffffu << (32 - stoi(mask));

		//remove via
		pos = text.find(GAPPP_ROUTE_DELIMITER);
		text.erase(0, pos + 1);

		//find gateway
		pos = text.find(GAPPP_ROUTE_DELIMITER);
		std::string gate = text.substr(0, pos);
		text.erase(0, pos + 1);

		serve = parse_address(ipv4);
		if (serve.has_value()) { r.gateway = serve->sin_addr.s_addr; }
		else { return std::nullopt; }

		//find port
		pos = text.find(GAPPP_ROUTE_DELIMITER);
		text.erase(0, pos + 1);
		r.out_port = stoi(text);

		return r;
	}

	/**
	 * parse a single line of route info
	 * @param line
	 * @return
	 */
	std::optional<struct route> parse_route(const std::string &line) {
		size_t gate = line.find("via");

		if (gate != std::string::npos) {
			return GAPPP::parse_line_gateway(line);
		} else {
			return GAPPP::parse_line_no_gateway(line);
		}
	}

	/**
	 * Parse a table of routes input from text
	 * @param input
	 * @return
	 */
	routing_table parse_table(std::istream &input) {
		std::string line;
		routing_table routes;
		while (std::getline(input, line)) {
			auto r = parse_route(line);
			if (r.has_value()) {
				routes.push_back(*r);
			}
		}
		return routes;
	}

	void printTablePrinter(const routing_table& table, std::ostream &os = std::cout) {
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
