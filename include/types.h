//
// Created by cth451 on 22-4-19.
//

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <cstdint>

namespace GAPPP {
	struct route {
		uint32_t network;
		uint32_t mask;
		uint32_t gateway;
		uint16_t out_port;
	};

	struct routing_table {
		std::vector<struct route> routes;
	};

	struct packet_info {
		uint16_t incoming_port;
		uint16_t outgoing_port;
	};
}

#endif //TYPES_H
