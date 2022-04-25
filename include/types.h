//
// Created by cth451 on 22-4-19.
//

#ifndef GAPPP_TYPES
#define GAPPP_TYPES

#include <vector>
#include <cstdint>

/**
 * See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html on efficient data structure copying
 */
namespace GAPPP {
	struct route {
		uint32_t network;
		uint32_t mask;
		uint32_t gateway;
		uint16_t out_port;
	};

	typedef std::vector<struct route> routing_table;

	struct gpu_routing_table {
		uint16_t num;
		struct route *routes;
	};

	struct packet_info {
		uint16_t incoming_port;
		uint16_t outgoing_port;
		uint16_t buf_len;
		unsigned char *buf;
	};
}

#endif //GAPPP_TYPES
