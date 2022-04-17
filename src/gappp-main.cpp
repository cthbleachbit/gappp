//
// Created by cth451 on 2022/4/3.
//

#include <cstdint>
#include <iostream>
#include <fmt/format.h>

int main(int argc, char **argv) {
	uint8_t addr_bytes[6] = {1,2,3,4,5,6};
	std::cout << fmt::format("{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", addr_bytes[0], addr_bytes[1], addr_bytes[2], addr_bytes[3], addr_bytes[4], addr_bytes[5]) << std::endl;
	return 0;
}