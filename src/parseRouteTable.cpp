#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>

struct route {
	uint32_t network;
	uint32_t mask;
	uint32_t gateway;
	uint16_t out_port;
};

struct gpu_routing_table {
	uint16_t num;
	std::vector<route> routes;
};

struct route parseLineNoGate(std::string text) {
	std::string space_delimiter = " ";
	struct route r;
	int pos;


	//find address
	pos = text.find(space_delimiter);
	std::string addr = text.substr(0, pos);
	text.erase(0, pos + 1);

	//split address into ip and mask
	size_t slash = addr.find("/");
	std::string ipv4 = addr.substr(0, slash);

	//parse address
	struct sockaddr_in serve;
	int domain, s;
	domain = AF_INET;
	s = inet_pton(domain, ipv4.c_str(), &serve.sin_addr);
	r.network = serve.sin_addr.s_addr;

	//find the mask
	std::string mask = addr.substr(slash + 1, addr.length());
	r.mask = 0xffffffffu << (32 - stoi(mask));

	r.gateway = 0;

	//find port
	pos = text.find(space_delimiter);
	text.erase(0, pos + 1);
	r.out_port = stoi(text);

	return r;
}

struct route parseLineGate(std::string text) {

	std::string space_delimiter = " ";
	struct route r;
	int pos;

	//find address
	pos = text.find(space_delimiter);
	std::string addr = text.substr(0, pos);
	text.erase(0, pos + 1);


	//split address into ip
	size_t slash = addr.find("/");
	std::string ipv4 = addr.substr(0, slash);


	//parse address
	struct sockaddr_in serve;
	int domain, s;
	domain = AF_INET;
	s = inet_pton(domain, ipv4.c_str(), &serve.sin_addr);
	r.network = serve.sin_addr.s_addr;

	//find the mask
	std::string mask = addr.substr(slash + 1, addr.length());
	r.mask = 0xffffffffu << (32 - stoi(mask));

	//remove via
	pos = text.find(space_delimiter);
	text.erase(0, pos + 1);

	//find gateway
	pos = text.find(space_delimiter);
	std::string gate = text.substr(0, pos);
	text.erase(0, pos + 1);
	struct sockaddr_in serve_gate;
	s = inet_pton(domain, gate.c_str(), &serve_gate.sin_addr);
	r.gateway = serve_gate.sin_addr.s_addr;

	//find port
	pos = text.find(space_delimiter);
	text.erase(0, pos + 1);
	r.out_port = stoi(text);

	return r;
}

void printTablePrinter(gpu_routing_table table) {

	std::string header = "Network   Mask  Gate    Port";
	std::cout << header << std::endl;
	for (auto &route: table.routes) {

		std::cout << route.network << " ";
		std::cout << route.mask << " ";
		std::cout << route.gateway << " ";
		std::cout << route.out_port << std::endl;//<< mask << gate<< port << endl;
	}

}

int main() {
	std::string line;
	std::ifstream myfile("test.txt");
	if (myfile.is_open()) {

		struct gpu_routing_table table;

		while (getline(myfile, line)) {
			std::cout << line << endl;
			size_t gate = line.find("via");
			struct route r;

			if (gate != std::string::npos) {
				r = parseLineGate(line);
			} else {
				r = parseLineNoGate(line);
			}

			table.routes.push_back(r);

		}

		printTablePrinter(table);
		myfile.close();
	} else std::cout << "Unable to open file";

	return 0;
}
