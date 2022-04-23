#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>
using namespace std;


struct route {
		uint32_t network;
		uint32_t mask;
		uint32_t gateway;
		uint16_t out_port;
	};

struct gpu_routing_table {
		uint16_t num;
		vector<route> routes;
	};




struct route parseLineNoGate(string text){

    string space_delimiter = " ";
    struct route r;
    int pos;
    

    //find address
    pos = text.find(space_delimiter);
    string addr = text.substr(0,pos);
    text.erase(0,pos+1);
    
    //split address into ip and mask 
    size_t slash = addr.find("/");
    string ipv4 = addr.substr(0,slash);
    
    //parse address 
    struct sockaddr_in serve;
    int domain,s;
    domain = AF_INET;
    s = inet_pton(domain,ipv4.c_str() , &serve.sin_addr);
    r.network = serve.sin_addr.s_addr; 
    
    //find the mask 
    string mask = addr.substr(slash+1,addr.length()); 
    r.mask = 0xffffffffu << (32 - stoi(mask));
    
    r.gateway = 0;
    
    //find port    
    pos = text.find(space_delimiter); 
    text.erase(0, pos + 1);
    r.out_port = stoi(text);

    return r;
}

struct route parseLineGate(string text){

    string space_delimiter = " "; 
    struct route r;
    int pos;
    
    //find address
    pos = text.find(space_delimiter);
    string addr = text.substr(0,pos);
    text.erase(0,pos+1); 
    

    //split address into ip
    size_t slash = addr.find("/");
    string ipv4 = addr.substr(0,slash);
     
    
    //parse address
    struct sockaddr_in serve;
    int domain,s;
    domain = AF_INET;
    s = inet_pton(domain,ipv4.c_str() , &serve.sin_addr);
    r.network = serve.sin_addr.s_addr; 
    
    //find the mask
    string mask = addr.substr(slash + 1,addr.length());
    r.mask = 0xffffffffu << (32 - stoi(mask));
    
    //remove via
    pos = text.find(space_delimiter); 
    text.erase(0, pos + 1);
    
    //find gateway 	
    pos = text.find(space_delimiter); 
    string gate = text.substr(0,pos);
    text.erase(0,pos+1);    
    struct sockaddr_in serve_gate;
    s = inet_pton(domain, gate.c_str() ,&serve_gate.sin_addr);
    r.gateway=serve_gate.sin_addr.s_addr;
    
    //find port    
    pos = text.find(space_delimiter); 
    text.erase(0, pos + 1);
    r.out_port = stoi(text);

    return r;
}
void printTablePrinter(gpu_routing_table table){

  string header ="Network   Mask  Gate    Port";
  cout<< header << endl;
  for(auto & route:table.routes){
     
     cout << route.network <<" ";
     cout <<route.mask <<" ";
     cout << route.gateway << " ";
     cout << route.out_port << endl;//<< mask << gate<< port << endl;

  }

}



int main () {
  string line;
  ifstream myfile ("test.txt");
  if (myfile.is_open())
  {
	
    struct gpu_routing_table table;

    while(getline(myfile, line)) {
        cout <<line<<endl;	
	size_t gate = line.find("via");
	struct route r;
	
	if(gate != string::npos){
	    r = parseLineGate(line);
	}else{
	    r = parseLineNoGate(line);
	}
        
        table.routes.push_back(r);


    }
     
    printTablePrinter(table);
    myfile.close();
  }

  else cout << "Unable to open file"; 

  return 0;
}
