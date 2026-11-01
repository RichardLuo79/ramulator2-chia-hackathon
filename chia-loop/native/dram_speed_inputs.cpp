// Precompute immutable, independently offered Lat-Tp traffic outside timing.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "ramulator/frontend/lat_tp_addresses.h"

using namespace Ramulator;
int main(int argc, char** argv) {
  if (argc != 3) throw std::invalid_argument("usage: dram_speed_inputs output_dir requests");
  const size_t n = std::stoull(argv[2]);
  if (!n || n > 10000000) throw std::invalid_argument("unsupported input population");
  std::filesystem::create_directories(argv[1]);
  const std::vector<int> positions{1,3,2}, counts{1,4,8}, levels{0,1,8,4};
  for (const std::string pattern : {"streaming", "random"}) {
    for (const int ratio : {100,75,50}) {
      std::mt19937_64 addresses(12345), types(12345 ^ 0x54595045);
      std::uniform_int_distribution<int> kind(0,99);
      std::vector<char> bytes(16 + n*9);
      std::copy_n("RMSPD001", 8, bytes.data());
      auto encode = [](char* ptr, uint64_t value) {
        for (int i=0; i<8; ++i) ptr[i] = static_cast<char>(value >> (8*i));
      };
      encode(bytes.data()+8, n);
      size_t reads = 0;
      for (size_t i=0; i<n; ++i) {
        const auto av = pattern == "streaming"
          ? LatTp::streaming(i, 6, positions, counts, 32, 4, 5, 65536, 16, 64, true)
          : LatTp::random(addresses, 6, positions, counts, 4, 5, 65536, 64, 16);
        const auto address = LatTp::physical_byte_address(av, levels, 4,5,65536,64,16,64);
        const int type = kind(types) >= ratio;
        reads += type == 0;
        encode(bytes.data()+16+i*9, address);
        bytes[16+i*9+8] = static_cast<char>(type);
      }
      const std::string name = pattern + "-r" + std::to_string(ratio) + ".bin";
      const auto path = std::filesystem::path(argv[1]) / name;
      if (std::filesystem::exists(path)) throw std::runtime_error("refuse to overwrite frozen input");
      std::ofstream out(path, std::ios::binary);
      out.write(bytes.data(), bytes.size());
      out.close();
      if (!out) throw std::runtime_error("input write failed");
      std::cout << "{\"path\":\"" << name << "\",\"pattern\":\"" << pattern
                << "\",\"read_percent\":" << ratio << ",\"requests\":" << n
                << ",\"reads\":" << reads << ",\"writes\":" << n-reads << "}\n";
    }
  }
}
