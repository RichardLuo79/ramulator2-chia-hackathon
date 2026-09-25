// Untimed virtual-page inventory. Use the same input_instr ABI as ChampSim.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <unordered_set>
#include <vector>
#include "trace_instruction.h"

int main()
{
  static_assert(sizeof(input_instr) == 64);
  std::unordered_set<uint64_t> pages;
  std::array<input_instr, 16384> buffer;
  uint64_t instructions = 0;
  while (auto count = std::fread(buffer.data(), sizeof(input_instr), buffer.size(), stdin)) {
    instructions += count;
    for (std::size_t i = 0; i < count; ++i) {
      pages.insert(buffer[i].ip >> 12);
      for (auto address : buffer[i].source_memory)
        if (address) pages.insert(address >> 12);
      for (auto address : buffer[i].destination_memory)
        if (address) pages.insert(address >> 12);
    }
  }
  if (std::ferror(stdin) || !instructions) return 1;
  std::vector<uint64_t> sorted(pages.begin(), pages.end());
  std::sort(sorted.begin(), sorted.end());
  std::printf("%llu\n", static_cast<unsigned long long>(instructions));
  for (auto page : sorted) std::printf("%llu\n", static_cast<unsigned long long>(page));
  return std::ferror(stdout) ? 1 : 0;
}
