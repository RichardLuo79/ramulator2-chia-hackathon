// Link against the configured ChampSim objects, replacing its main().
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include "champsim.h"
#include "dram_controller.h"
#include "vmem.h"

const std::size_t NUM_CPUS = 4;
const unsigned BLOCK_SIZE = 64, PAGE_SIZE = 4096;
const unsigned LOG2_BLOCK_SIZE = 6, LOG2_PAGE_SIZE = 12;

int main(int argc, char** argv)
{
  assert(argc == 3);
  unsetenv("CHAMPSIM_PLACEMENT_FILE");
  MEMORY_CONTROLLER dram{champsim::chrono::picoseconds{312}, champsim::chrono::picoseconds{625},
      24, 24, 24, 52, champsim::chrono::microseconds{32000}, {}, 64, 64, 1,
      champsim::data::bytes{8}, 65536, 1024, 1, 8, 4, 8192};
  VirtualMemory prepare{champsim::data::bytes{4096}, 5, champsim::chrono::picoseconds{50000}, dram, 1};
  prepare.prepare_placement(argv[1], argv[2]);
  setenv("CHAMPSIM_PLACEMENT_FILE", argv[2], 1);
  using key = std::tuple<uint32_t, uint64_t, unsigned>;
  std::map<key, uint64_t> expected;
  for (bool reverse : {false, true}) {
    VirtualMemory vm{champsim::data::bytes{4096}, 5, champsim::chrono::picoseconds{50000}, dram, 1};
    std::vector<uint64_t> pages{0, 1, 511, 512, 513, 4096, 65536};
    if (reverse) std::reverse(pages.begin(), pages.end());
    std::set<uint64_t> physical_data, roots;
    for (uint32_t index = 0; index < NUM_CPUS; ++index) {
      uint32_t cpu = reverse ? static_cast<uint32_t>(NUM_CPUS - index - 1) : index;
      roots.insert(vm.get_pte_pa(cpu, champsim::page_number{0}, 5).first.to<uint64_t>());
      for (auto vpn : pages) {
        const auto page = champsim::page_number{vpn};
        auto [pa, penalty] = vm.va_to_pa(cpu, page);
        assert(penalty == vm.minor_fault_penalty);
        assert(vm.va_to_pa(cpu, page).second.count() == 0);
        assert(physical_data.insert(pa.to<uint64_t>()).second);
        assert(pa.to<uint64_t>() < (8ULL << 30) / PAGE_SIZE);
        for (uint64_t offset : {0ULL, 1ULL, 63ULL, 64ULL, 4095ULL}) {
          auto byte_address = champsim::splice(pa, champsim::page_offset{offset}).to<uint64_t>();
          assert(byte_address % PAGE_SIZE == offset);
          assert(byte_address / PAGE_SIZE == pa.to<uint64_t>());
        }
        for (unsigned level = 0; level <= 5; ++level) {
          const uint64_t address = level ? vm.get_pte_pa(cpu, page, level).first.to<uint64_t>() : pa.to<uint64_t>();
          if (level) assert(address % PAGE_SIZE == vm.get_offset(page, level) * 8);
          if (reverse) assert(expected.at({cpu, vpn, level}) == address);
          else expected.emplace(key{cpu, vpn, level}, address);
          if (level) assert(vm.get_pte_pa(cpu, page, level).second.count() == 0);
        }
      }
    }
    assert(roots.size() == NUM_CPUS);
    bool missing = false;
    try { vm.va_to_pa(0, champsim::page_number{1234567}); } catch (const std::out_of_range&) { missing = true; }
    assert(missing);
  }
  unsetenv("CHAMPSIM_PLACEMENT_FILE");
  VirtualMemory a{champsim::data::bytes{4096}, 5, champsim::chrono::picoseconds{50000}, dram, 1};
  VirtualMemory b{champsim::data::bytes{4096}, 5, champsim::chrono::picoseconds{50000}, dram, 1};
  for (uint64_t i = 0; i < 200; ++i) assert(a.va_to_pa(0, champsim::page_number{i}) == b.va_to_pa(0, champsim::page_number{i}));
  std::cout << "PASS placement order independence, address spaces, roots, first-touch penalties, missing keys and legacy determinism\n";
}
