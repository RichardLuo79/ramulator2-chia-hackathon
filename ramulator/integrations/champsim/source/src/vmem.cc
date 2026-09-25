/*
 *    Copyright 2023 The ChampSim Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vmem.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <fmt/core.h>
#include <set>
#include <stdexcept>

#include "champsim.h"
#include "dram_controller.h"
#include "util/bits.h"

using namespace champsim::data::data_literals;

VirtualMemory::VirtualMemory(champsim::data::bytes page_table_page_size, std::size_t page_table_levels, champsim::chrono::clock::duration minor_penalty,
                             MEMORY_CONTROLLER& dram_, std::optional<uint64_t> randomization_seed_)
    : randomization_seed(randomization_seed_), dram(dram_), minor_fault_penalty(minor_penalty), pt_levels(page_table_levels),
      pte_page_size(page_table_page_size),
      next_pte_page(
          champsim::dynamic_extent{champsim::data::bits{LOG2_PAGE_SIZE}, champsim::data::bits{champsim::lg2(champsim::data::bytes{pte_page_size}.count())}}, 0)
{
  assert(pte_page_size > 1_kiB);
  assert(champsim::is_power_of_2(pte_page_size.count()));

  champsim::page_number last_vpage{
      champsim::lowest_address_for_size(champsim::data::bytes{PAGE_SIZE + champsim::ipow(pte_page_size.count(), static_cast<unsigned>(pt_levels))})};
  champsim::data::bits required_bits{LOG2_PAGE_SIZE + champsim::lg2(last_vpage.to<uint64_t>())};
  if (required_bits > champsim::address::bits) {
    fmt::print("[VMEM] WARNING: virtual memory configuration would require {} bits of addressing.\n", required_bits); // LCOV_EXCL_LINE
  }
  if (required_bits > champsim::data::bits{champsim::lg2(dram.size().count())}) {
    fmt::print("[VMEM] WARNING: physical memory size is smaller than virtual memory size.\n"); // LCOV_EXCL_LINE
  }
  populate_pages();
  shuffle_pages();
  if (const char* path = std::getenv("CHAMPSIM_PLACEMENT_FILE"))
    load_placement(path);
}

VirtualMemory::VirtualMemory(champsim::data::bytes page_table_page_size, std::size_t page_table_levels, champsim::chrono::clock::duration minor_penalty,
                             MEMORY_CONTROLLER& dram_)
    : VirtualMemory(page_table_page_size, page_table_levels, minor_penalty, dram_, {})
{
}

void VirtualMemory::populate_pages()
{
  assert(dram.size() > 1_MiB);
  const auto capacity = placement_bytes ? champsim::data::bytes{static_cast<long long>(placement_bytes)} : dram.size();
  ppage_free_list.resize(((capacity - 1_MiB) / PAGE_SIZE).count());
  assert(ppage_free_list.size() != 0);
  champsim::page_number base_address =
      champsim::page_number{champsim::lowest_address_for_size(std::max<champsim::data::mebibytes>(champsim::data::bytes{PAGE_SIZE}, 1_MiB))};
  for (auto it = ppage_free_list.begin(); it != ppage_free_list.end(); it++) {
    *it = base_address;
    base_address++;
  }
}

void VirtualMemory::shuffle_pages()
{
  if (randomization_seed.has_value())
    std::shuffle(ppage_free_list.begin(), ppage_free_list.end(), std::mt19937_64{randomization_seed.value()});
}

champsim::dynamic_extent VirtualMemory::extent(std::size_t level) const
{
  const champsim::data::bits lower{LOG2_PAGE_SIZE + champsim::lg2(pte_page_size.count()) * (level - 1)};
  const auto size = static_cast<std::size_t>(champsim::lg2(pte_page_size.count()));
  return champsim::dynamic_extent{lower, size};
}

champsim::data::bits VirtualMemory::shamt(std::size_t level) const { return extent(level).lower; }

uint64_t VirtualMemory::get_offset(champsim::address vaddr, std::size_t level) const { return champsim::address_slice{extent(level), vaddr}.to<uint64_t>(); }

uint64_t VirtualMemory::get_offset(champsim::page_number vaddr, std::size_t level) const { return get_offset(champsim::address{vaddr}, level); }

champsim::page_number VirtualMemory::ppage_front() const
{
  if (placement_bytes && available_ppages() == 0)
    throw std::runtime_error("placement exceeds physical memory capacity");
  assert(available_ppages() > 0);
  return ppage_free_list.front();
}

void VirtualMemory::ppage_pop()
{
  ppage_free_list.pop_front();
  if (available_ppages() == 0 && !placement_bytes) {
    fmt::print("[VMEM] WARNING: Out of physical memory, freeing ppages\n");
    populate_pages();
    shuffle_pages();
  }
}

std::size_t VirtualMemory::available_ppages() const { return (ppage_free_list.size()); }

std::pair<champsim::page_number, champsim::chrono::clock::duration> VirtualMemory::va_to_pa(uint32_t cpu_num, champsim::page_number vaddr)
{
  const auto key = std::make_pair(cpu_num, champsim::page_number{vaddr});
  auto [ppage, fault] = vpage_to_ppage_map.try_emplace(key, fixed_placement ? fixed_data.at(key) : ppage_front());

  // this vpage doesn't yet have a ppage mapping
  if (fault && !fixed_placement) {
    ppage_pop();
  }

  auto penalty = fault ? minor_fault_penalty : champsim::chrono::clock::duration::zero();

  if constexpr (champsim::debug_print) {
    fmt::print("[VMEM] {} paddr: {} vpage: {} fault: {}\n", __func__, ppage->second, champsim::page_number{vaddr}, fault);
  }

  return std::pair{ppage->second, penalty};
}

std::pair<champsim::address, champsim::chrono::clock::duration> VirtualMemory::get_pte_pa(uint32_t cpu_num, champsim::page_number vaddr, std::size_t level)
{
  champsim::dynamic_extent pte_table_entry_extent{champsim::address::bits, shamt(level + 1)};
  const auto key = std::make_tuple(cpu_num, uint32_t(level), champsim::address_slice{pte_table_entry_extent, vaddr});
  auto [ppage, fault] = page_table.try_emplace(
      key, fixed_placement ? fixed_tables.at(key) : champsim::address{champsim::splice(active_pte_page, next_pte_page)});

  // this PTE doesn't yet have a mapping
  if (fault && !fixed_placement) {
    next_pte_page++;
    if (champsim::page_offset{next_pte_page} == champsim::page_offset{0}) {
      active_pte_page = ppage_front();
      ppage_pop();
    }
  }

  auto offset = get_offset(vaddr, level);
  champsim::address paddr{
      champsim::splice(ppage->second, champsim::address_slice{champsim::dynamic_extent{champsim::data::bits{champsim::lg2(pte_entry::byte_multiple)},
                                                                                       static_cast<std::size_t>(champsim::lg2(pte_page_size.count()))},
                                                              offset})};
  if constexpr (champsim::debug_print) {
    fmt::print("[VMEM] {} paddr: {} vaddr: {} pt_page_offset: {} translation_level: {} fault: {}\n", __func__, paddr, vaddr, offset, level, fault);
  }

  auto penalty = minor_fault_penalty;
  if (!fault) {
    penalty = champsim::chrono::clock::duration::zero();
  }

  return {paddr, penalty};
}

void VirtualMemory::prepare_placement(const std::string& inventory, const std::string& output)
{
  if (fixed_placement)
    throw std::runtime_error("cannot prepare a map from an already fixed placement");
  std::ifstream in(inventory);
  std::string magic;
  uint64_t cores = 0, page_bytes = 0, capacity = 0;
  if (!(in >> magic >> cores >> page_bytes >> capacity) || magic != "CHAMPSIM_PAGES_V1" ||
      cores != NUM_CPUS || page_bytes != PAGE_SIZE || capacity <= (1ULL << 20) ||
      capacity > static_cast<uint64_t>(dram.size().count()) || capacity % PAGE_SIZE ||
      static_cast<uint64_t>(pte_page_size.count() * pte_entry::byte_multiple) != PAGE_SIZE)
    throw std::runtime_error("placement inventory geometry does not match this frontend");
  std::set<std::pair<uint32_t, champsim::page_number>> pages;
  uint32_t cpu;
  uint64_t vpn;
  while (in >> cpu >> vpn) {
    if (cpu >= cores || vpn >= (1ULL << (64 - LOG2_PAGE_SIZE)) ||
        !pages.emplace(cpu, champsim::page_number{vpn}).second)
      throw std::runtime_error("duplicate or invalid inventory page");
  }
  if (!in.eof() || pages.empty())
    throw std::runtime_error("invalid or empty placement inventory");
  std::set<decltype(page_table)::key_type> tables;
  auto add_table = [&](uint32_t core, champsim::page_number page, uint32_t level) {
    tables.emplace(core, level, champsim::address_slice{
        champsim::dynamic_extent{champsim::address::bits, shamt(level + 1)}, page});
  };
  for (uint32_t core = 0; core < cores; ++core)
    add_table(core, champsim::page_number{0}, static_cast<uint32_t>(pt_levels));
  for (const auto& [core, page] : pages)
    for (uint32_t level = 1; level <= pt_levels; ++level)
      add_table(core, page, level);
  placement_bytes = capacity;
  populate_pages();
  shuffle_pages();
  if (tables.size() + pages.size() > available_ppages())
    throw std::runtime_error("data and page tables exceed physical memory capacity");
  std::ofstream out(output);
  out << "CHAMPSIM_PLACEMENT_V1 " << cores << ' ' << PAGE_SIZE << ' '
      << pte_page_size.count() * pte_entry::byte_multiple << ' ' << pt_levels << ' '
      << capacity << ' ' << randomization_seed.value_or(0) << ' ' << pages.size() << ' ' << tables.size() << '\n';
  // One 4 KiB frame per 512-entry page table. Reuse the allocator's shuffled
  // free-frame pool, but do not pack distinct tables into overlapping offsets.
  for (const auto& [core, level, prefix] : tables) {
    out << "P " << core << ' ' << level << ' ' << prefix.to<uint64_t>() << ' '
        << (ppage_front().to<uint64_t>() << LOG2_PAGE_SIZE) << '\n';
    ppage_pop();
  }
  for (const auto& [core, page] : pages) {
    out << "D " << core << ' ' << page.to<uint64_t>() << ' ' << ppage_front().to<uint64_t>() << '\n';
    ppage_pop();
  }
  out.close();
  if (!out)
    throw std::runtime_error("failed to write placement");
  fmt::print("[VMEM] Prepared placement: data={} tables={} bytes={}\n", pages.size(), tables.size(), capacity);
}

void VirtualMemory::load_placement(const std::string& path)
{
  std::ifstream in(path);
  std::string magic;
  uint64_t cores = 0, page_bytes = 0, table_bytes = 0, levels = 0, capacity = 0, seed = 0, data_count = 0, table_count = 0;
  if (!(in >> magic >> cores >> page_bytes >> table_bytes >> levels >> capacity >> seed >> data_count >> table_count) ||
      magic != "CHAMPSIM_PLACEMENT_V1" || cores != NUM_CPUS || page_bytes != PAGE_SIZE ||
      table_bytes != static_cast<uint64_t>(pte_page_size.count() * pte_entry::byte_multiple) || table_bytes != PAGE_SIZE ||
      levels != pt_levels || seed != randomization_seed.value_or(0) || capacity <= (1ULL << 20) ||
      capacity > static_cast<uint64_t>(dram.size().count()) || capacity % PAGE_SIZE || !data_count || !table_count ||
      data_count + table_count > (capacity - (1ULL << 20)) / PAGE_SIZE)
    throw std::runtime_error("placement header does not match this frontend");
  std::set<uint64_t> frames;
  auto check_frame = [&](uint64_t frame) {
    if (frame < (1ULL << 20) / PAGE_SIZE || frame >= capacity / PAGE_SIZE || !frames.insert(frame).second)
      throw std::runtime_error("placement has an aliased or out-of-range frame");
  };
  for (uint64_t i = 0; i < table_count; ++i) {
    char kind;
    uint32_t cpu = 0, level = 0;
    uint64_t prefix = 0, base = 0;
    if (!(in >> kind >> cpu >> level >> prefix >> base) || kind != 'P' || cpu >= cores ||
        !level || level > pt_levels || base % PAGE_SIZE)
      throw std::runtime_error("invalid placement page table");
    auto extent = champsim::dynamic_extent{champsim::address::bits, shamt(level + 1)};
    auto slice = champsim::address_slice{extent, prefix};
    if (slice.to<uint64_t>() != prefix || !fixed_tables.emplace(std::make_tuple(cpu, level, slice), champsim::address{base}).second)
      throw std::runtime_error("duplicate or invalid page-table key");
    check_frame(base / PAGE_SIZE);
  }
  for (uint64_t i = 0; i < data_count; ++i) {
    char kind;
    uint32_t cpu = 0;
    uint64_t vpn = 0, ppn = 0;
    if (!(in >> kind >> cpu >> vpn >> ppn) || kind != 'D' || cpu >= cores || vpn >= (1ULL << (64 - LOG2_PAGE_SIZE)) ||
        !fixed_data.emplace(std::make_pair(cpu, champsim::page_number{vpn}), champsim::page_number{ppn}).second)
      throw std::runtime_error("invalid placement data page");
    check_frame(ppn);
  }
  std::string extra;
  if (in >> extra)
    throw std::runtime_error("unexpected trailing placement records");
  // These maps record actual first touches, not addresses allocated during preparation.
  vpage_to_ppage_map.clear();
  page_table.clear();
  placement_bytes = capacity;
  fixed_placement = true;
  ppage_free_list.clear();
  fmt::print("[VMEM] Fixed placement loaded: data={} tables={} bytes={}\n", data_count, table_count, capacity);
}
