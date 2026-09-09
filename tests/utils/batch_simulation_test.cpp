// Shared batch kernel vs a tick-only reference, including admission and writes.
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "ramulator/base/batch_simulation.h"
#include "ramulator/memory_system/i_memory_system.h"

using namespace Ramulator;

class TestMemory final : public IMemorySystem {
 public:
  struct Event { Clk_t due; Request request; };
  std::vector<Event> pending;
  std::vector<std::tuple<Addr_t, int, Clk_t, int64_t, int>> admissions;
  Clk_t clock = 0;
  int capacity = 3;
  bool send(Request& request) override {
    if (pending.size() >= static_cast<size_t>(capacity)) return false;
    admissions.emplace_back(request.addr, request.type_id, clock, request.frontend_id, request.source_id);
    if (request.size_bytes != 64) throw std::runtime_error("batch did not set request size");
    if (request.type_id == Request::Type::Write && request.addr % 5 == 0) {
      if (request.callback) request.callback(request);  // Inline coalesced acknowledgement.
    } else {
      pending.push_back({clock + 1 + request.addr / 64 % 9, request});
    }
    return true;
  }
  void tick() override {
    ++clock;
    for (size_t i = 0; i < pending.size();) {
      if (pending[i].due > clock) { ++i; continue; }
      auto event = pending[i];
      pending.erase(pending.begin() + i);
      if (event.request.type_id == Request::Type::Read) event.request.depart = event.due;
      if (event.request.callback) event.request.callback(event.request);
    }
  }
  Clk_t idle_ticks() override {
    if (pending.empty()) return std::numeric_limits<Clk_t>::max();
    auto first = std::min_element(pending.begin(), pending.end(), [](const auto& a, const auto& b) { return a.due < b.due; });
    return std::max<Clk_t>(0, first->due - clock - 1);
  }
  void fast_forward(Clk_t ticks) override { clock += ticks; }
  int get_clock_ratio() override { return 1; }
  int get_tx_bytes() override { return 64; }
};

static BatchResult reference(TestMemory& memory, const std::vector<BatchRequest>& input,
                             BatchState& state, bool writes) {
  BatchResult result;
  result.admitted.resize(input.size(), -1);
  result.departed.resize(input.size(), -1);
  size_t reads = 0, write_count = 0;
  auto tick = [&] { ++state.clock; memory.tick(); };
  for (size_t i = 0; i < input.size(); ++i) {
    const auto& row = input[i];
    while (state.clock < row.arrival) tick();
    Request request(row.address, row.type);
    request.frontend_id = state.next_frontend_id++;
    request.source_id = row.source;
    request.size_bytes = 64;
    if (row.type == Request::Type::Read) {
      ++reads;
      request.callback = [&, i](Request& done) { result.departed[i] = done.depart; ++result.completed_reads; };
    } else if (writes) {
      ++write_count;
      request.callback = [&, i](Request&) { result.departed[i] = state.clock; ++result.completed_writes; };
    }
    while (!memory.send(request)) tick();
    result.admitted[i] = state.clock;
  }
  while (result.completed_reads < reads || result.completed_writes < write_count) tick();
  result.elapsed = state.clock;
  return result;
}

int main() {
  std::mt19937 random(1731);
  for (int test = 0; test < 500; ++test) {
    TestMemory actual_memory, expected_memory;
    actual_memory.capacity = expected_memory.capacity = 1 + random() % 8;
    BatchState actual_state, expected_state;
    for (int call = 0; call < 2; ++call) {
      std::vector<BatchRequest> input;
      Clk_t arrival = call * 1000;
      for (int i = 0; i < 100; ++i) {
        arrival += random() % 20;
        input.push_back({static_cast<Addr_t>((random() % 128) * 64), static_cast<int>(random() % 2), arrival, i % 3});
      }
      const bool writes = test % 2 == 0;
      auto expected = reference(expected_memory, input, expected_state, writes);
      auto actual = run_batch(actual_memory, input, actual_state, writes);
      if (actual.admitted != expected.admitted || actual.departed != expected.departed ||
          actual.elapsed != expected.elapsed || actual.completed_reads != expected.completed_reads ||
          actual.completed_writes != expected.completed_writes || actual_memory.admissions != expected_memory.admissions ||
          actual_state.clock != actual_memory.clock || actual_state.next_frontend_id != expected_state.next_frontend_id) {
        throw std::runtime_error("shared batch differs from tick-only reference");
      }
    }
  }
  for (const auto& input : std::vector<std::vector<BatchRequest>>{
        {{0, 2, 0}}, {{-1, 0, 0}}, {{0, 0, -1}}, {{0, 0, 1}, {0, 0, 0}}, {{0, 0, 0, -2}}}) {
    TestMemory memory;
    BatchState state;
    bool rejected = false;
    try { run_batch(memory, input, state); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected || !memory.admissions.empty()) throw std::runtime_error("invalid input was admitted");
  }
  for (const std::string text : {"bad\n", "arrive,addr,type,source\n1,64,0,0junk\n", "arrive,addr,type,source\n1,64,0\n"}) {
    std::istringstream stream(text);
    bool rejected = false;
    try { read_batch(stream); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid batch table accepted");
  }
  std::istringstream stream("arrive,addr,type,source\n0,64,0,1\n0,320,1,2\n");
  auto input = read_batch(stream);
  TestMemory memory;
  BatchState state;
  auto result = run_batch(memory, input, state, true);
  std::ostringstream output;
  write_batch(output, input, result, 0);
  if (result.completed_reads != 1 || result.completed_writes != 1 ||
      output.str().find("0,0,0,1,2,320,1,0") == std::string::npos) {
    throw std::runtime_error("inline write acknowledgement or batch table failed");
  }
  std::cout << "1000 batch parity calls passed, including persistent clock, writes and backpressure\n";
}
