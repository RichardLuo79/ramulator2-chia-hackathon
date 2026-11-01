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
#include "ramulator/frontend/lat_tp_addresses.h"
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
  // Full queues, simultaneous arrivals, inline writes, earlier completions,
  // and completions spanning the warmup boundary. Neither recording nor idle
  // skipping may change a single admitted/departed timestamp.
  for (int test = 0; test < 100; ++test) {
    std::vector<BatchRequest> requests;
    for (int i = 0; i < 300; ++i)
      requests.push_back({static_cast<Addr_t>((random()%200)*64), int(random()%2),
                         static_cast<Clk_t>((i/10)*20), 0});
    TestMemory reference_memory;
    BatchState reference_state;
    BatchOptions reference_options;
    reference_options.tick_only = true;
    reference_options.warmup_requests = 100;
    const auto expected = run_batch(reference_memory, requests, reference_state, reference_options);
    for (const bool recording : {true, false}) {
      TestMemory memory;
      BatchState state;
      auto options = reference_options;
      options.tick_only = false;
      options.skip_idle_retries = true;
      options.record_requests = recording;
      const auto actual = run_batch(memory, requests, state, options);
      if ((recording && (actual.admitted != expected.admitted || actual.departed != expected.departed)) ||
          (!recording && (!actual.admitted.empty() || !actual.departed.empty())) ||
          actual.admission_digest != expected.admission_digest || actual.completion_digest != expected.completion_digest ||
          actual.elapsed != expected.elapsed || actual.measurement_start_cycle != expected.measurement_start_cycle ||
          actual.final_admission_cycle != expected.final_admission_cycle ||
          actual.warmup_reads_outstanding != expected.warmup_reads_outstanding ||
          actual.warmup_writes_outstanding != expected.warmup_writes_outstanding ||
          actual.measured_reads_completed + actual.measured_writes_completed != 200 ||
          actual.measured_admission_wait_sum != expected.measured_admission_wait_sum ||
          memory.admissions != reference_memory.admissions ||
          actual.measured_wall_s < actual.drain_wall_s)
        throw std::runtime_error("standalone batch option parity failed");
    }
  }
  // Address boundaries, burst-column units and all 32 banks. Inverse bit
  // decomposition follows the existing RoBaRaCoCh mapper (6-byte-offset bits,
  // 6 column-slot bits, 0 rank bits, 3 bank-group bits, 2 bank bits).
  const std::vector<int> positions{1,3,2}, counts{1,4,8}, levels{0,1,8,4};
  for (int row : {0,1,65535}) for (int bank=0; bank<4; ++bank)
    for (int group=0; group<8; ++group) for (int slot : {0,1,63}) {
      AddrVec_t av{0,0,group,bank,row,slot*16};
      auto address = LatTp::physical_byte_address(av, levels,4,5,65536,64,16,64);
      if (address < 0 || address >= (1LL<<33) || address%64 ||
          ((address>>6)&63) != slot || ((address>>12)&7) != group ||
          ((address>>15)&3) != bank || (address>>17) != row)
        throw std::runtime_error("Lat-Tp physical roundtrip failed");
    }
  for (size_t i=0; i<10000; ++i) {
    const auto av = LatTp::streaming(i,6,positions,counts,32,4,5,65536,16,64,true);
    const size_t flat=i%32, local=i/32+flat*2;
    if (av != AddrVec_t{0,0,int(flat%8),int(flat/8),int(local/64),int(local%64)*16})
      throw std::runtime_error("legacy streaming recipe changed");
  }
  std::mt19937_64 helper_rng(12345), legacy_rng(12345);
  for (int i=0; i<10000; ++i) {
    auto av = LatTp::random(helper_rng,6,positions,counts,4,5,65536,64,16);
    AddrVec_t old(6,0);
    for (size_t j=0; j<positions.size(); ++j)
      old[positions[j]] = std::uniform_int_distribution<int>(0,counts[j]-1)(legacy_rng);
    old[4] = std::uniform_int_distribution<int>(0,65535)(legacy_rng);
    old[5] = std::uniform_int_distribution<int>(0,63)(legacy_rng)*16;
    if (old != av || legacy_rng != helper_rng) throw std::runtime_error("legacy random draws changed");
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
