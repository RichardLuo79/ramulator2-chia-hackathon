#include "ramulator/base/batch_simulation.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ramulator/base/request.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace Ramulator {

BatchResult run_batch(IMemorySystem& memory, const std::vector<BatchRequest>& input,
                      BatchState& state, bool drain_writes) {
  BatchOptions options;
  options.drain_writes = drain_writes;
  return run_batch(memory, input, state, options);
}

BatchResult run_batch(IMemorySystem& memory, const std::vector<BatchRequest>& input,
                      BatchState& state, const BatchOptions& options) {
  if (options.warmup_requests > input.size()) {
    throw std::invalid_argument("batch warmup exceeds input population");
  }
  Clk_t previous = -1;
  for (const auto& entry : input) {
    if (entry.address < 0 || entry.arrival < 0 || entry.arrival < previous || entry.source < -1 ||
        (entry.type != Request::Type::Read && entry.type != Request::Type::Write)) {
      throw std::invalid_argument("batch requests need valid types/addresses and ordered arrivals");
    }
    previous = entry.arrival;
  }
  if (state.clock < 0 || state.next_frontend_id < 0 ||
      input.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - state.next_frontend_id)) {
    throw std::invalid_argument("invalid or exhausted batch state");
  }
  BatchResult result;
  if (options.record_requests) {
    result.admitted.assign(input.size(), -1);
    result.departed.assign(input.size(), -1);
  }
  size_t expected_reads = 0, expected_writes = 0;
  const int tx_bytes = memory.get_tx_bytes();
  // Aggregate digests include identity and event time; recording may be off.
  // They are repeatability checks, not cryptographic evidence identities.
  auto digest = [](uint64_t value) {
    value ^= value >> 30; value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27; value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  auto tick = [&]() {
    if (state.clock == std::numeric_limits<Clk_t>::max()) {
      throw std::overflow_error("batch clock exhausted");
    }
    ++state.clock;
    memory.tick();
  };
  auto jump = [&](Clk_t ticks) {
    if (ticks < 0 || ticks > std::numeric_limits<Clk_t>::max() - state.clock) {
      throw std::overflow_error("invalid batch idle interval");
    }
    memory.fast_forward(ticks);
    state.clock += ticks;
  };
  auto advance = [&](Clk_t ticks) {
    while (ticks > 0) {
      const Clk_t idle = options.tick_only ? 0 : memory.idle_ticks();
      if (idle < 0) throw std::runtime_error("memory returned a negative idle interval");
      const Clk_t skipped = std::min(idle, ticks - 1);
      if (skipped > 0) {
        jump(skipped);
        ticks -= skipped;
      } else {
        tick();
        --ticks;
      }
    }
  };
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto cpu_start = std::clock();
  auto measured_start = start;
  auto measured_cpu_start = cpu_start;
  auto boundary = [&] {
    result.measurement_start_cycle = state.clock;
    result.warmup_reads_outstanding = expected_reads - result.completed_reads;
    result.warmup_writes_outstanding = expected_writes - result.completed_writes;
    measured_start = Clock::now();
    measured_cpu_start = std::clock();
  };
  for (size_t i = 0; i < input.size(); ++i) {
    if (i == options.warmup_requests) boundary();
    const auto& entry = input[i];
    if (entry.arrival > state.clock) advance(entry.arrival - state.clock);
    Request request(entry.address, entry.type);
    request.source_id = entry.source;
    request.frontend_id = state.next_frontend_id++;
    request.size_bytes = tx_bytes;
    if (entry.type == Request::Type::Read) {
      ++expected_reads;
      request.callback = [&, i](Request& done) {
        if (options.record_requests) result.departed[i] = done.depart;
        result.completion_digest += digest(i ^ (static_cast<uint64_t>(done.depart) << 24));
        ++result.completed_reads;
        if (i >= options.warmup_requests) ++result.measured_reads_completed;
      };
    } else if (options.drain_writes) {
      ++expected_writes;
      request.callback = [&, i](Request&) {
        // Some controllers acknowledge/coalesce writes without setting depart.
        // Record callback time, not an invented physical write-burst completion.
        if (options.record_requests) result.departed[i] = state.clock;
        result.completion_digest += digest(i ^ (static_cast<uint64_t>(state.clock) << 24));
        ++result.completed_writes;
        if (i >= options.warmup_requests) ++result.measured_writes_completed;
      };
    }
    while (!memory.send(request)) {
      // idle_ticks guarantees that no service/state transition is skipped.
      // Oracle controllers return zero and still execute every required tick.
      const Clk_t idle = options.skip_idle_retries && !options.tick_only ? memory.idle_ticks() : 0;
      if (idle < 0 || idle == std::numeric_limits<Clk_t>::max())
        throw std::runtime_error("rejected batch request has no finite next service event");
      if (idle > 0) jump(idle);
      tick();
    }
    if (options.record_requests) result.admitted[i] = state.clock;
    result.admission_digest += digest(i ^ (static_cast<uint64_t>(state.clock) << 24));
    if (i >= options.warmup_requests) {
      if (entry.type == Request::Type::Read) ++result.measured_reads;
      else ++result.measured_writes;
      const auto wait = state.clock - entry.arrival;
      result.measured_admission_wait_sum += static_cast<uint64_t>(wait);
      result.measured_admission_wait_max = std::max(result.measured_admission_wait_max, wait);
    }
  }
  if (options.warmup_requests == input.size()) boundary();
  result.final_admission_cycle = state.clock;
  const auto drain_start = Clock::now();
  const auto drain_cpu_start = std::clock();
  while (result.completed_reads < expected_reads || result.completed_writes < expected_writes) {
    const Clk_t idle = options.tick_only ? 0 : memory.idle_ticks();
    if (idle == std::numeric_limits<Clk_t>::max()) {
      throw std::runtime_error("batch has outstanding callbacks but memory reports no events");
    }
    if (idle < 0) throw std::runtime_error("memory returned a negative idle interval");
    if (idle > 0) jump(idle);
    tick();
  }
  const auto end = Clock::now();
  const auto cpu_end = std::clock();
  result.warmup_wall_s = std::chrono::duration<double>(measured_start - start).count();
  result.measured_wall_s = std::chrono::duration<double>(end - measured_start).count();
  result.drain_wall_s = std::chrono::duration<double>(end - drain_start).count();
  result.warmup_cpu_s = double(measured_cpu_start - cpu_start) / CLOCKS_PER_SEC;
  result.measured_cpu_s = double(cpu_end - measured_cpu_start) / CLOCKS_PER_SEC;
  result.drain_cpu_s = double(cpu_end - drain_cpu_start) / CLOCKS_PER_SEC;
  result.elapsed = state.clock;
  return result;
}

std::vector<BatchRequest> read_batch(std::istream& stream) {
  std::string line;
  if (!std::getline(stream, line) || line != "arrive,addr,type,source") {
    throw std::invalid_argument("batch input requires arrive,addr,type,source header");
  }
  std::vector<BatchRequest> result;
  while (std::getline(stream, line)) {
    std::istringstream row(line);
    BatchRequest entry{};
    char a, b, c;
    if (!(row >> entry.arrival >> a >> entry.address >> b >> entry.type >> c >> entry.source) ||
        a != ',' || b != ',' || c != ',' || !(row >> std::ws).eof()) {
      throw std::invalid_argument("malformed numeric batch input row");
    }
    result.push_back(entry);
  }
  if (!stream.eof()) throw std::runtime_error("batch input read failed");
  return result;
}

std::vector<BatchRequest> read_speed_batch(std::istream& stream, Clk_t interval) {
  char header[16];
  if (!stream.read(header, sizeof(header)) || std::string(header, 8) != "RMSPD001" || interval <= 0)
    throw std::invalid_argument("invalid compact speed input header/interval");
  auto u64 = [](const char* data) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(static_cast<unsigned char>(data[i])) << (8*i);
    return value;
  };
  const auto count = u64(header + 8);
  if (count == 0 || count > 100000000 ||
      count - 1 > uint64_t(std::numeric_limits<Clk_t>::max() / interval))
    throw std::invalid_argument("invalid compact speed input population");
  std::vector<char> bytes(count * 9);
  if (!stream.read(bytes.data(), bytes.size()) || stream.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("truncated or trailing compact speed input");
  std::vector<BatchRequest> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto address = u64(bytes.data() + i*9);
    const auto type = static_cast<unsigned char>(bytes[i*9 + 8]);
    if (address >= (1ULL << 33) || address % 64 || type > 1)
      throw std::invalid_argument("compact speed input exceeds frozen 8-GiB/64-byte geometry");
    result.push_back({static_cast<Addr_t>(address), type, static_cast<Clk_t>(i)*interval, 0});
  }
  return result;
}

void write_batch(std::ostream& stream, const std::vector<BatchRequest>& input,
                 const BatchResult& result, std::int64_t first_frontend_id) {
  if (input.size() != result.admitted.size() || input.size() != result.departed.size()) {
    throw std::invalid_argument("batch output population differs from its input");
  }
  stream << "arrive,admit,depart,type,source,addr,frontend_id,frontend_sub_id\n";
  for (size_t i = 0; i < input.size(); ++i) {
    const auto& entry = input[i];
    stream << entry.arrival << ',' << result.admitted[i] << ',' << result.departed[i] << ','
           << entry.type << ',' << entry.source << ',' << entry.address << ','
           << first_frontend_id + static_cast<int64_t>(i) << ",0\n";
  }
  if (!stream) throw std::runtime_error("batch output write failed");
}

}  // namespace Ramulator
