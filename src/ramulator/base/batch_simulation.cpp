#include "ramulator/base/batch_simulation.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ramulator/base/request.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace Ramulator {

BatchResult run_batch(IMemorySystem& memory, const std::vector<BatchRequest>& input,
                      BatchState& state, bool drain_writes) {
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
  result.admitted.assign(input.size(), -1);
  result.departed.assign(input.size(), -1);
  size_t expected_reads = 0, expected_writes = 0;
  const int tx_bytes = memory.get_tx_bytes();
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
      const Clk_t idle = memory.idle_ticks();
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
  for (size_t i = 0; i < input.size(); ++i) {
    const auto& entry = input[i];
    if (entry.arrival > state.clock) advance(entry.arrival - state.clock);
    Request request(entry.address, entry.type);
    request.source_id = entry.source;
    request.frontend_id = state.next_frontend_id++;
    request.size_bytes = tx_bytes;
    if (entry.type == Request::Type::Read) {
      ++expected_reads;
      request.callback = [&, i](Request& done) {
        result.departed[i] = done.depart;
        ++result.completed_reads;
      };
    } else if (drain_writes) {
      ++expected_writes;
      request.callback = [&, i](Request&) {
        // Some controllers acknowledge/coalesce writes without setting depart.
        // Record callback time, not an invented physical write-burst completion.
        result.departed[i] = state.clock;
        ++result.completed_writes;
      };
    }
    while (!memory.send(request)) advance(1);
    result.admitted[i] = state.clock;
  }
  while (result.completed_reads < expected_reads || result.completed_writes < expected_writes) {
    const Clk_t idle = memory.idle_ticks();
    if (idle == std::numeric_limits<Clk_t>::max()) {
      throw std::runtime_error("batch has outstanding callbacks but memory reports no events");
    }
    if (idle < 0) throw std::runtime_error("memory returned a negative idle interval");
    if (idle > 0) jump(idle);
    tick();
  }
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
