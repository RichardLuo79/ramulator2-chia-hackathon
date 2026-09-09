#ifndef RAMULATOR_BASE_BATCH_SIMULATION_H
#define RAMULATOR_BASE_BATCH_SIMULATION_H

#include <cstdint>
#include <cstddef>
#include <iosfwd>
#include <vector>

#include "ramulator/base/type.h"

namespace Ramulator {

class IMemorySystem;

struct BatchRequest {
  Addr_t address;
  int type;
  Clk_t arrival;
  int source = -1;
};

struct BatchState {
  Clk_t clock = 0;
  std::int64_t next_frontend_id = 0;
};

struct BatchResult {
  std::vector<Clk_t> admitted;
  std::vector<Clk_t> departed;
  Clk_t elapsed = 0;
  size_t completed_reads = 0;
  size_t completed_writes = 0;
};

// Shared array replay, with no frontend ticking. Offered arrivals are absolute
// controller-clock ticks; backpressure delays admission, not the input schedule.
// The legacy Python API drains reads only and returns -1 for writes. The isolated
// diagnostic explicitly drains both callback populations.
BatchResult run_batch(IMemorySystem& memory, const std::vector<BatchRequest>& input,
                      BatchState& state, bool drain_writes = false);

// The trusted evaluator emits this plain numeric table before loading a model.
// No completion data or future traffic is passed through the model value API.
std::vector<BatchRequest> read_batch(std::istream& stream);
void write_batch(std::ostream& stream, const std::vector<BatchRequest>& input,
                 const BatchResult& result, std::int64_t first_frontend_id);

}  // namespace Ramulator
#endif
