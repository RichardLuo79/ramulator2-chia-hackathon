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
  size_t measured_reads = 0, measured_writes = 0;
  size_t measured_reads_completed = 0, measured_writes_completed = 0;
  size_t warmup_reads_outstanding = 0, warmup_writes_outstanding = 0;
  Clk_t measurement_start_cycle = 0, final_admission_cycle = 0;
  uint64_t measured_admission_wait_sum = 0;
  Clk_t measured_admission_wait_max = 0;
  uint64_t admission_digest = 0, completion_digest = 0;
  double warmup_wall_s = 0, measured_wall_s = 0, drain_wall_s = 0;
  double warmup_cpu_s = 0, measured_cpu_s = 0, drain_cpu_s = 0;
};

// Opt-in standalone timing. The old overload below keeps its original replay
// and recording defaults. Warmup is a prefix, not a separate drained batch.
struct BatchOptions {
  bool drain_writes = true;
  bool record_requests = true;
  bool skip_idle_retries = false;
  bool tick_only = false;  // Qualification reference, never a model change.
  size_t warmup_requests = 0;
};
BatchResult run_batch(IMemorySystem& memory, const std::vector<BatchRequest>& input,
                      BatchState& state, const BatchOptions& options);

// Shared array replay, with no frontend ticking. Offered arrivals are absolute
// controller-clock ticks; backpressure delays admission, not the input schedule.
// The legacy Python API drains reads only and returns -1 for writes. The isolated
// diagnostic explicitly drains both callback populations.
BatchResult run_batch(IMemorySystem& memory, const std::vector<BatchRequest>& input,
                      BatchState& state, bool drain_writes = false);

// The trusted evaluator emits this plain numeric table before loading a model.
// No completion data or future traffic is passed through the model value API.
std::vector<BatchRequest> read_batch(std::istream& stream);
// Compact immutable standalone inputs: RMSPD001, LE uint64 count, then count
// (LE uint64 physical address, uint8 type) records. Offered time = index*interval.
std::vector<BatchRequest> read_speed_batch(std::istream& stream, Clk_t interval);
void write_batch(std::ostream& stream, const std::vector<BatchRequest>& input,
                 const BatchResult& result, std::int64_t first_frontend_id);

}  // namespace Ramulator
#endif
