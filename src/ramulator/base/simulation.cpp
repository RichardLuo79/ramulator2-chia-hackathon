#include "ramulator/base/simulation.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace Ramulator {

void run_simulation(IFrontEnd& frontend, IMemorySystem& memory_system) {
  const int fe_tick = frontend.get_clock_ratio();
  const int mem_tick = memory_system.get_clock_ratio();
  if (fe_tick <= 0 || mem_tick <= 0) {
    throw std::runtime_error("clock_ratio must be > 0 for both frontend and memory system");
  }

  constexpr Clk_t MAX_JUMP = 1'000'000'000;
  int fe_count = mem_tick - 1, mem_count = fe_tick - 1;
  for (;;) {
    if (++fe_count >= mem_tick) {
      fe_count = 0;
      frontend.tick();
    }

    if (frontend.is_finished()) {
      break;
    }

    if (++mem_count >= fe_tick) {
      mem_count = 0;
      memory_system.tick();
    }

    // From post-iteration counter c, tick k of a side with period P occurs
    // after (P - c) + (k - 1) * P interleave steps. Elide only stretches that
    // both components declare idle; this is the same algebra as the binding.
    Clk_t mem_idle = memory_system.idle_ticks();
    if (mem_idle > 0 && !frontend.is_finished()) {
      mem_idle = std::min(mem_idle, MAX_JUMP);
      const uint64_t j_mem =
          static_cast<uint64_t>(fe_tick - mem_count) + static_cast<uint64_t>(mem_idle) * fe_tick - 1;
      const uint64_t max_fe = (static_cast<uint64_t>(fe_count) + j_mem) / mem_tick;
      if (max_fe > 0) {
        Clk_t fe_idle = frontend.idle_ticks(static_cast<Clk_t>(std::min<uint64_t>(max_fe, MAX_JUMP)));
        if (fe_idle > 0) {
          const uint64_t j_fe =
              static_cast<uint64_t>(mem_tick - fe_count) + static_cast<uint64_t>(fe_idle) * mem_tick - 1;
          const uint64_t j = std::min(j_fe, j_mem);
          if (j >= 2) {
            const uint64_t fe_ff = (static_cast<uint64_t>(fe_count) + j) / mem_tick;
            const uint64_t mem_ff = (static_cast<uint64_t>(mem_count) + j) / fe_tick;
            fe_count = static_cast<int>((static_cast<uint64_t>(fe_count) + j) % mem_tick);
            mem_count = static_cast<int>((static_cast<uint64_t>(mem_count) + j) % fe_tick);
            if (fe_ff > 0) {
              frontend.fast_forward(static_cast<Clk_t>(fe_ff));
            }
            if (mem_ff > 0) {
              memory_system.fast_forward(static_cast<Clk_t>(mem_ff));
            }
          }
        }
      }
    }
  }
}

}  // namespace Ramulator
