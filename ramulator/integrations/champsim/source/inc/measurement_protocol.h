#ifndef CHAMPSIM_MEASUREMENT_PROTOCOL_H
#define CHAMPSIM_MEASUREMENT_PROTOCOL_H

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace champsim {
inline bool background_replay_enabled()
{
  static const bool enabled = [] {
    const char* setting = std::getenv("CHAMPSIM_COMPLETION_POLICY");
    const std::string policy = setting ? setting : "finite";
    if (policy != "finite" && policy != "background-replay")
      throw std::invalid_argument("unknown CHAMPSIM_COMPLETION_POLICY");
    return policy == "background-replay";
  }();
  return enabled;
}

// Sample at the existing ten-million-cycle cadence. Low IPC with positive
// retirement is valid; two empty windows are a reported no-progress failure.
struct retirement_watchdog {
  std::uint64_t last = 0;
  unsigned empty_windows = 0;
  bool sample(std::uint64_t retired)
  {
    empty_windows = retired == last ? empty_windows + 1 : 0;
    last = retired;
    return empty_windows >= 2;
  }
};
} // namespace champsim
#endif
