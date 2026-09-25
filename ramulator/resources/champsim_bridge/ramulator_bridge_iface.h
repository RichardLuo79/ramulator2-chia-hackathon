#ifndef CHAMPSIM_RAMULATOR_BRIDGE_IFACE_H
#define CHAMPSIM_RAMULATOR_BRIDGE_IFACE_H

#include <vector>

namespace champsim
{
class channel;
}

// C++17-safe interface to the Ramulator2 backend (implementation is a
// separate C++20 translation unit). active() checks RAMULATOR_MODEL/
// RAMULATOR_STD once and constructs the bridge on first call.
namespace ramulator_bridge
{
bool active();
long warmup_cycle(std::vector<champsim::channel*>& queues);
long cycle(std::vector<champsim::channel*>& queues);
void measurement_begin();
void measurement_end(unsigned cpu);
void measurement_report();
} // namespace ramulator_bridge

#endif
