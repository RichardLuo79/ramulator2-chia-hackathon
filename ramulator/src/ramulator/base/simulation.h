#ifndef RAMULATOR_BASE_SIMULATION_H
#define RAMULATOR_BASE_SIMULATION_H

namespace Ramulator {

class IFrontEnd;
class IMemorySystem;

// Run already-connected components to frontend completion. Construction,
// observation and finalization belong to the caller so isolated runners can
// load inputs before admitting a candidate library.
void run_simulation(IFrontEnd& frontend, IMemorySystem& memory_system);

}  // namespace Ramulator

#endif  // RAMULATOR_BASE_SIMULATION_H
