#ifndef RAMULATOR_CONTROLLER_IMPL_ZOO_PROBE_H
#define RAMULATOR_CONTROLLER_IMPL_ZOO_PROBE_H

#include <algorithm>

#include "ramulator/base/request.h"
#include "ramulator/dram/device.h"

namespace Ramulator {

// Standard-agnostic probe used by the model-zoo baseline controllers: measure
// the cross-bank (cross-bank-group where the standard has groups) open-row
// RD->RD gap — the data-burst occupancy of the shared channel in controller
// ticks, i.e. the peak-bandwidth service time. Same-bank gaps would fold in
// tCCD_L and understate channel capacity. The probe walks prerequisite
// chains in the device's early history, long before simulated traffic
// starts, so it does not perturb simulation timing.
inline Clk_t zoo_probe_burst_gap(DRAMDevice& dev) {
  const auto* spec = dev.m_spec;
  const int rd = spec->supported_requests[Request::Type::Read];
  AddrVec_t av_a(spec->level_count, 0);
  AddrVec_t av_b(spec->level_count, 0);
  const int far_level = spec->has_level("BankGroup") ? spec->get_level_id("BankGroup")
                                                     : spec->get_level_id("Bank");
  av_b[far_level] = 1;

  // Open both rows (issue everything up to, but not including, the RD).
  for (const auto* av : {&av_a, &av_b}) {
    Clk_t t = 1000;
    for (;;) {
      int cmd = dev.get_preq_command(rd, *av, t);
      if (cmd == rd) {
        break;
      }
      Clk_t e = std::max(t, dev.earliest_ready(cmd, *av));
      dev.issue_command(cmd, *av, e);
      t = e + 1;
    }
  }
  Clk_t t = std::max<Clk_t>(2000, dev.earliest_ready(rd, av_a));
  dev.issue_command(rd, av_a, t);
  return std::max<Clk_t>(dev.earliest_ready(rd, av_b) - t, 1);
}

}  // namespace Ramulator

#endif  // RAMULATOR_CONTROLLER_IMPL_ZOO_PROBE_H
