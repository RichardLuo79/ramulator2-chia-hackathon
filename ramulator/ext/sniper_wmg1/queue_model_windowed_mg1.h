// Sniper WMG1 adapter declarations. See LICENSE, UPSTREAM.json and the patch.
// No Sniper runtime, global singleton or statistics registry is imported.
#ifndef RAMULATOR_SNIPER_WMG1_H
#define RAMULATOR_SNIPER_WMG1_H

#include <algorithm>
#include <compare>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace Ramulator::Sniper {

using core_id_t = int;
using UInt64 = uint64_t;

// The kernel uses picoseconds. Negative cutoff times are intentional at
// startup; unlike simulated timestamps they must not wrap to UINT64_MAX.
// PS() truncates nonnegative floating-point delays just like upstream PS().
class SubsecondTime {
 public:
  static SubsecondTime PS(int64_t ps) { return SubsecondTime(ps); }
  static SubsecondTime Zero() { return PS(0); }
  uint64_t getPS() const { return m_ps; }
  auto operator<=>(const SubsecondTime&) const = default;
  SubsecondTime operator-(SubsecondTime other) const { return PS(m_ps - other.m_ps); }
  SubsecondTime& operator+=(SubsecondTime other) { m_ps += other.m_ps; return *this; }
 private:
  explicit SubsecondTime(int64_t ps) : m_ps(ps) {}
  int64_t m_ps;
};

class QueueModelWindowedMG1 {
 public:
  explicit QueueModelWindowedMG1(SubsecondTime window);
  ~QueueModelWindowedMG1();
  SubsecondTime computeQueueDelay(SubsecondTime pkt_time, SubsecondTime processing_time,
                                 core_id_t requester = -1);
 private:
  const SubsecondTime m_window_size;
  UInt64 m_total_requests;
  SubsecondTime m_total_utilized_time;
  SubsecondTime m_total_queue_delay;
  std::multimap<SubsecondTime, SubsecondTime> m_window;
  UInt64 m_num_arrivals;
  UInt64 m_service_time_sum;
  UInt64 m_service_time_sum2;
  void addItem(SubsecondTime pkt_time, SubsecondTime service_time);
  void removeItems(SubsecondTime earliest_time);
};

}  // namespace Ramulator::Sniper
#endif
