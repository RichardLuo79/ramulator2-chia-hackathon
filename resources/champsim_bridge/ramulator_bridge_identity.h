#ifndef CHAMPSIM_RAMULATOR_BRIDGE_IDENTITY_H
#define CHAMPSIM_RAMULATOR_BRIDGE_IDENTITY_H

#include <cstdint>
#include <limits>

#include "access_type.h"

namespace ramulator_bridge
{

struct request_identity {
  int source_id = -1;
  std::int64_t frontend_id = -1;
  std::int64_t frontend_sub_id = 0;
};

/**
 * Derive an identity that is stable across Ramulator backpressure retries.
 *
 * ChampSim's dynamic instruction ID is trustworthy for demand reads, but it
 * is not unique by itself: one instruction can generate multiple memory
 * transactions.  The sub-ID therefore encodes the frontend virtual
 * transaction address and whether the demand was a load or an RFO. The
 * physical address is deliberately excluded from the identity: ChampSim's
 * first-touch mapper can assign a logical request differently in independently
 * paced runs, and the evaluator must expose that divergence as a separate
 * physical-consistency statistic. Cache writebacks inherit the ID of the
 * request that happened to evict the line, and prefetch/page-walk packets do
 * not have a demand instruction identity, so those remain explicitly
 * ineligible (frontend_id == -1). The trace matcher rejects duplicate stable
 * keys, keeping this bridge stateless and fail-closed rather than retaining
 * O(total requests) duplicate-detection state.
 */
inline request_identity make_request_identity(std::uint32_t cpu, std::uint64_t instr_id, std::uint64_t frontend_virtual_address, std::uint64_t physical_address,
                                              access_type type, int tx_bytes)
{
  // Physical equality is checked independently after logical pairing.
  static_cast<void>(physical_address);
  request_identity result;
  if (cpu <= static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    result.source_id = static_cast<int>(cpu);
  }

  const bool is_load = type == access_type::LOAD;
  const bool is_rfo = type == access_type::RFO;
  if (result.source_id < 0 || (!is_load && !is_rfo) || tx_bytes <= 0 || instr_id > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return result;
  }

  const auto transaction = frontend_virtual_address / static_cast<std::uint64_t>(tx_bytes);
  constexpr auto KIND_COUNT = std::uint64_t{2};
  if (transaction > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / KIND_COUNT) {
    return result;
  }

  result.frontend_id = static_cast<std::int64_t>(instr_id);
  result.frontend_sub_id = static_cast<std::int64_t>(transaction * KIND_COUNT + (is_rfo ? 1 : 0));
  return result;
}

} // namespace ramulator_bridge

#endif
