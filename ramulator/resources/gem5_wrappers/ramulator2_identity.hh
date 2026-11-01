#ifndef __MEM_RAMULATOR2_IDENTITY_HH__
#define __MEM_RAMULATOR2_IDENTITY_HH__

#include <cstdint>
#include <limits>

namespace gem5
{

namespace memory
{

struct Ramulator2RequestIdentity
{
    int sourceId = 0;
    std::int64_t frontendId = -1;
    std::int64_t frontendSubId = 0;
};

/**
 * Select a stable Ramulator source from gem5's execution context.
 *
 * O3 demand requests carry ContextID through translation and the classic
 * caches. Requests without a valid context (writebacks and prefetches in
 * particular) retain the bridge's historical source 0; those requests also
 * lack an instruction sequence and remain ineligible for stable matching.
 */
inline int
ramulator2SourceId(bool has_context_id, std::int64_t context_id)
{
    if (has_context_id && context_id >= 0 &&
        context_id <= std::numeric_limits<int>::max()) {
        return static_cast<int>(context_id);
    }
    return 0;
}

/**
 * Derive a stateless identity that is unchanged on gem5 timing retries.
 *
 * The dynamic O3 instruction sequence identifies a demand instruction (or the
 * demand lineage propagated into a hardware prefetch). Physical transaction
 * address, Ramulator request kind, and gem5 requestor distinguish split and
 * multi-level-prefetch descendants. A missing or overflowing sequence makes
 * the request explicitly ineligible (frontendId == -1). Dynamic sequence
 * numbers are stable for retries of one Packet, but they include squashed
 * instructions and are therefore only a provisional closed-loop identity
 * across independently paced simulations; the evaluator must retain coverage
 * and address checks.
 */
inline Ramulator2RequestIdentity
ramulator2RequestIdentity(int source_id, bool has_inst_seq,
                          std::uint64_t inst_seq,
                          std::uint64_t physical_address,
                          bool is_write, std::uint64_t requestor_id,
                          int tx_bytes)
{
    Ramulator2RequestIdentity result;
    result.sourceId = source_id;

    constexpr auto MaxId =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    constexpr auto MaxRequestor =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint16_t>::max());
    if (source_id < 0 || !has_inst_seq || inst_seq > MaxId ||
        requestor_id > MaxRequestor || tx_bytes <= 0) {
        return result;
    }

    const auto transaction =
        physical_address / static_cast<std::uint64_t>(tx_bytes);
    const auto kind = is_write ? std::uint64_t{1} : std::uint64_t{0};
    constexpr auto KindCount = std::uint64_t{2};
    constexpr auto RequestorCount = MaxRequestor + 1;
    constexpr auto DiscriminatorCount = RequestorCount * KindCount;
    const auto discriminator = requestor_id * KindCount + kind;
    if (transaction > (MaxId - discriminator) / DiscriminatorCount) {
        return result;
    }

    result.frontendId = static_cast<std::int64_t>(inst_seq);
    result.frontendSubId = static_cast<std::int64_t>(
        transaction * DiscriminatorCount + discriminator);
    return result;
}

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_IDENTITY_HH__
