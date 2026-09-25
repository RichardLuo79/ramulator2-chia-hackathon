#ifndef RAMULATOR_FRONTEND_LAT_TP_ADDRESSES_H
#define RAMULATOR_FRONTEND_LAT_TP_ADDRESSES_H

// Shared, stateless Lat-Tp coordinate recipes. Preserve draw order (including
// size-one fields) and stagger policy when used by the legacy frontend.
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include "ramulator/base/type.h"

namespace Ramulator::LatTp {
inline AddrVec_t streaming(size_t idx, int size, const std::vector<int>& positions,
    const std::vector<int>& counts, int units, int row_pos, int col_pos,
    int rows, int prefetch, int stream_cls, bool stagger) {
  AddrVec_t av(size, 0);
  int flat = static_cast<int>(idx % units);
  size_t local = idx / units;
  if (stagger && stream_cls > 0) {
    const size_t banks_per_phase = (static_cast<size_t>(units) + stream_cls - 1) / stream_cls;
    local += units <= stream_cls ? static_cast<size_t>(flat) * stream_cls / units
                                : (static_cast<size_t>(flat) / banks_per_phase) % stream_cls;
  }
  for (int i = static_cast<int>(positions.size()) - 1; i >= 0; --i) {
    av[positions[i]] = flat % counts[i];
    flat /= counts[i];
  }
  av[row_pos] = static_cast<int>((local / stream_cls) % rows);
  av[col_pos] = static_cast<int>(local % stream_cls) * prefetch;
  return av;
}

inline AddrVec_t random(std::mt19937_64& rng, int size, const std::vector<int>& positions,
    const std::vector<int>& counts, int row_pos, int col_pos, int rows, int cls, int prefetch) {
  AddrVec_t av(size, 0);
  for (size_t i = 0; i < positions.size(); ++i) {
    std::uniform_int_distribution<int> dist(0, counts[i] - 1);
    av[positions[i]] = dist(rng);
  }
  std::uniform_int_distribution<int> row_dist(0, rows - 1), cls_dist(0, cls - 1);
  av[row_pos] = row_dist(rng);
  av[col_pos] = cls_dist(rng) * prefetch;
  return av;
}

inline Addr_t physical_byte_address(const AddrVec_t& av, const std::vector<int>& levels,
    int row_pos, int col_pos, int rows, int cls, int prefetch, int tx_bytes) {
  uint64_t line = av[row_pos];
  auto append = [&line](uint64_t count, uint64_t value) {
    if (!count || value >= count ||
        line > (static_cast<uint64_t>(std::numeric_limits<Addr_t>::max()) - value) / count)
      throw std::runtime_error("Lat-Tp: physical address exceeds its geometry");
    line = line * count + value;
  };
  if (av[0] != 0 || av[row_pos] < 0 || av[row_pos] >= rows ||
      av[col_pos] < 0 || av[col_pos] % prefetch != 0)
    throw std::runtime_error("Lat-Tp: invalid single-channel physical address");
  for (int position = row_pos - 1; position >= 1; --position) append(levels[position], av[position]);
  append(cls, av[col_pos] / prefetch);
  if (tx_bytes <= 0 || (tx_bytes & (tx_bytes - 1)))
    throw std::runtime_error("Lat-Tp: unsupported transaction size");
  append(tx_bytes, 0);
  return static_cast<Addr_t>(line);
}
}  // namespace Ramulator::LatTp
#endif
