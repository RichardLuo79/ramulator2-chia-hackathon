#pragma once

// The only Ramulator header supplied to new CHIA model implementations.
// This is a C++20 interface, qualified with the campaign's recorded toolchain;
// it is not a compiler-independent binary ABI or a memory-safety sandbox.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Ramulator::AtomicModel {

using Cycle = std::int64_t;
inline constexpr std::uint32_t api_version = 1;

enum class AccessType { Read, Write };

// A value copy of the currently admitted transaction, never the frontend's
// request or callback object. No observation IDs, future traffic or queue
// storage are reachable through this interface.
struct Access {
  std::int64_t address;
  AccessType type;
  std::vector<int> levels;
  int size_bytes;
};

// Configuration values only. No DRAM device, command machinery, frontend,
// controller pointer, workload label or live statistics object is supplied.
struct Hardware {
  std::string standard;
  std::vector<std::string> level_names;
  std::vector<int> level_sizes;
  std::map<std::string, Cycle> timings;
  Cycle read_latency;
  int tck_ps;
  int transaction_bytes;
  int channel_width;
  int device_width;
  int internal_prefetch_size;
  int read_capacity;
  int write_capacity;
  double write_low_watermark;
  double write_high_watermark;

  std::size_t level_index(const std::string& name) const {
    const auto it = std::find(level_names.begin(), level_names.end(), name);
    if (it == level_names.end()) throw std::invalid_argument("Unknown DRAM level: " + name);
    return static_cast<std::size_t>(it - level_names.begin());
  }

  Cycle timing(const std::string& name) const { return timings.at(name); }
  int level_size(const std::string& name) const { return level_sizes.at(level_index(name)); }
};

class Parameters {
 public:
  explicit Parameters(std::map<std::string, double> overrides) : m_overrides(std::move(overrides)) {}

  // The implementation owns names, defaults and ranges. This is called during
  // initialize(); no harness-selected fitting constants are injected.
  double number(const std::string& name, double fallback, double minimum, double maximum) {
    if (name.empty() || (name.front() >= '0' && name.front() <= '9') || name.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos ||
        !std::isfinite(fallback) || !std::isfinite(minimum) || !std::isfinite(maximum) ||
        minimum > maximum || fallback < minimum || fallback > maximum) {
      throw std::invalid_argument("Invalid atomic model parameter declaration: " + name);
    }
    const auto it = m_overrides.find(name);
    const double value = it == m_overrides.end() ? fallback : it->second;
    if (!std::isfinite(value) || value < minimum || value > maximum) {
      throw std::invalid_argument("Atomic model parameter outside its declared range: " + name);
    }
    m_used.insert(name);
    return value;
  }

  void require_all_used() const {
    for (const auto& [name, value] : m_overrides) {
      if (!m_used.contains(name)) throw std::invalid_argument("Unknown atomic model parameter: " + name);
    }
  }

 private:
  std::map<std::string, double> m_overrides;
  std::set<std::string> m_used;
};

class Model {
 public:
  virtual ~Model() = default;
  virtual void initialize(Hardware hardware, Parameters& parameters) = 0;

  // Called exactly once per admitted request, never on a rejected admission,
  // simulator tick, completion or statistics reset. Return an absolute future
  // DRAM cycle. The trusted controller retains that value and serves callbacks.
  virtual Cycle predict(Access access, Cycle now) = 0;
};

using VersionFunction = std::uint32_t (*)();
using CreateFunction = Model* (*)();

}  // namespace Ramulator::AtomicModel

// Each candidate defines these two symbols. It is loaded by trusted code only
// after input staging and the native execution boundary have been established.
extern "C" std::uint32_t ramulator_atomic_model_api_version();
extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model();
