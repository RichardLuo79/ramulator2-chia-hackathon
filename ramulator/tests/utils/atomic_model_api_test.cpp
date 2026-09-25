// The model-visible API is independently compilable, without frontend headers.
#include "ramulator/controller/atomic_model/api.h"

#include <iostream>
#include <limits>

using namespace Ramulator::AtomicModel;

template <typename T> concept HasCallback = requires(T value) { value.callback; };
template <typename T> concept HasObservation = requires(T value) { value.frontend_id; };
template <typename T> concept HasCommand = requires(T value) { value.command; };
template <typename T> concept HasDevice = requires(T value) { value.device; };
static_assert(!HasCallback<Access> && !HasObservation<Access> && !HasCommand<Access>);
static_assert(!HasDevice<Hardware>);

static void check(bool condition) {
  if (!condition) throw std::runtime_error("atomic model API check failed");
}

template <typename F> static void rejects(F test) {
  bool rejected = false;
  try { test(); } catch (const std::exception&) { rejected = true; }
  check(rejected);
}

int main() {
  Hardware hardware{};
  hardware.level_names = {"channel", "bank", "row"};
  hardware.level_sizes = {1, 32, 65536};
  hardware.timings = {{"nCL", 40}, {"nRCD", 40}};
  check(hardware.level_index("bank") == 1 && hardware.level_size("bank") == 32);
  check(hardware.timing("nCL") == 40);
  rejects([&] { hardware.level_index("future_request"); });
  rejects([&] { hardware.timing("unknown_timing"); });

  Parameters parameters({{"delay", 12}});
  check(parameters.number("delay", 10, 1, 20) == 12);
  check(parameters.number("default_only", 1.5, 0, 3) == 1.5);
  parameters.require_all_used();
  Parameters undeclared({{"delay", 12}});
  rejects([&] { undeclared.require_all_used(); });
  rejects([&] { parameters.number("delay", 1, 0, 2); });
  rejects([&] { parameters.number("invalid", 0, 2, 1); });
  rejects([&] { parameters.number("invalid", 10, 1, 2); });
  for (const std::string name : {"", "1starts_with_digit", "invalid-name", "../outside"}) {
    rejects([&] { parameters.number(name, 1, 0, 2); });
  }
  const double infinity = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { parameters.number("nan", nan, 0, 2); });
  rejects([&] { parameters.number("infinite", 1, 0, infinity); });
  Parameters nonfinite({{"delay", infinity}});
  rejects([&] { nonfinite.number("delay", 1, 0, 2); });

  // The old wrapper imposed arbitrary 128-parameter/64-character limits.
  // The value API has no such modeling restrictions.
  Parameters many({});
  for (int i = 0; i < 150; ++i) {
    check(many.number("parameter_" + std::to_string(i), i, 0, 200) == i);
  }
  check(many.number(std::string(150, 'p'), 1, 0, 2) == 1);
  many.require_all_used();
  std::cout << "atomic model value API and parameter checks passed\n";
}
