// Deterministic parity test against a one-interleave-step reference runner.
// This deliberately does not copy the production tick-elision algebra.
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ramulator/base/simulation.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

using namespace Ramulator;
using Events = std::vector<std::pair<char, Clk_t>>;

class TestFrontend final : public IFrontEnd {
 public:
  Events& events;
  Clk_t period, limit;
  Clk_t elided = 0;

  TestFrontend(Events& events, int ratio, Clk_t period, Clk_t limit)
      : events(events), period(period), limit(limit) {
    m_clock_ratio = ratio;
  }
  void tick() override {
    ++m_clk;
    if (m_clk % period == 0) events.emplace_back('f', m_clk);
  }
  bool is_finished() override { return m_clk >= limit; }
  Clk_t idle_ticks(Clk_t max_useful) override {
    return std::min({max_useful, period - m_clk % period - 1, limit - m_clk - 1});
  }
  void fast_forward(Clk_t ticks) override { m_clk += ticks; elided += ticks; }
  Clk_t clock() const { return m_clk; }
};

class TestMemory final : public IMemorySystem {
 public:
  Events& events;
  int ratio;
  Clk_t period, clock = 0, elided = 0;

  TestMemory(Events& events, int ratio, Clk_t period) : events(events), ratio(ratio), period(period) {}
  bool send(Request&) override { return false; }
  void tick() override {
    ++clock;
    if (clock % period == 0) events.emplace_back('m', clock);
  }
  Clk_t idle_ticks() override { return period - clock % period - 1; }
  void fast_forward(Clk_t ticks) override { clock += ticks; elided += ticks; }
  int get_clock_ratio() override { return ratio; }
  int get_tx_bytes() override { return 64; }
};

static void reference_run(TestFrontend& frontend, TestMemory& memory) {
  int frontend_count = memory.ratio - 1;
  int memory_count = frontend.get_clock_ratio() - 1;
  for (;;) {
    if (++frontend_count == memory.ratio) {
      frontend_count = 0;
      frontend.tick();
    }
    if (frontend.is_finished()) return;
    if (++memory_count == frontend.get_clock_ratio()) {
      memory_count = 0;
      memory.tick();
    }
  }
}

int main() {
  std::mt19937 random(1701);
  Clk_t total_elided = 0;
  for (int test = 0; test < 1500; ++test) {
    int fe_ratio = 1 + random() % 17, mem_ratio = 1 + random() % 17;
    Clk_t fe_period = 1 + random() % 1000, mem_period = 1 + random() % 1000;
    Clk_t limit = 1 + random() % 10000;
    Events expected, actual;
    TestFrontend reference_frontend(expected, fe_ratio, fe_period, limit);
    TestMemory reference_memory(expected, mem_ratio, mem_period);
    TestFrontend frontend(actual, fe_ratio, fe_period, limit);
    TestMemory memory(actual, mem_ratio, mem_period);
    reference_run(reference_frontend, reference_memory);
    run_simulation(frontend, memory);
    if (actual != expected || frontend.clock() != reference_frontend.clock() ||
        memory.clock != reference_memory.clock) {
      throw std::runtime_error("shared lifecycle differs from one-step reference");
    }
    total_elided += frontend.elided + memory.elided;
  }
  for (int invalid_ratio : {0, -1}) {
    Events events;
    TestFrontend frontend(events, 1, 10, 20);
    TestMemory memory(events, invalid_ratio, 10);
    bool rejected = false;
    try { run_simulation(frontend, memory); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid memory clock ratio was accepted");
  }
  {
    Events events;
    TestFrontend frontend(events, 0, 10, 20);
    TestMemory memory(events, 1, 10);
    bool rejected = false;
    try { run_simulation(frontend, memory); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid frontend clock ratio was accepted");
  }
  if (total_elided == 0) throw std::runtime_error("fixture did not exercise tick elision");
  std::cout << "1500 lifecycle parity cases passed; elided ticks: " << total_elided << '\n';
}
