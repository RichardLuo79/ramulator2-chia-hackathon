#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "ramulator/base/factory.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"
#include "ramulator/python/binding_utils.h"

// ---- Simulation wrapper ----

class Simulation {
  std::unique_ptr<IFrontEnd> m_frontend;
  std::unique_ptr<IMemorySystem> m_memory_system;
  bool m_finalized = false;

  void update_stats() {
    m_frontend->update_stats_recursive();
    m_memory_system->update_stats_recursive();
  }

 public:
  explicit Simulation(nb::dict config) {
    ConfigNode cfg = py_to_confignode(config);

    m_frontend.reset(Factory::create_frontend(cfg));
    m_memory_system.reset(Factory::create_memory_system(cfg));

    m_frontend->connect_memory_system(m_memory_system.get());
    m_memory_system->connect_frontend(m_frontend.get());
  }

  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;

  ~Simulation() noexcept {
    try {
      finalize();
    } catch (...) {
    }
  }

  void run() {
    const int fe_tick = m_frontend->get_clock_ratio();
    const int mem_tick = m_memory_system->get_clock_ratio();
    if (fe_tick <= 0 || mem_tick <= 0) {
      throw std::runtime_error("clock_ratio must be > 0 for both frontend and memory system");
    }

    constexpr Clk_t MAX_JUMP = 1'000'000'000;
    int fe_count = mem_tick - 1, mem_count = fe_tick - 1;
    for (;;) {
      if (++fe_count >= mem_tick) {
        fe_count = 0;
        m_frontend->tick();
      }

      if (m_frontend->is_finished()) {
        break;
      }

      if (++mem_count >= fe_tick) {
        mem_count = 0;
        m_memory_system->tick();
      }

      // Tick elision: when both sides are provably idle, jump the interleave
      // through the idle stretch and bulk-advance their clocks. Counter
      // algebra: from post-iteration counter c, the k-th upcoming tick of a
      // side with period P happens after (P - c) + (k-1)*P more iterations.
      Clk_t mem_idle = m_memory_system->idle_ticks();
      if (mem_idle > 0 && !m_frontend->is_finished()) {
        mem_idle = std::min(mem_idle, MAX_JUMP);
        const uint64_t j_mem =
            static_cast<uint64_t>(fe_tick - mem_count) + static_cast<uint64_t>(mem_idle) * fe_tick - 1;
        // Frontend ticks usable within the memory-allowed window:
        const uint64_t max_fe = (static_cast<uint64_t>(fe_count) + j_mem) / mem_tick;
        if (max_fe > 0) {
          Clk_t fe_idle = m_frontend->idle_ticks(static_cast<Clk_t>(std::min<uint64_t>(max_fe, MAX_JUMP)));
          if (fe_idle > 0) {
            const uint64_t j_fe =
                static_cast<uint64_t>(mem_tick - fe_count) + static_cast<uint64_t>(fe_idle) * mem_tick - 1;
            const uint64_t j = std::min(j_fe, j_mem);
            if (j >= 2) {
              const uint64_t fe_ff = (static_cast<uint64_t>(fe_count) + j) / mem_tick;
              const uint64_t mem_ff = (static_cast<uint64_t>(mem_count) + j) / fe_tick;
              fe_count = static_cast<int>((static_cast<uint64_t>(fe_count) + j) % mem_tick);
              mem_count = static_cast<int>((static_cast<uint64_t>(mem_count) + j) % fe_tick);
              if (fe_ff > 0) {
                m_frontend->fast_forward(static_cast<Clk_t>(fe_ff));
              }
              if (mem_ff > 0) {
                m_memory_system->fast_forward(static_cast<Clk_t>(mem_ff));
              }
            }
          }
        }
      }
    }
  }

  void finalize() {
    if (m_finalized) {
      return;
    }
    m_frontend->finalize();
    m_memory_system->finalize();
    m_finalized = true;
  }

  nb::dict get_stats() {
    update_stats();
    ConfigNode::Map root;
    root["frontend"] = m_frontend->collect_stats();
    root["memory_system"] = m_memory_system->collect_stats();

    return nb::cast<nb::dict>(confignode_to_py(ConfigNode(std::move(root))));
  }

  std::string get_stats_yaml() {
    update_stats();
    std::ostringstream ss;
    m_frontend->print_stats(ss);
    m_memory_system->print_stats(ss);
    return ss.str();
  }
};

// ---- nanobind module ----

// Batch (array-in/array-out) driver: serves a whole request trace through the
// memory system with no frontend ticking and no Python in the loop. Requests
// arrive at the given controller-clock ticks (non-decreasing); backpressure
// retries tick-by-tick; idle stretches are jumped via idle_ticks().
class BatchSim {
 public:
  explicit BatchSim(const nb::dict& config_dict) {
    ConfigNode cfg = py_to_confignode(config_dict);
    m_frontend.reset(Factory::create_frontend(cfg));
    m_memory_system.reset(Factory::create_memory_system(cfg));
    m_frontend->connect_memory_system(m_memory_system.get());
    m_memory_system->connect_frontend(m_frontend.get());
  }

  ~BatchSim() noexcept {
    try {
      finalize();
    } catch (...) {
    }
  }

  BatchSim(const BatchSim&) = delete;
  BatchSim& operator=(const BatchSim&) = delete;

  std::vector<int64_t> run(const std::vector<int64_t>& addrs,
                           const std::vector<int>& types,
                           const std::vector<int64_t>& arrives) {
    if (addrs.size() != types.size() || addrs.size() != arrives.size()) {
      throw std::runtime_error("BatchSim.run: addrs/types/arrives must have equal length");
    }
    const size_t n = addrs.size();
    std::vector<int64_t> departs(n, -1);
    size_t completed_reads = 0;
    size_t expected_reads = 0;
    const int tx_bytes = m_memory_system->get_tx_bytes();

    Clk_t clk = 0;
    auto advance = [&](Clk_t ticks) {
      while (ticks > 0) {
        Clk_t idle = m_memory_system->idle_ticks();
        Clk_t jump = std::min(idle, ticks - 1);
        if (jump > 0) {
          m_memory_system->fast_forward(jump);
          clk += jump;
          ticks -= jump;
        } else {
          m_memory_system->tick();
          clk++;
          ticks--;
        }
      }
    };

    for (size_t i = 0; i < n; i++) {
      if (arrives[i] > clk) {
        advance(arrives[i] - clk);
      }
      Request req(static_cast<Addr_t>(addrs[i]), types[i]);
      req.frontend_id = m_next_frontend_id++;
      req.size_bytes = tx_bytes;
      if (types[i] == Request::Type::Read) {
        expected_reads++;
        req.callback = [&departs, &completed_reads, i](Request& r) {
          departs[i] = r.depart;
          completed_reads++;
        };
      }
      while (!m_memory_system->send(req)) {
        advance(1);
      }
    }

    while (completed_reads < expected_reads) {
      Clk_t idle = m_memory_system->idle_ticks();
      if (idle == std::numeric_limits<Clk_t>::max()) {
        throw std::runtime_error("BatchSim.run: outstanding reads but memory system reports no events");
      }
      if (idle > 0) {
        m_memory_system->fast_forward(idle);
        clk += idle;
      }
      m_memory_system->tick();
      clk++;
    }
    return departs;
  }

  void finalize() {
    if (m_finalized) {
      return;
    }
    m_frontend->finalize();
    m_memory_system->finalize();
    m_finalized = true;
  }

  nb::dict get_stats() {
    m_frontend->update_stats_recursive();
    m_memory_system->update_stats_recursive();
    nb::dict stats;
    stats["frontend"] = confignode_to_py(m_frontend->collect_stats());
    stats["memory_system"] = confignode_to_py(m_memory_system->collect_stats());
    return stats;
  }

 private:
  std::unique_ptr<IFrontEnd> m_frontend;
  std::unique_ptr<IMemorySystem> m_memory_system;
  std::int64_t m_next_frontend_id = 0;
  bool m_finalized = false;
};


NB_MODULE(_ramulator, m) {
  m.doc() = "Ramulator2 Python bindings";

  nb::class_<BatchSim>(m, "BatchSim")
      .def(nb::init<nb::dict>(), nb::arg("config"), "Create a batch simulation from a configuration dict.")
      .def("run", &BatchSim::run, nb::arg("addrs"), nb::arg("types"), nb::arg("arrives"),
           "Serve a request array; returns per-request depart ticks (-1 for writes).")
      .def("finalize", &BatchSim::finalize, "Finalize the simulation and flush final outputs.")
      .def("get_stats", &BatchSim::get_stats, "Update derived stats and return them as a dict.");

  nb::class_<Simulation>(m, "Simulation")
      .def(nb::init<nb::dict>(), nb::arg("config"), "Create a simulation from a configuration dict.")
      .def("run", &Simulation::run, "Run the simulation to completion.")
      .def("finalize", &Simulation::finalize, "Finalize the simulation and flush final outputs.")
      .def("get_stats", &Simulation::get_stats, "Update derived stats and return them as a dict.")
      .def("get_stats_yaml", &Simulation::get_stats_yaml, "Update derived stats and return them as a YAML string.");
}
