#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "ramulator/base/factory.h"
#include "ramulator/base/batch_simulation.h"
#include "ramulator/base/simulation.h"
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
    run_simulation(*m_frontend, *m_memory_system);
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
    std::vector<BatchRequest> input;
    input.reserve(addrs.size());
    for (size_t i = 0; i < addrs.size(); ++i) {
      input.push_back({static_cast<Addr_t>(addrs[i]), types[i], arrives[i]});
    }
    return run_batch(*m_memory_system, input, m_batch_state).departed;
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
  BatchState m_batch_state;
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
