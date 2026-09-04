#include <algorithm>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>

#include "ramulator/base/param.h"
#include "ramulator/base/utils.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/frontend/impl/processor/simpleO3/core.h"
#include "ramulator/frontend/impl/processor/simpleO3/llc.h"
#include "ramulator/translation/i_translation.h"

namespace Ramulator {

// Trace format (one file per core, passed via "traces" param):
// One instruction per line, space-separated.
//   <bubble_count> <load_addr> [store_addr]
//
// - bubble_count: number of non-memory instructions before this memory access
// - load_addr:    load address (decimal or 0x hex)
// - store_addr:   optional store/writeback address (decimal or 0x hex)
//
// Example:
//   3 20734016
//   8 20841280 20841280
//
// The trace replays cyclically.
class SimpleO3 final : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, SimpleO3, "SimpleO3")

 private:
  ITranslation* m_translation;

  int m_num_cores = -1;
  std::vector<std::unique_ptr<SimpleO3Core>> m_cores;
  std::unique_ptr<SimpleO3LLC> m_llc;

  int m_num_expected_insts;
  std::vector<std::string> m_traces;
  int m_ipc;
  int m_depth;
  int m_llc_latency;
  int m_llc_linesize_bytes;
  int m_llc_associativity;
  int m_llc_num_mshr_per_core;
  std::string m_llc_capacity_str;
  std::string m_crit_trace_path;
  std::string m_request_trace_path;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_num_expected_insts, int, "num_expected_insts").required();
    RAMULATOR_PARSE_PARAM(m_traces, std::vector<std::string>, "traces").required();
    RAMULATOR_PARSE_PARAM(m_ipc, int, "ipc").default_val(4);
    RAMULATOR_PARSE_PARAM(m_depth, int, "inst_window_depth").default_val(128);
    RAMULATOR_PARSE_PARAM(m_llc_latency, int, "llc_latency").default_val(47);
    RAMULATOR_PARSE_PARAM(m_llc_linesize_bytes, int, "llc_linesize").default_val(64);
    RAMULATOR_PARSE_PARAM(m_llc_associativity, int, "llc_associativity").default_val(8);
    RAMULATOR_PARSE_PARAM(m_llc_capacity_str, std::string, "llc_capacity_per_core").default_val("2MB");
    RAMULATOR_PARSE_PARAM(m_llc_num_mshr_per_core, int, "llc_num_mshr_per_core").default_val(16);
    RAMULATOR_PARSE_PARAM(m_crit_trace_path, std::string, "crit_trace_path").default_val("");
    RAMULATOR_PARSE_PARAM(m_request_trace_path, std::string, "request_trace_path").default_val("");

    if (m_num_expected_insts <= 0) {
      throw std::runtime_error("SimpleO3 num_expected_insts must be positive");
    }
    if (m_traces.empty()) {
      throw std::runtime_error("SimpleO3 requires at least one core trace");
    }
    if (m_traces.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("SimpleO3 core count exceeds the supported range");
    }
    if (m_ipc <= 0 || m_depth <= 0) {
      throw std::runtime_error("SimpleO3 IPC and instruction-window depth must be positive");
    }
    if (m_llc_num_mshr_per_core <= 0) {
      throw std::runtime_error("SimpleO3 LLC MSHRs per core must be positive");
    }

    m_num_cores = static_cast<int>(m_traces.size());
    const size_t llc_capacity_per_core = parse_capacity_str(m_llc_capacity_str);
    if (llc_capacity_per_core == 0 ||
        llc_capacity_per_core > static_cast<size_t>(std::numeric_limits<int>::max()) / m_num_cores) {
      throw std::runtime_error("SimpleO3 aggregate LLC capacity exceeds the supported range");
    }
    if (m_llc_num_mshr_per_core > std::numeric_limits<int>::max() / m_num_cores) {
      throw std::runtime_error("SimpleO3 aggregate LLC MSHR count exceeds the supported range");
    }

    RAMULATOR_CREATE_CHILD(m_translation, ITranslation);

    m_llc = std::make_unique<SimpleO3LLC>(
        m_clk, m_llc_latency, static_cast<int>(llc_capacity_per_core * m_num_cores), m_llc_linesize_bytes,
        m_llc_associativity, m_llc_num_mshr_per_core * m_num_cores, m_request_trace_path);
    m_llc->m_hit_completion_callback = [this](const SimpleO3LLC::LogicalRequest& logical) {
      this->receive_hit(logical);
    };

    for (int id = 0; id < m_num_cores; id++) {
      auto core = std::make_unique<SimpleO3Core>(m_clk, id, m_ipc, m_depth, m_num_expected_insts, m_traces[id],
                                                 m_translation, m_llc.get());
      core->m_callback = [this](Request& req) { return this->receive(req); };
      if (!m_crit_trace_path.empty()) {
        core->open_crit_trace(fmt::format("{}.core{}", m_crit_trace_path, id));
      }
      m_cores.push_back(std::move(core));
    }

    m_stats.add("num_expected_insts", m_num_expected_insts);
    m_stats.add("llc_eviction", m_llc->s_llc_eviction);
    m_stats.add("llc_read_access", m_llc->s_llc_read_access);
    m_stats.add("llc_write_access", m_llc->s_llc_write_access);
    m_stats.add("llc_read_misses", m_llc->s_llc_read_misses);
    m_stats.add("llc_write_misses", m_llc->s_llc_write_misses);
    m_stats.add("llc_mshr_unavailable", m_llc->s_llc_mshr_unavailable);
    m_stats.add("logical_requests_completed", m_llc->s_logical_requests_completed);
    m_stats.add("logical_requests_live", m_llc->s_logical_requests_live);
    m_stats.add("logical_requests_peak", m_llc->s_logical_requests_peak);
    m_stats.add("logical_requests_hit", m_llc->s_logical_requests_hit);
    m_stats.add("logical_requests_mshr_merge", m_llc->s_logical_requests_mshr_merge);
    m_stats.add("logical_requests_miss_owner", m_llc->s_logical_requests_miss_owner);
    m_stats.add("internal_writebacks_generated", m_llc->s_internal_writebacks_generated);
    m_stats.add("internal_writebacks_completed", m_llc->s_internal_writebacks_completed);
    m_stats.add("internal_writebacks_live", m_llc->s_internal_writebacks_live);

    for (int core_id = 0; core_id < m_cores.size(); core_id++) {
      m_stats.add(fmt::format("insts_issued_core_{}", core_id), m_cores[core_id]->s_insts_issued);
      m_stats.add(fmt::format("cycles_recorded_core_{}", core_id), m_cores[core_id]->s_cycles_recorded);
      m_stats.add(fmt::format("memory_access_cycles_recorded_core_{}", core_id), m_cores[core_id]->s_mem_access_cycles);
    }
  }

  void tick() override {
    m_clk++;

    constexpr Clk_t kHeartbeatInterval = 10'000'000;
    if (m_clk % kHeartbeatInterval == 0) {
      m_logger.info(fmt::format("Processor Heartbeat {} cycles.", m_clk));
    }

    m_llc->tick();
    for (auto& core : m_cores) {
      core->tick();
    }
  }

  void receive(Request& req) {
    auto logical_requests = m_llc->take_receive_requests(req.addr);
    if (std::any_of(logical_requests.begin(), logical_requests.end(),
                    [](const auto& logical) { return logical.path == SimpleO3LLC::LogicalPath::Hit; })) {
      throw std::runtime_error("SimpleO3 memory completion contains an LLC-hit request");
    }
    m_llc->receive(req);

    // Completed LLC requests are returned to their source cores.
    for (const auto& logical : logical_requests) {
      m_llc->complete_logical_request(logical, m_clk);
      Request r = logical.req;
      r.type_id = logical.original_type;
      r.arrive = req.arrive;
      r.depart = req.depart;
      if (r.source_id < 0 || r.source_id >= m_num_cores) {
        throw std::runtime_error("SimpleO3 logical completion has an invalid source id");
      }
      m_cores[r.source_id]->receive(r);
    }
  };

  void receive_hit(const SimpleO3LLC::LogicalRequest& logical) {
    if (logical.path != SimpleO3LLC::LogicalPath::Hit) {
      throw std::runtime_error("SimpleO3 hit completion has a non-hit logical path");
    }
    m_llc->complete_logical_request(logical, m_clk);
    Request r = logical.req;
    r.type_id = logical.original_type;
    if (r.source_id < 0 || r.source_id >= m_num_cores) {
      throw std::runtime_error("SimpleO3 logical hit completion has an invalid source id");
    }
    m_cores[r.source_id]->receive(r);
  }

  // Tick elision: upcoming frontend ticks that are provably no-ops. Only
  // valid when every core is stalled (full window, un-ready tail) and the
  // LLC has no due or pending work; the next event is then the earliest
  // LLC latency-queue entry. Memory-side deliveries are bounded separately
  // by the memory system's own idle_ticks in the simulation loop.
  Clk_t idle_ticks(Clk_t max_useful) override {
    for (auto& core : m_cores) {
      if (!core->is_stalled()) {
        return 0;
      }
    }
    Clk_t next = m_llc->next_event();
    if (next <= m_clk + 1) {
      return 0;
    }
    return std::min(max_useful, next - m_clk - 1);
  }

  void fast_forward(Clk_t ticks) override {
    // Contract: ticks <= idle_ticks(), so every skipped tick would only
    // have counted blocked cycles. Heartbeat prints for skipped spans are
    // dropped (cosmetic only).
    m_clk += ticks;
    for (auto& core : m_cores) {
      core->fast_forward(ticks);
    }
  }

  bool is_finished() override {
    for (auto& core : m_cores) {
      if (!(core->reached_expected_num_insts)) {
        return false;
      }
      if (!core->issue_roi_quiescent()) {
        return false;
      }
    }
    // Stop issuing at the fixed per-core ROI, but keep ticking the LLC and
    // memory system until every accepted logical request has completed.
    return m_llc->is_quiescent();
  }

  void finalize() override {
    m_llc->finalize();
  }

  void connect_memory_system(IMemorySystem* memory_system) override {
    m_llc->connect_memory_system(memory_system);
  };

  int get_num_cores() override {
    return m_num_cores;
  };
};

}  // namespace Ramulator
