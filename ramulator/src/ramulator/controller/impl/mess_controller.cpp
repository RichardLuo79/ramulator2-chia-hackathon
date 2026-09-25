#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>
#include "mess_mem_ctrl.h"

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/base/request.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/dram/device.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

// Model-zoo baseline: the Mess simulator (bsc-mem/Mess-simulator,
// Standalone/src/mess_mem_ctrl.cpp; a successor to the MICRO 2024 integrated
// artifact, not that exact publication revision). Memory latency is servoed
// onto a family of bandwidth-latency curves, one curve per read percentage.
// The pinned upstream MessMemCtrl owns all prediction and feedback state.
// This adapter owns only Ramulator admission, callbacks and measured counters.
// Curves here are fitted to OUR cycle-level oracle (not real hardware) via
// the lat-tp harness, making this a model-vs-model comparison.
// Both legacy text curves and upstream JSON are accepted by the loader.
class MessController final : public IController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IController, MessController, "Mess")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_curve_path, std::string, "curve_path").required();
    RAMULATOR_PARSE_PARAM(m_window_accesses, int, "window_accesses").default_val(1000);
    RAMULATOR_PARSE_PARAM(m_converge, double, "converge").default_val(0.05);
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "trace_path").default_val("");

    // RAMULATOR_CHILD: dram
    std::string dram_impl = m_config["dram"]["impl"].as<std::string>();
    m_device.init(DRAMSpec::create(dram_impl, m_config));
    m_tCK_ps = m_device.m_spec->get_timing_value("tCK_ps");
    if (m_window_accesses <= 0 || m_converge != 0.05) {
      throw std::runtime_error("Mess: window_accesses must be positive; upstream fixes converge=0.05");
    }
    if (get_tx_bytes() != 64 || m_tCK_ps <= 0) {
      throw std::runtime_error("Mess: upstream requires 64-byte transactions and a positive DRAM clock");
    }
    // Upstream calls this CPU frequency, but its interface only needs a clock
    // unit. Supply DRAM GHz so access() consumes and returns DRAM cycles.
    // Each Ramulator channel has its own predictor and single-channel curve.
    m_model = std::make_unique<MessMemCtrl>(m_curve_path, m_window_accesses,
                                          1000.0 / m_tCK_ps, 1);
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    if (!m_trace_path.empty()) {
      m_trace_file.open(fmt::format("{}.ch{}", m_trace_path, m_channel_id));
      m_trace_file << "arrive,depart,type,source,addr,cls,wbank,wact,wbus,clamped,frontend_id,frontend_sub_id,admission_ordinal\n";
    }
    m_stats.add("cycles", m_measured_clk);
    m_stats.add("num_read_reqs", s_num_read_reqs);
    m_stats.add("num_write_reqs", s_num_write_reqs);
    m_stats.add("num_read_reqs_served", s_num_read_reqs_served);
    m_stats.add("num_write_reqs_served", s_num_write_reqs_served);
    m_stats.add("read_latency", s_read_latency);
    m_stats.add("avg_read_latency", s_avg_read_latency);
  }

  void set_channel_id(int channel_id) override {
    IController::set_channel_id(channel_id);
    m_device.set_channel_id(channel_id);
  }
  int get_tx_bytes() const override {
    return m_device.m_spec->get_tx_bytes();
  }
  int get_num_levels() const override {
    return m_device.m_spec->level_count;
  }
  float get_tCK() const override {
    return m_tCK_ps / 1000.0f;
  }
  const DRAMSpec* get_spec() const override {
    return m_device.m_spec;
  }
  bool priority_send(Request& req) override {
    throw std::runtime_error("Mess: maintenance requests unsupported");
  }

  bool send(Request& req) override {
    const bool is_read = (req.type_id == Request::Type::Read);
    if (!is_read && req.type_id != Request::Type::Write) {
      throw std::runtime_error("Mess only supports Read/Write");
    }
    const Clk_t t = m_clk + 1;

    const Clk_t depart = t + m_model->access(t, !is_read);
    if (m_trace_file.is_open()) {
      m_trace_file << m_clk << ',' << depart << ',' << req.type_id << ','
                   << req.source_id << ',' << req.addr << ",0,0,0,0,0,"
                   << req.frontend_id << ',' << req.frontend_sub_id << ',' << req.admission_ordinal << '\n';
    }
    if (is_read) {
      req.arrive = m_clk;
      req.depart = depart;
      s_num_read_reqs++;
      m_pending.push(PendingRead{depart, req});
    } else {
      s_num_write_reqs++;
      s_num_write_reqs_served++;
      if (req.callback) {
        req.callback(req);
      }
    }
    return true;
  }

  void tick() override {
    m_clk++;
    m_measured_clk++;
    while (!m_pending.empty() && m_pending.top().depart <= m_clk) {
      Request req = m_pending.top().req;
      m_pending.pop();
      s_read_latency += req.depart - req.arrive;
      s_num_read_reqs_served++;
      if (req.callback) {
        req.callback(req);
      }
    }
  }

  Clk_t idle_ticks() override {
    if (m_pending.empty()) {
      return std::numeric_limits<Clk_t>::max();
    }
    Clk_t next = m_pending.top().depart;
    return (next > m_clk + 1) ? (next - m_clk - 1) : 0;
  }
  void fast_forward(Clk_t ticks) override {
    m_clk += ticks;
    m_measured_clk += ticks;
  }

  void update_stats() override {
    s_avg_read_latency = (s_num_read_reqs_served > 0) ? (float)s_read_latency / (float)s_num_read_reqs_served : 0;
  }
  void finalize() override {
    if (m_trace_file.is_open()) {
      m_trace_file.close();
    }
    update_stats();
  }
  void reset_stats() override {
    m_measured_clk = 0;
    s_num_read_reqs = s_num_write_reqs = 0;
    s_num_read_reqs_served = s_num_write_reqs_served = 0;
    s_read_latency = 0;
    s_avg_read_latency = 0;
  }

 private:
  struct PendingRead {
    Clk_t depart;
    Request req;
    bool operator>(const PendingRead& o) const {
      return depart > o.depart;
    }
  };

  DRAMDevice m_device;
  int m_tCK_ps = 0;
  std::string m_curve_path;
  int m_window_accesses = 1000;
  double m_converge = 0.05;

  std::unique_ptr<MessMemCtrl> m_model;
  Clk_t m_clk = 0;
  std::priority_queue<PendingRead, std::vector<PendingRead>, std::greater<PendingRead>> m_pending;

  std::string m_trace_path;
  std::ofstream m_trace_file;

  Clk_t m_measured_clk = 0;
  size_t s_num_read_reqs = 0, s_num_write_reqs = 0;
  size_t s_num_read_reqs_served = 0, s_num_write_reqs_served = 0;
  size_t s_read_latency = 0;
  float s_avg_read_latency = 0;
};

}  // namespace Ramulator
