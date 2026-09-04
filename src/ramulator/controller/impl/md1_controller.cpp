#include <algorithm>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/base/request.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/controller/impl/zoo_probe.h"
#include "ramulator/dram/device.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

// Model-zoo baseline: zsim's MD1Memory (src/mem_ctrls.{h,cpp}) — the whole
// memory controller as one aggregate M/D/1 server. Request rate is smoothed
// per fixed phase; utilization rho = smoothed rate x deterministic burst
// service, clamped at 0.95 (zsim profiles these as "clamped loads"); every
// request is charged latency = zero-load latency + M/D/1 wait
// rho*S / (2*(1-rho)). Pure formula latency: no banks, no rows, no physical
// serialization — contention lives entirely inside the formula, as in zsim.
class MD1Controller final : public IController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IController, MD1Controller, "MD1")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_phase_ticks, int, "phase_ticks").default_val(10000);
    RAMULATOR_PARSE_PARAM(m_smoothing, float, "smoothing").default_val(0.5f);
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "trace_path").default_val("");

    // RAMULATOR_CHILD: dram
    std::string dram_impl = m_config["dram"]["impl"].as<std::string>();
    m_device.init(DRAMSpec::create(dram_impl, m_config));
    m_tCK_ps = m_device.m_spec->get_timing_value("tCK_ps");
    m_rl = m_device.m_spec->read_latency;
    m_burst = zoo_probe_burst_gap(m_device);
    m_phase_end = m_phase_ticks;
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    if (!m_trace_path.empty()) {
      m_trace_file.open(fmt::format("{}.ch{}", m_trace_path, m_channel_id));
      m_trace_file << "arrive,depart,type,source,addr,cls,wbank,wact,wbus,clamped\n";
    }
    m_stats.add("cycles", m_measured_clk);
    m_stats.add("num_read_reqs", s_num_read_reqs);
    m_stats.add("num_write_reqs", s_num_write_reqs);
    m_stats.add("num_read_reqs_served", s_num_read_reqs_served);
    m_stats.add("num_write_reqs_served", s_num_write_reqs_served);
    m_stats.add("read_latency", s_read_latency);
    m_stats.add("avg_read_latency", s_avg_read_latency);
    m_stats.add("clamped_loads", s_clamped_loads);
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
    throw std::runtime_error("MD1: maintenance requests unsupported");
  }

  bool send(Request& req) override {
    const bool is_read = (req.type_id == Request::Type::Read);
    if (!is_read && req.type_id != Request::Type::Write) {
      throw std::runtime_error("MD1 only supports Read/Write");
    }
    const Clk_t t = m_clk + 1;
    roll_phase(t);
    m_phase_accesses++;

    // M/D/1: W = rho * S / (2 * (1 - rho)), rho from the smoothed phase rate.
    float rho = std::min(m_smoothed_rate * static_cast<float>(m_burst), 0.95f);
    if (m_smoothed_rate * static_cast<float>(m_burst) > 0.95f) {
      s_clamped_loads++;
    }
    Clk_t wait = static_cast<Clk_t>(rho * static_cast<float>(m_burst) / (2.0f * (1.0f - rho)));
    Clk_t depart = t + m_rl + wait;

    if (m_trace_file.is_open()) {
      m_trace_file << m_clk << ',' << depart << ',' << req.type_id << ','
                   << req.source_id << ',' << req.addr << ",0,0,0,0,0\n";
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
    s_clamped_loads = 0;
  }

 private:
  struct PendingRead {
    Clk_t depart;
    Request req;
    bool operator>(const PendingRead& o) const {
      return depart > o.depart;
    }
  };

  void roll_phase(Clk_t t) {
    while (t >= m_phase_end) {
      float rate = static_cast<float>(m_phase_accesses) / static_cast<float>(m_phase_ticks);
      m_smoothed_rate = m_smoothing * rate + (1.0f - m_smoothing) * m_smoothed_rate;
      m_phase_accesses = 0;
      m_phase_end += m_phase_ticks;
    }
  }

  DRAMDevice m_device;
  int m_tCK_ps = 0;
  Clk_t m_rl = 0;
  Clk_t m_burst = 1;
  int m_phase_ticks = 10000;
  float m_smoothing = 0.5f;

  Clk_t m_clk = 0;
  Clk_t m_phase_end = 10000;
  size_t m_phase_accesses = 0;
  float m_smoothed_rate = 0.0f;
  std::priority_queue<PendingRead, std::vector<PendingRead>, std::greater<PendingRead>> m_pending;

  std::string m_trace_path;
  std::ofstream m_trace_file;

  Clk_t m_measured_clk = 0;
  size_t s_num_read_reqs = 0, s_num_write_reqs = 0;
  size_t s_num_read_reqs_served = 0, s_num_write_reqs_served = 0;
  size_t s_read_latency = 0;
  size_t s_clamped_loads = 0;
  float s_avg_read_latency = 0;
};

}  // namespace Ramulator
