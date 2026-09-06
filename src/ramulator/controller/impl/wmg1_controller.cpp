#include <algorithm>
#include <deque>
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

// Model-zoo baseline: Sniper's QueueModelWindowedMG1
// (common/performance_model/queue_model_windowed_mg1.cc) applied to the DRAM
// channel, as Sniper's dram_perf_model does. A sliding time window holds
// recent requests' service times; the M/G/1 wait
//   t_q = lambda * E[S^2] / (2 * (1 - rho))
// is computed from window-measured lambda, E[S^2], and utilization
// (rho capped at 0.99, t_q clamped to the window — "if requesters do not
// throttle based on returned latency, it's their problem, not ours").
// Service time is the deterministic data-burst gap. Formula latency only —
// no banks, no physical serialization, faithful to the original.
class WMG1Controller final : public IController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IController, WMG1Controller, "WMG1")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_window_ns, int, "window_ns").default_val(10000);
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "trace_path").default_val("");

    // RAMULATOR_CHILD: dram
    std::string dram_impl = m_config["dram"]["impl"].as<std::string>();
    m_device.init(DRAMSpec::create(dram_impl, m_config));
    m_tCK_ps = m_device.m_spec->get_timing_value("tCK_ps");
    m_rl = m_device.m_spec->read_latency;
    m_burst = zoo_probe_burst_gap(m_device);
    m_window = std::max<Clk_t>(static_cast<Clk_t>(
        (static_cast<double>(m_window_ns) * 1000.0) / m_tCK_ps), 1);
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
    throw std::runtime_error("WMG1: maintenance requests unsupported");
  }

  bool send(Request& req) override {
    const bool is_read = (req.type_id == Request::Type::Read);
    if (!is_read && req.type_id != Request::Type::Write) {
      throw std::runtime_error("WMG1 only supports Read/Write");
    }
    const Clk_t t = m_clk + 1;

    // Advance the window and drop expired arrivals.
    while (!m_arrivals.empty() && m_arrivals.front() < t - m_window) {
      m_arrivals.pop_front();
      m_service_sum -= m_burst;
      m_service_sum2 -= static_cast<double>(m_burst) * static_cast<double>(m_burst);
    }

    Clk_t wait = 0;
    if (!m_arrivals.empty()) {
      const double n = static_cast<double>(m_arrivals.size());
      const double w = static_cast<double>(m_window);
      double rho = std::min(static_cast<double>(m_service_sum) / w, 0.99);
      double es2 = m_service_sum2 / n;
      double lambda = n / w;
      wait = static_cast<Clk_t>(lambda * es2 / (2.0 * (1.0 - rho)));
      wait = std::min<Clk_t>(wait, m_window);
    }
    m_arrivals.push_back(t);
    m_service_sum += m_burst;
    m_service_sum2 += static_cast<double>(m_burst) * static_cast<double>(m_burst);

    Clk_t depart = t + m_rl + wait;
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
  Clk_t m_rl = 0;
  Clk_t m_burst = 1;
  int m_window_ns = 10000;
  Clk_t m_window = 1;

  Clk_t m_clk = 0;
  std::deque<Clk_t> m_arrivals;
  Clk_t m_service_sum = 0;
  double m_service_sum2 = 0.0;
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
