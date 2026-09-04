#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/base/request.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/dram/device.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

// Model-zoo baseline: the Mess simulator (bsc-mem/Mess-simulator,
// Standalone/src/mess_mem_ctrl.cpp; MICRO 2024). Memory latency is servoed
// onto a family of bandwidth-latency curves, one curve per read percentage.
// Faithful port of the standalone semantics:
//   - measurement window closes every `window` accesses; measured bandwidth
//     = accesses/elapsed, read ratio = reads/accesses;
//   - bandwidth and applied latency are EMA-smoothed with convergeSpeed
//     alpha = 0.05;
//   - latency = linear interpolation on the nearest read-pct curve;
//   - above 99% of the curve's max bandwidth an overflow factor grows by
//     0.02 per update (decays by 0.01) and latency = (1+overflow)*curve max;
//   - applied latency never falls below the lead-off (minimum) latency.
// Curves here are fitted to OUR cycle-level oracle (not real hardware) via
// the lat-tp harness, making this a model-vs-model comparison. Curve file
// format: text lines "read_pct bandwidth_GBps latency_ns".
class MessController final : public IController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IController, MessController, "Mess")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_curve_path, std::string, "curve_path").required();
    RAMULATOR_PARSE_PARAM(m_window_accesses, int, "window_accesses").default_val(1000);
    RAMULATOR_PARSE_PARAM(m_converge, float, "converge").default_val(0.05f);
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "trace_path").default_val("");

    // RAMULATOR_CHILD: dram
    std::string dram_impl = m_config["dram"]["impl"].as<std::string>();
    m_device.init(DRAMSpec::create(dram_impl, m_config));
    m_tCK_ps = m_device.m_spec->get_timing_value("tCK_ps");
    load_curves();
    m_applied_lat = m_leadoff_lat;
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
    m_stats.add("overflow_updates", s_overflow_updates);
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

    m_win_count++;
    if (is_read) {
      m_win_reads++;
    }
    if (m_win_count >= static_cast<size_t>(m_window_accesses)) {
      update_latency(t);
    }

    Clk_t depart = t + static_cast<Clk_t>(m_applied_lat);
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
    s_overflow_updates = 0;
  }

 private:
  struct PendingRead {
    Clk_t depart;
    Request req;
    bool operator>(const PendingRead& o) const {
      return depart > o.depart;
    }
  };
  struct Curve {
    std::vector<double> bw;   // accesses per tick, ascending
    std::vector<double> lat;  // ticks
    double max_bw = 0.0;
    double max_lat = 0.0;
  };

  void load_curves() {
    // "read_pct bandwidth_GBps latency_ns" per line; converted to
    // accesses-per-tick and ticks at load (tx bytes + tCK), mirroring the
    // standalone's MB/s -> accesses-per-cycle and ns -> cycles conversions.
    std::ifstream f(m_curve_path);
    if (!f) {
      throw std::runtime_error(fmt::format("Mess: cannot open curve file '{}'", m_curve_path));
    }
    const double tck_ns = m_tCK_ps / 1000.0;
    const double tx = static_cast<double>(m_device.m_spec->get_tx_bytes());
    std::map<int, Curve> by_pct;
    int pct;
    double bw_gbps, lat_ns;
    while (f >> pct >> bw_gbps >> lat_ns) {
      Curve& c = by_pct[pct];
      c.bw.push_back(bw_gbps * tck_ns / tx);  // GB/s -> accesses/tick
      c.lat.push_back(lat_ns / tck_ns);       // ns -> ticks
    }
    if (by_pct.empty()) {
      throw std::runtime_error(fmt::format("Mess: no curve points in '{}'", m_curve_path));
    }
    m_leadoff_lat = std::numeric_limits<double>::max();
    for (auto& [p, c] : by_pct) {
      for (size_t i = 0; i < c.bw.size(); i++) {
        c.max_bw = std::max(c.max_bw, c.bw[i]);
        c.max_lat = std::max(c.max_lat, c.lat[i]);
        m_leadoff_lat = std::min(m_leadoff_lat, c.lat[i]);
      }
      m_pcts.push_back(p);
      m_curves.push_back(c);
    }
    // Nearest-curve lookup table over integer read pct [0,100], ties toward
    // the lower percentage (as the standalone's pctToCurveIdx sweeps do).
    for (int rp = 0; rp <= 100; rp++) {
      int best = 0;
      int bestd = 1 << 30;
      for (size_t i = 0; i < m_pcts.size(); i++) {
        int d = std::abs(m_pcts[i] - rp);
        if (d < bestd || (d == bestd && m_pcts[i] < m_pcts[best])) {
          bestd = d;
          best = static_cast<int>(i);
        }
      }
      m_pct_to_curve[rp] = best;
    }
  }

  void update_latency(Clk_t t) {
    const double elapsed = std::max<double>(static_cast<double>(t - m_win_start), 1.0);
    const double measured_bw = static_cast<double>(m_win_count) / elapsed;
    const int rp = static_cast<int>(
        100.0 * static_cast<double>(m_win_reads) / static_cast<double>(m_win_count) + 0.5);
    m_est_bw = m_converge * measured_bw + (1.0f - m_converge) * m_est_bw;

    const Curve& c = m_curves[m_pct_to_curve[std::clamp(rp, 0, 100)]];
    double target;
    if (m_est_bw > c.max_bw * 0.99) {
      m_overflow += 0.02;
      target = (1.0 + m_overflow) * c.max_lat;
      s_overflow_updates++;
    } else {
      m_overflow = std::max(0.0, m_overflow - 0.01);
      target = interpolate(c, m_est_bw);
      m_applied_lat = m_converge * target + (1.0f - m_converge) * m_applied_lat;
      m_applied_lat = std::max(m_applied_lat, m_leadoff_lat);
      m_win_count = m_win_reads = 0;
      m_win_start = t;
      return;
    }
    // Overflow path: latency jumps to the penalized curve max directly.
    m_applied_lat = m_converge * target + (1.0f - m_converge) * m_applied_lat;
    m_applied_lat = std::max(m_applied_lat, m_leadoff_lat);
    m_win_count = m_win_reads = 0;
    m_win_start = t;
  }

  static double interpolate(const Curve& c, double bw) {
    if (bw <= c.bw.front()) {
      return c.lat.front();
    }
    for (size_t i = 1; i < c.bw.size(); i++) {
      if (bw <= c.bw[i]) {
        double f = (bw - c.bw[i - 1]) / (c.bw[i] - c.bw[i - 1]);
        return c.lat[i - 1] + f * (c.lat[i] - c.lat[i - 1]);
      }
    }
    return c.lat.back();
  }

  DRAMDevice m_device;
  int m_tCK_ps = 0;
  std::string m_curve_path;
  int m_window_accesses = 1000;
  float m_converge = 0.05f;

  std::vector<int> m_pcts;
  std::vector<Curve> m_curves;
  int m_pct_to_curve[101] = {};
  double m_leadoff_lat = 0.0;

  Clk_t m_clk = 0;
  size_t m_win_count = 0, m_win_reads = 0;
  Clk_t m_win_start = 0;
  double m_est_bw = 0.0;
  double m_applied_lat = 0.0;
  double m_overflow = 0.0;
  std::priority_queue<PendingRead, std::vector<PendingRead>, std::greater<PendingRead>> m_pending;

  std::string m_trace_path;
  std::ofstream m_trace_file;

  Clk_t m_measured_clk = 0;
  size_t s_num_read_reqs = 0, s_num_write_reqs = 0;
  size_t s_num_read_reqs_served = 0, s_num_write_reqs_served = 0;
  size_t s_read_latency = 0;
  size_t s_overflow_updates = 0;
  float s_avg_read_latency = 0;
};

}  // namespace Ramulator
