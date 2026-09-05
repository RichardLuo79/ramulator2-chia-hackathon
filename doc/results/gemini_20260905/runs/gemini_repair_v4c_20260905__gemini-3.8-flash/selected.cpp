#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// CHIA_MODEL_INCLUDES_BEGIN
#include <deque>
#include <vector>
// CHIA_MODEL_INCLUDES_END

#include <fmt/format.h>

#include "ramulator/base/base.h"
#include "ramulator/base/param.h"
#include "ramulator/base/request.h"
#include "ramulator/controller/addr_mapper/i_addr_mapper.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/dram/device.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

/// Deliberately minimal seed for agent-driven controller synthesis.
///
/// The controller assigns every admitted request the same positive delay. It
/// does not model bank state, row locality, command timing, bus occupancy,
/// scheduling, write draining, or refresh. Separate bounded read and write
/// pools provide deterministic backpressure so the surrounding lifecycle and
/// closed-loop evaluator are exercised from the first iteration. Address
/// mapping is observational only: addresses never influence completion time.
class AtomicController final : public IController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IController, AtomicController, "Atomic")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_latency, int, "latency").default_val(-1);
    RAMULATOR_PARSE_PARAM(m_read_buffer_size, int, "read_buffer_size").default_val(64);
    RAMULATOR_PARSE_PARAM(m_write_buffer_size, int, "write_buffer_size").default_val(64);
    RAMULATOR_PARSE_PARAM(m_wr_low_watermark, float, "wr_low_watermark").default_val(0.5f);
    RAMULATOR_PARSE_PARAM(m_wr_high_watermark, float, "wr_high_watermark").default_val(0.8f);
    RAMULATOR_PARSE_PARAM(m_model_parameter_entries, std::vector<std::string>, "model_parameters").default_val({});
    RAMULATOR_PARSE_PARAM(m_refresh, std::string, "refresh").default_val("none");
    RAMULATOR_PARSE_PARAM(m_trace_path, std::string, "trace_path").default_val("");

    // RAMULATOR_CHILD: dram
    const std::string dram_impl = m_config["dram"]["impl"].as<std::string>();
    m_device.init(DRAMSpec::create(dram_impl, m_config));
    m_tCK_ps = m_device.m_spec->get_timing_value("tCK_ps");
    if (m_latency == -1) {
      m_latency = static_cast<int>(m_device.m_spec->read_latency);
    }
    if (m_latency <= 0) {
      throw std::runtime_error("Atomic latency must be positive");
    }
    if (m_read_buffer_size <= 0 || m_write_buffer_size <= 0) {
      throw std::runtime_error("Atomic buffer sizes must be positive");
    }
    if (!std::isfinite(m_wr_low_watermark) || !std::isfinite(m_wr_high_watermark) ||
        m_wr_low_watermark < 0 || m_wr_high_watermark > 1 || m_wr_low_watermark > m_wr_high_watermark) {
      throw std::runtime_error("Atomic requires 0 <= low <= high <= 1 write watermarks");
    }
    if (m_refresh != "none") {
      throw std::runtime_error(
          "Atomic skeleton supports refresh='none' only; refresh behavior is intentionally unspecified");
    }

    RAMULATOR_CREATE_CHILD(m_addr_mapper, IAddrMapper);
    parse_model_parameters();
    m_model_initializing = true;
    init_model();
    m_model_initializing = false;
    for (const auto& [name, value] : m_model_parameters) {
      if (!m_used_model_parameters.contains(name)) {
        throw std::runtime_error("Unknown Atomic model parameter: " + name);
      }
    }
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    if (!m_trace_path.empty()) {
      m_trace_file.open(fmt::format("{}.ch{}", m_trace_path, m_channel_id));
      if (!m_trace_file) {
        throw std::runtime_error(fmt::format("Atomic cannot open trace path '{}.ch{}'", m_trace_path, m_channel_id));
      }
      m_trace_file << "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal";
      for (const auto& level_name : m_device.m_spec->level_names) {
        m_trace_file << ',' << level_name;
      }
      m_trace_file << '\n';
    }

    m_stats.add("cycles", m_measured_clk);
    m_stats.add("num_read_reqs", s_num_read_reqs);
    m_stats.add("num_write_reqs", s_num_write_reqs);
    m_stats.add("num_read_reqs_served", s_num_read_reqs_served);
    m_stats.add("num_write_reqs_served", s_num_write_reqs_served);
    m_stats.add("read_latency", s_read_latency);
    m_stats.add("avg_read_latency", s_avg_read_latency);
    m_stats.add("send_rejects", s_send_rejects);
    m_stats.add("peak_inflight_reads", s_peak_inflight_reads);
    m_stats.add("peak_inflight_writes", s_peak_inflight_writes);
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
    throw std::runtime_error("Atomic skeleton does not model maintenance requests");
  }

  bool send(Request& req) override {
    const bool is_read = req.type_id == Request::Type::Read;
    if (!is_read && req.type_id != Request::Type::Write) {
      throw std::runtime_error(fmt::format(
          "Atomic skeleton supports only Read/Write requests, got type_id {}", req.type_id));
    }

    std::size_t& inflight = is_read ? m_inflight_reads : m_inflight_writes;
    const int capacity = is_read ? m_read_buffer_size : m_write_buffer_size;
    if (inflight >= static_cast<std::size_t>(capacity)) {
      s_send_rejects++;
      return false;
    }

    m_addr_mapper->apply(req);
    req.addr_vec[0] = m_channel_id;
    req.arrive = m_clk;
    req.depart = predict_departure(req);
    if (req.depart <= m_clk) {
      throw std::runtime_error("Atomic must commit a strictly future departure");
    }
    inflight++;
    if (is_read) {
      s_num_read_reqs++;
      s_peak_inflight_reads = std::max(s_peak_inflight_reads, inflight);
    } else {
      s_num_write_reqs++;
      s_peak_inflight_writes = std::max(s_peak_inflight_writes, inflight);
    }

    if (m_trace_file.is_open()) {
      m_trace_file << req.arrive << ',' << req.depart << ',' << req.type_id << ','
                   << req.source_id << ',' << req.addr << ',' << req.frontend_id << ','
                   << req.frontend_sub_id << ',' << req.admission_ordinal;
      for (int value : req.addr_vec) {
        m_trace_file << ',' << value;
      }
      m_trace_file << '\n';
    }

    m_pending.push(PendingRequest{req.depart, m_next_sequence++, req});
    return true;
  }

  void tick() override {
    m_clk++;
    m_measured_clk++;
    serve_completed();
  }

  Clk_t idle_ticks() override {
    if (m_pending.empty()) {
      return std::numeric_limits<Clk_t>::max();
    }
    const Clk_t next = m_pending.top().depart;
    return next > m_clk + 1 ? next - m_clk - 1 : 0;
  }

  void fast_forward(Clk_t ticks) override {
    m_clk += ticks;
    m_measured_clk += ticks;
  }

  void update_stats() override {
    s_avg_read_latency = s_num_read_reqs_served == 0
                             ? 0.0f
                             : static_cast<float>(s_read_latency) /
                                   static_cast<float>(s_num_read_reqs_served);
  }

  void reset_stats() override {
    m_measured_clk = 0;
    s_num_read_reqs = 0;
    s_num_write_reqs = 0;
    s_num_read_reqs_served = 0;
    s_num_write_reqs_served = 0;
    s_read_latency = 0;
    s_avg_read_latency = 0.0f;
    s_send_rejects = 0;
    s_peak_inflight_reads = m_inflight_reads;
    s_peak_inflight_writes = m_inflight_writes;
  }

  void finalize() override {
    if (m_trace_file.is_open()) {
      m_trace_file.close();
    }
    update_stats();
  }

 private:
  // Read-only configured behavior, not calibration constants.
  struct ModelControllerConfig {
    int read_buffer_size;
    int write_buffer_size;
    double wr_low_watermark;
    double wr_high_watermark;
  };

  ModelControllerConfig controller_config() const {
    return {m_read_buffer_size, m_write_buffer_size, m_wr_low_watermark, m_wr_high_watermark};
  }

  // Model-owned names, defaults, and valid ranges are declared in init_model().
  // Optional overrides use model_parameters=["name=value", ...] for every run.
  double model_param(const std::string& name, double fallback, double minimum, double maximum) {
    if (!m_model_initializing || name.empty() || name.size() > 64 ||
        name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos ||
        !std::isfinite(fallback) || !std::isfinite(minimum) || !std::isfinite(maximum) ||
        minimum > maximum || fallback < minimum || fallback > maximum) {
      throw std::runtime_error("Invalid Atomic model parameter declaration");
    }
    m_used_model_parameters.insert(name);
    if (m_used_model_parameters.size() > 128) {
      throw std::runtime_error("Too many Atomic model parameters");
    }
    const auto it = m_model_parameters.find(name);
    const double value = it == m_model_parameters.end() ? fallback : it->second;
    if (value < minimum || value > maximum) {
      throw std::runtime_error("Atomic model parameter outside declared range: " + name);
    }
    return value;
  }

  void parse_model_parameters() {
    if (m_model_parameter_entries.size() > 128) {
      throw std::runtime_error("Too many Atomic model parameter overrides");
    }
    for (const auto& entry : m_model_parameter_entries) {
      const auto equal = entry.find('=');
      if (equal == std::string::npos || equal == 0 || equal > 64 || entry.size() > 256) {
        throw std::runtime_error("Atomic model_parameters entries must be name=value");
      }
      const auto name = entry.substr(0, equal);
      const auto text = entry.substr(equal + 1);
      std::size_t consumed = 0;
      const double value = std::stod(text, &consumed);
      if (consumed != text.size() || !std::isfinite(value) || !m_model_parameters.emplace(name, value).second) {
        throw std::runtime_error("Invalid or duplicate Atomic model parameter: " + name);
      }
    }
  }

  // CHIA_MODEL_BEGIN
  struct ScheduledBurst {
    Clk_t depart;
    int bg;
  };

  // DRAM timing parameters (queried from m_device.m_spec in init_model())
  Clk_t m_read_lat = 0;
  Clk_t m_nBL = 0;
  Clk_t m_nRCD = 0;
  Clk_t m_nRP = 0;
  Clk_t m_nRAS = 0;
  Clk_t m_nRC = 0;
  Clk_t m_nCCDS = 0;
  Clk_t m_nCCDL = 0;
  Clk_t m_nCWL = 0;
  Clk_t m_nWTRS = 0;
  Clk_t m_t_WTR = 0;
  Clk_t m_nRTP = 0;
  double m_write_drain_gap = 0.0;

  // Controller configuration
  int m_read_buf_size = 0;
  int m_write_buf_size = 0;
  double m_wr_low_wm = 0.0;
  double m_wr_high_wm = 0.0;
  int m_write_high_wm = 0;
  int m_write_low_wm = 0;

  // Hierarchy levels
  int m_bank_level = -1;
  int m_row_level = -1;
  int m_rank_level = -1;
  int m_bg_level = -1;
  int m_num_ranks = 1;
  int m_num_bgs = 1;
  int m_num_banks = 1;
  int m_total_banks = 1;

  // Bounded dynamic state
  std::vector<int> m_bank_open_row;
  std::vector<Clk_t> m_bank_col_ready;
  std::vector<Clk_t> m_bank_act_ready;
  std::deque<ScheduledBurst> m_scheduled_departures;
  int m_buffered_writes = 0;
  Clk_t m_write_drain_end = 0;
  Clk_t m_last_bus_busy = 0;

  int require_timing(const DRAMSpec* spec, const std::string& name) const {
    if (!spec->has_timing(name)) {
      throw std::runtime_error("DRAMSpec missing required timing: " + name);
    }
    return spec->get_timing_value(name);
  }

  void init_model() {
    const auto cfg = controller_config();
    m_read_buf_size = cfg.read_buffer_size;
    m_write_buf_size = cfg.write_buffer_size;
    m_wr_low_wm = cfg.wr_low_watermark;
    m_wr_high_wm = cfg.wr_high_watermark;
    m_write_high_wm = std::max(1, static_cast<int>(m_wr_high_wm * m_write_buf_size));
    m_write_low_wm = std::max(0, static_cast<int>(m_wr_low_wm * m_write_buf_size));

    const DRAMSpec* spec = m_device.m_spec;
    if (spec->read_latency <= 0) {
      throw std::runtime_error("DRAMSpec read_latency must be positive");
    }
    m_read_lat = spec->read_latency;
    m_nBL = require_timing(spec, "nBL");
    m_nRCD = require_timing(spec, "nRCD");
    m_nRP = require_timing(spec, "nRP");
    m_nRAS = require_timing(spec, "nRAS");
    m_nRC = spec->has_timing("nRC") ? spec->get_timing_value("nRC") : (m_nRAS + m_nRP);
    m_nCCDS = spec->has_timing("nCCDS") ? spec->get_timing_value("nCCDS") : m_nBL;
    m_nCCDL = spec->has_timing("nCCDL") ? spec->get_timing_value("nCCDL") : m_nCCDS;
    m_nCWL = spec->has_timing("nCWL") ? spec->get_timing_value("nCWL") : require_timing(spec, "nCL");
    m_nWTRS = spec->has_timing("nWTRS") ? spec->get_timing_value("nWTRS")
            : (spec->has_timing("nWTR") ? spec->get_timing_value("nWTR") : 0);
    m_nRTP = require_timing(spec, "nRTP");
    m_t_WTR = m_nCWL + m_nBL + m_nWTRS;

    m_write_drain_gap = model_param("write_drain_gap", static_cast<double>(m_nBL), 1.0, 64.0);

    m_bank_level = spec->get_level_id("Bank");
    m_row_level = spec->get_level_id("Row");
    m_rank_level = spec->has_level("Rank") ? spec->get_level_id("Rank") : -1;
    m_bg_level = spec->has_level("BankGroup") ? spec->get_level_id("BankGroup") : -1;

    m_num_ranks = (m_rank_level >= 0) ? spec->get_level_size("Rank") : 1;
    m_num_bgs = (m_bg_level >= 0) ? spec->get_level_size("BankGroup") : 1;
    m_num_banks = spec->get_level_size("Bank");
    m_total_banks = std::max(1, m_num_ranks * m_num_bgs * m_num_banks);

    m_bank_open_row.assign(m_total_banks, -1);
    m_bank_col_ready.assign(m_total_banks, 0);
    m_bank_act_ready.assign(m_total_banks, 0);
    m_scheduled_departures.clear();
    m_buffered_writes = 0;
    m_write_drain_end = 0;
    m_last_bus_busy = 0;
  }

  Clk_t predict_departure(const Request& req) {
    const Clk_t now = m_clk;
    const bool is_read = (req.type_id == Request::Type::Read);

    // Evict expired departures from tracking deque
    while (!m_scheduled_departures.empty() && m_scheduled_departures.front().depart <= now) {
      m_scheduled_departures.pop_front();
    }

    // Account for background write draining during idle bus periods
    if (now > m_last_bus_busy && m_write_drain_gap > 0.0) {
      const Clk_t idle_ticks = now - m_last_bus_busy;
      const int drained = static_cast<int>(idle_ticks / m_write_drain_gap);
      m_buffered_writes = std::max(0, m_buffered_writes - drained);
      m_last_bus_busy = std::min(now, m_last_bus_busy + static_cast<Clk_t>(drained * m_write_drain_gap));
    }

    // Determine flat bank and row
    const int rank = (m_rank_level >= 0 && m_rank_level < static_cast<int>(req.addr_vec.size()))
                         ? req.addr_vec[m_rank_level]
                         : 0;
    const int bg = (m_bg_level >= 0 && m_bg_level < static_cast<int>(req.addr_vec.size()))
                       ? req.addr_vec[m_bg_level]
                       : 0;
    const int bank = (m_bank_level >= 0 && m_bank_level < static_cast<int>(req.addr_vec.size()))
                         ? req.addr_vec[m_bank_level]
                         : 0;
    int flat_bank = (rank * m_num_bgs + bg) * m_num_banks + bank;
    if (flat_bank < 0 || flat_bank >= m_total_banks) {
      flat_bank = 0;
    }
    const int row = (m_row_level >= 0 && m_row_level < static_cast<int>(req.addr_vec.size()))
                        ? req.addr_vec[m_row_level]
                        : 0;

    if (!is_read) {
      // WRITE handling:
      m_buffered_writes++;
      m_bank_open_row[flat_bank] = row;

      if (m_buffered_writes >= m_write_high_wm) {
        const int num_drain = m_buffered_writes - m_write_low_wm;
        m_buffered_writes = m_write_low_wm;
        const Clk_t drain_duration = static_cast<Clk_t>(num_drain * m_write_drain_gap + m_t_WTR);
        Clk_t drain_start = std::max(now, m_write_drain_end);
        if (!m_scheduled_departures.empty()) {
          drain_start = std::max(drain_start, m_scheduled_departures.back().depart);
        }
        m_write_drain_end = drain_start + drain_duration;
        m_last_bus_busy = std::max(m_last_bus_busy, m_write_drain_end);
      }
      return now + 1;
    }

    // READ handling:
    const int open_row = m_bank_open_row[flat_bank];
    Clk_t cmd_ready = now + 1;

    if (open_row == row) {
      // Row hit: row is already open, RD command only needs col_ready
      cmd_ready = std::max(now + 1, m_bank_col_ready[flat_bank]);
    } else if (open_row == -1) {
      // Empty bank: needs ACT then RD
      const Clk_t act_issue = std::max(now + 1, m_bank_act_ready[flat_bank]);
      cmd_ready = act_issue + m_nRCD;
      m_bank_open_row[flat_bank] = row;
    } else {
      // Row conflict: needs PREpb then ACT then RD
      const Clk_t pre_ready = (m_bank_act_ready[flat_bank] > m_nRP)
                                  ? (m_bank_act_ready[flat_bank] - m_nRP)
                                  : (now + 1);
      const Clk_t pre_issue = std::max(now + 1, pre_ready);
      const Clk_t act_issue = std::max(pre_issue + m_nRP, m_bank_act_ready[flat_bank]);
      cmd_ready = act_issue + m_nRCD;
      m_bank_open_row[flat_bank] = row;
    }

    // Earliest read departure based on bank command completion
    const Clk_t bank_data_avail = cmd_ready + m_read_lat;

    // Earliest read departure based on write drain blocking
    const Clk_t target_depart = std::max(bank_data_avail, m_write_drain_end + m_nBL);

    // Schedule on the shared channel data bus avoiding collisions with existing bursts
    Clk_t sched_depart = target_depart;
    for (const auto& burst : m_scheduled_departures) {
      const Clk_t gap = (burst.bg == bg) ? m_nCCDL : m_nCCDS;
      if (sched_depart + gap <= burst.depart) {
        break;
      }
      if (sched_depart < burst.depart + gap) {
        sched_depart = burst.depart + gap;
      }
    }

    // Update bank readiness based on actual command issue time
    const Clk_t actual_cmd = sched_depart - m_read_lat;
    m_bank_col_ready[flat_bank] = actual_cmd + m_nCCDL;
    if (open_row == row) {
      m_bank_act_ready[flat_bank] = std::max(m_bank_act_ready[flat_bank], actual_cmd + m_nRTP + m_nRP);
    } else {
      const Clk_t act_actual = actual_cmd - m_nRCD;
      m_bank_act_ready[flat_bank] = act_actual + m_nRC;
    }

    if (sched_depart <= now) {
      sched_depart = now + 1;
    }

    // Insert scheduled burst into sorted tracking deque
    ScheduledBurst new_burst{sched_depart, bg};
    auto it = std::upper_bound(
        m_scheduled_departures.begin(), m_scheduled_departures.end(), new_burst,
        [](const ScheduledBurst& a, const ScheduledBurst& b) {
          return a.depart < b.depart;
        });
    m_scheduled_departures.insert(it, new_burst);

    m_last_bus_busy = std::max(m_last_bus_busy, sched_depart);

    return sched_depart;
  }
  // CHIA_MODEL_END

  struct PendingRequest {
    Clk_t depart;
    std::uint64_t sequence;
    Request req;

    bool operator>(const PendingRequest& other) const {
      return depart != other.depart ? depart > other.depart : sequence > other.sequence;
    }
  };

  void serve_completed() {
    while (!m_pending.empty() && m_pending.top().depart <= m_clk) {
      Request req = m_pending.top().req;
      m_pending.pop();
      if (req.type_id == Request::Type::Read) {
        m_inflight_reads--;
        s_num_read_reqs_served++;
        s_read_latency += req.depart - req.arrive;
      } else {
        m_inflight_writes--;
        s_num_write_reqs_served++;
      }
      if (req.callback) {
        req.callback(req);
      }
    }
  }

  DRAMDevice m_device;
  IAddrMapper* m_addr_mapper = nullptr;
  int m_tCK_ps = 0;
  int m_latency = -1;
  int m_read_buffer_size = 64;
  int m_write_buffer_size = 64;
  float m_wr_low_watermark = 0.5f;
  float m_wr_high_watermark = 0.8f;
  std::vector<std::string> m_model_parameter_entries;
  std::map<std::string, double> m_model_parameters;
  std::set<std::string> m_used_model_parameters;
  bool m_model_initializing = false;
  std::string m_refresh = "none";

  std::priority_queue<PendingRequest, std::vector<PendingRequest>, std::greater<PendingRequest>> m_pending;
  std::uint64_t m_next_sequence = 0;
  std::size_t m_inflight_reads = 0;
  std::size_t m_inflight_writes = 0;

  std::string m_trace_path;
  std::ofstream m_trace_file;

  Clk_t m_measured_clk = 0;
  std::size_t s_num_read_reqs = 0;
  std::size_t s_num_write_reqs = 0;
  std::size_t s_num_read_reqs_served = 0;
  std::size_t s_num_write_reqs_served = 0;
  std::size_t s_read_latency = 0;
  std::size_t s_send_rejects = 0;
  std::size_t s_peak_inflight_reads = 0;
  std::size_t s_peak_inflight_writes = 0;
  float s_avg_read_latency = 0.0f;
};

}  // namespace Ramulator
