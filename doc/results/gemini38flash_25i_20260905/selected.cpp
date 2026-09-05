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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
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
  struct BankState {
    int open_row = -1;
    Clk_t ready_for_act = 0;
    Clk_t ready_for_rd = 0;
    Clk_t ready_for_pre = 0;
    Clk_t act_time = 0;
  };

  struct BusBurst {
    Clk_t start;
    Clk_t end;
  };

  int m_rank_level = -1;
  int m_bg_level = -1;
  int m_bank_level = -1;
  int m_row_level = -1;

  int m_num_ranks = 1;
  int m_num_bgs = 1;
  int m_num_banks_per_bg = 1;
  int m_total_banks = 1;

  Clk_t m_nCL = 0;
  Clk_t m_nBL = 0;
  Clk_t m_nRCD = 0;
  Clk_t m_nRP = 0;
  Clk_t m_nRAS = 0;
  Clk_t m_nRC = 0;
  Clk_t m_nCWL = 0;
  Clk_t m_nWR = 0;
  Clk_t m_nRTP = 0;
  Clk_t m_nCCDS = 0;
  Clk_t m_nCCDL = 0;
  Clk_t m_nWTRL = 0;
  Clk_t m_nRRDS = 0;

  Clk_t m_read_lat = 0;
  Clk_t m_write_lat = 0;

  double m_drain_overhead = 4.0;
  double m_init_write_interval = 64.0;
  double m_max_write_interval_sample = 10000.0;
  double m_write_interval_ema_alpha = 0.05;
  double m_avg_write_interval = 64.0;

  std::vector<BankState> m_banks;
  std::vector<BusBurst> m_bus_bursts;
  Clk_t m_write_drain_until = 0;
  Clk_t m_last_write_bus_end = 0;
  Clk_t m_last_write_arrival = 0;

  void init_model() {
    const auto* spec = m_device.m_spec;
    m_rank_level = spec->has_level("Rank") ? spec->get_level_id("Rank") : -1;
    m_bg_level = spec->has_level("BankGroup") ? spec->get_level_id("BankGroup") : -1;
    m_bank_level = spec->has_level("Bank") ? spec->get_level_id("Bank") : -1;
    m_row_level = spec->has_level("Row") ? spec->get_level_id("Row") : -1;

    m_num_ranks = (m_rank_level >= 0) ? spec->get_level_size("Rank") : 1;
    m_num_bgs = (m_bg_level >= 0) ? spec->get_level_size("BankGroup") : 1;
    m_num_banks_per_bg = (m_bank_level >= 0) ? spec->get_level_size("Bank") : 1;
    m_total_banks = std::max(1, m_num_ranks * m_num_bgs * m_num_banks_per_bg);

    m_nCL = static_cast<Clk_t>(spec->get_timing_value("nCL"));
    m_nBL = static_cast<Clk_t>(spec->get_timing_value("nBL"));
    m_nRCD = static_cast<Clk_t>(spec->get_timing_value("nRCD"));
    m_nRP = static_cast<Clk_t>(spec->get_timing_value("nRP"));
    m_nRAS = static_cast<Clk_t>(spec->get_timing_value("nRAS"));
    m_nRC = static_cast<Clk_t>(spec->get_timing_value("nRC"));
    m_nCWL = static_cast<Clk_t>(spec->get_timing_value("nCWL"));
    m_nWR = static_cast<Clk_t>(spec->get_timing_value("nWR"));
    m_nRTP = static_cast<Clk_t>(spec->get_timing_value("nRTP"));

    m_nCCDS = spec->has_timing("nCCDS") ? static_cast<Clk_t>(spec->get_timing_value("nCCDS")) : m_nBL;
    m_nCCDL = spec->has_timing("nCCDL") ? static_cast<Clk_t>(spec->get_timing_value("nCCDL")) : m_nCCDS;
    m_nWTRL = spec->has_timing("nWTRL") ? static_cast<Clk_t>(spec->get_timing_value("nWTRL")) :
              (spec->has_timing("nWTRS") ? static_cast<Clk_t>(spec->get_timing_value("nWTRS")) : 0);
    m_nRRDS = spec->has_timing("nRRDS") ? static_cast<Clk_t>(spec->get_timing_value("nRRDS")) :
              (spec->has_timing("nRRD") ? static_cast<Clk_t>(spec->get_timing_value("nRRD")) : 0);

    m_read_lat = (spec->read_latency > 0) ? spec->read_latency : (m_nCL + m_nBL);
    m_write_lat = m_nCWL + m_nBL;

    m_drain_overhead = model_param("drain_overhead", 4.0, 0.0, 32.0);
    m_init_write_interval = model_param("init_write_interval", 64.0, 1.0, 10000.0);
    m_max_write_interval_sample = model_param("max_write_interval_sample", 10000.0, 100.0, 1000000.0);
    m_write_interval_ema_alpha = model_param("write_interval_ema_alpha", 0.05, 0.001, 1.0);

    m_avg_write_interval = m_init_write_interval;
    m_banks.assign(m_total_banks, BankState{});
    m_bus_bursts.clear();
    m_write_drain_until = 0;
    m_last_write_bus_end = 0;
    m_last_write_arrival = 0;
  }

  Clk_t schedule_burst(Clk_t earliest_start, Clk_t len) {
    Clk_t t = earliest_start;
    for (const auto& b : m_bus_bursts) {
      if (t + len <= b.start) {
        break;
      }
      if (t < b.end) {
        t = b.end;
      }
    }
    BusBurst new_burst{t, t + len};
    auto it = std::lower_bound(m_bus_bursts.begin(), m_bus_bursts.end(), new_burst,
                               [](const BusBurst& a, const BusBurst& b) {
                                 return a.start < b.start;
                               });
    m_bus_bursts.insert(it, new_burst);
    return t;
  }

  Clk_t predict_departure(const Request& req) {
    const bool is_read = (req.type_id == Request::Type::Read);
    const auto cfg = controller_config();
    const int wr_high = std::max(1, static_cast<int>(cfg.wr_high_watermark * cfg.write_buffer_size));
    const int wr_low = std::max(0, static_cast<int>(cfg.wr_low_watermark * cfg.write_buffer_size));

    int rank = (m_rank_level >= 0 && m_rank_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_rank_level] : 0;
    int bg = (m_bg_level >= 0 && m_bg_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_bg_level] : 0;
    int bank = (m_bank_level >= 0 && m_bank_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_bank_level] : 0;
    int row = (m_row_level >= 0 && m_row_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_row_level] : 0;

    int flat_bank = (rank * m_num_bgs + bg) * m_num_banks_per_bg + bank;
    if (flat_bank < 0 || flat_bank >= m_total_banks) {
      flat_bank = 0;
    }

    // Prune completed bursts from bus tracker
    std::size_t write_idx = 0;
    for (std::size_t i = 0; i < m_bus_bursts.size(); ++i) {
      if (m_bus_bursts[i].end > m_clk) {
        if (write_idx != i) {
          m_bus_bursts[write_idx] = m_bus_bursts[i];
        }
        write_idx++;
      }
    }
    m_bus_bursts.resize(write_idx);

    if (!is_read) {
      if (m_last_write_arrival > 0 && m_clk > m_last_write_arrival) {
        Clk_t delta = m_clk - m_last_write_arrival;
        if (delta < static_cast<Clk_t>(m_max_write_interval_sample)) {
          m_avg_write_interval = (1.0 - m_write_interval_ema_alpha) * m_avg_write_interval +
                                 m_write_interval_ema_alpha * static_cast<double>(delta);
        }
      }
      m_last_write_arrival = m_clk;

      const std::size_t next_inflight_writes = m_inflight_writes + 1;

      if (next_inflight_writes >= static_cast<std::size_t>(wr_high)) {
        const int num_drained = std::max(1, wr_high - wr_low);
        const Clk_t drain_duration = static_cast<Clk_t>(num_drained * (m_nBL + m_drain_overhead));
        if (m_clk >= m_write_drain_until) {
          Clk_t drain_start = std::max(m_clk + 1, m_last_write_bus_end);
          m_write_drain_until = drain_start + drain_duration + m_nWTRL;
          m_last_write_bus_end = drain_start + drain_duration;
        } else {
          m_write_drain_until += static_cast<Clk_t>(m_nBL + m_drain_overhead);
          m_last_write_bus_end = m_write_drain_until - m_nWTRL;
        }
        BankState& b = m_banks[flat_bank];
        b.open_row = row;
        b.act_time = m_clk;
        b.ready_for_act = m_clk + m_nRC;
        b.ready_for_pre = m_clk + m_nCWL + m_nBL + m_nWR;
        b.ready_for_rd = m_clk + m_nCWL + m_nBL + m_nWTRL;
        return std::max(m_write_drain_until, m_clk + 1);
      }

      if (m_inflight_reads == 0 && m_clk >= m_write_drain_until) {
        Clk_t write_start = std::max(m_clk + 1, m_last_write_bus_end);
        Clk_t burst_start = schedule_burst(write_start, m_nBL);
        m_last_write_bus_end = burst_start + m_nBL;
        Clk_t depart = burst_start + m_write_lat;
        BankState& b = m_banks[flat_bank];
        b.open_row = row;
        b.act_time = write_start;
        b.ready_for_act = write_start + m_nRC;
        b.ready_for_pre = write_start + m_nCWL + m_nBL + m_nWR;
        b.ready_for_rd = write_start + m_nCWL + m_nBL + m_nWTRL;
        return std::max(depart, m_clk + 1);
      }

      const int remaining = wr_high - static_cast<int>(next_inflight_writes);
      const Clk_t est_delay = static_cast<Clk_t>(std::max(0, remaining) * m_avg_write_interval);
      const Clk_t drain_start = std::max(m_clk + est_delay, m_write_drain_until);
      return std::max(drain_start + m_write_lat, m_clk + 1);
    }

    BankState& b = m_banks[flat_bank];
    Clk_t cas_issue = 0;

    if (b.open_row == row) {
      cas_issue = std::max(m_clk + 1, b.ready_for_rd);
      b.ready_for_rd = cas_issue + m_nCCDL;
      b.ready_for_pre = std::max(b.ready_for_pre, cas_issue + m_nRTP);
    } else if (b.open_row == -1) {
      Clk_t act_issue = std::max(m_clk + 1, b.ready_for_act);
      cas_issue = act_issue + m_nRCD;
      b.open_row = row;
      b.act_time = act_issue;
      b.ready_for_act = act_issue + m_nRC;
      b.ready_for_pre = act_issue + m_nRAS;
      b.ready_for_rd = cas_issue + m_nCCDL;
    } else {
      Clk_t pre_ready = std::max(m_clk + 1, b.ready_for_pre);
      Clk_t act_issue = std::max(b.ready_for_act, pre_ready + m_nRP);
      cas_issue = act_issue + m_nRCD;
      b.open_row = row;
      b.act_time = act_issue;
      b.ready_for_act = act_issue + m_nRC;
      b.ready_for_pre = act_issue + m_nRAS;
      b.ready_for_rd = cas_issue + m_nCCDL;
    }

    Clk_t earliest_burst = cas_issue + m_nCL;
    if (m_write_drain_until > 0 && earliest_burst < m_write_drain_until) {
      earliest_burst = m_write_drain_until;
    }
    if (m_last_write_bus_end > 0 && earliest_burst < m_last_write_bus_end + m_nWTRL) {
      earliest_burst = m_last_write_bus_end + m_nWTRL;
    }
    earliest_burst = std::max(earliest_burst, m_clk + 1);

    Clk_t burst_start = schedule_burst(earliest_burst, m_nBL);
    if (burst_start > earliest_burst) {
      Clk_t actual_cas = burst_start - m_nCL;
      b.ready_for_rd = std::max(b.ready_for_rd, actual_cas + m_nCCDL);
      b.ready_for_pre = std::max(b.ready_for_pre, actual_cas + m_nRTP);
    }

    Clk_t depart = burst_start + m_nBL;
    return std::max(depart, m_clk + 1);
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
