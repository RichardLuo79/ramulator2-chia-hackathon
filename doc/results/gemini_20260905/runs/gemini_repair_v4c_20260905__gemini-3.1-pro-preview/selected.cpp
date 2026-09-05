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
  int find_level(const std::string& target) const {
    for (size_t i = 0; i < m_device.m_spec->level_names.size(); ++i) {
      std::string name = m_device.m_spec->level_names[i];
      std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { 
          return (c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c; 
      });
      name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
      if (name == target) return static_cast<int>(i);
    }
    return -1;
  }

  int m_bg_level;
  int m_b_level;
  int m_r_level;
  int m_num_banks_per_bg;
  int m_num_flat_banks;

  int m_nBL;
  int m_lat_hit;
  int m_lat_empty;
  int m_lat_conflict;

  int m_occ_hit;
  int m_occ_empty;
  int m_occ_conflict;

  int m_write_high_wm_count;
  int m_write_low_wm_count;

  Clk_t m_last_arrive = 0;
  double m_bus_Q = 0.0;

  std::vector<Clk_t> m_bank_last_arrive;
  std::vector<double> m_bank_Q;
  std::vector<int> m_open_row;

  int m_pending_writes = 0;

  void init_model() {
    m_nBL = m_device.m_spec->get_timing_value("nBL");
    int nCL = m_device.m_spec->get_timing_value("nCL");
    int nRCD = m_device.m_spec->get_timing_value("nRCD");
    int nRP = m_device.m_spec->get_timing_value("nRP");

    m_lat_hit = nCL + m_nBL;
    m_lat_empty = nCL + nRCD + m_nBL;
    m_lat_conflict = nCL + nRCD + nRP + m_nBL;

    m_occ_hit = m_nBL;
    m_occ_empty = nRCD + m_nBL;
    m_occ_conflict = nRP + nRCD + m_nBL;

    m_bg_level = find_level("bankgroup");
    m_b_level = find_level("bank");
    m_r_level = find_level("row");

    int num_bgs = m_bg_level >= 0 ? m_device.m_spec->organization.level_sizes[m_bg_level] : 1;
    m_num_banks_per_bg = m_b_level >= 0 ? m_device.m_spec->organization.level_sizes[m_b_level] : 1;
    m_num_flat_banks = num_bgs * m_num_banks_per_bg;
    if (m_num_flat_banks <= 0) m_num_flat_banks = 1;

    m_open_row.assign(m_num_flat_banks, -1);
    m_bank_last_arrive.assign(m_num_flat_banks, 0);
    m_bank_Q.assign(m_num_flat_banks, 0.0);

    auto config = controller_config();
    m_write_high_wm_count = static_cast<int>(config.write_buffer_size * config.wr_high_watermark);
    m_write_low_wm_count = static_cast<int>(config.write_buffer_size * config.wr_low_watermark);

    m_last_arrive = 0;
    m_bus_Q = 0.0;
    m_pending_writes = 0;
  }

  Clk_t predict_departure(const Request& req) {
    bool is_read = (req.type_id == Request::Type::Read);

    int bg = (m_bg_level >= 0 && m_bg_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_bg_level] : 0;
    int b = (m_b_level >= 0 && m_b_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_b_level] : 0;
    int row = (m_r_level >= 0 && m_r_level < static_cast<int>(req.addr_vec.size())) ? req.addr_vec[m_r_level] : 0;
    
    int flat_bank = bg * m_num_banks_per_bg + b;
    if (flat_bank < 0 || flat_bank >= m_num_flat_banks) {
      flat_bank = 0;
    }

    int base_lat;
    double bank_occupancy;

    if (m_open_row[flat_bank] == row) {
      base_lat = m_lat_hit;
      bank_occupancy = m_occ_hit;
    } else if (m_open_row[flat_bank] != -1) {
      base_lat = m_lat_conflict;
      bank_occupancy = m_occ_conflict;
    } else {
      base_lat = m_lat_empty;
      bank_occupancy = m_occ_empty;
    }

    Clk_t bus_elapsed = req.arrive - m_last_arrive;
    if (bus_elapsed > 0) {
      m_bus_Q = std::max(0.0, m_bus_Q - static_cast<double>(bus_elapsed));
    }
    m_last_arrive = req.arrive;

    Clk_t bank_elapsed = req.arrive - m_bank_last_arrive[flat_bank];
    if (bank_elapsed > 0) {
      m_bank_Q[flat_bank] = std::max(0.0, m_bank_Q[flat_bank] - static_cast<double>(bank_elapsed));
    }
    m_bank_last_arrive[flat_bank] = req.arrive;

    double current_bus_wait = m_bus_Q;
    double current_bank_wait = m_bank_Q[flat_bank];

    m_open_row[flat_bank] = row;
    
    double wait_time = std::max(current_bus_wait, current_bank_wait);

    if (!is_read) {
      m_bank_Q[flat_bank] += bank_occupancy;
      m_pending_writes++;
      if (m_pending_writes >= m_write_high_wm_count) {
        int drain_count = m_pending_writes - m_write_low_wm_count;
        m_pending_writes = m_write_low_wm_count;
        m_bus_Q += static_cast<double>(drain_count * m_nBL);
      }
      return req.arrive + base_lat + static_cast<Clk_t>(wait_time);
    }

    m_bus_Q += static_cast<double>(m_nBL);
    m_bank_Q[flat_bank] += bank_occupancy;

    return req.arrive + base_lat + static_cast<Clk_t>(wait_time);
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
