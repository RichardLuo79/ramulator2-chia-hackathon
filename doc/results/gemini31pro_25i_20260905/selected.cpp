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
  int m_nBL = 0;
  int m_nRP = 0;
  int m_nRCD = 0;
  int m_nRC = 0;
  int m_nCCDL = 0;
  
  int m_level_rank = -1;
  int m_level_bg = -1;
  int m_level_bank = -1;
  int m_level_row = -1;
  
  int m_rank_count = 1;
  int m_bg_count = 1;
  int m_bank_count = 1;
  
  int m_wb_high = 51;
  int m_wb_low = 32;
  
  int m_virtual_writes = 0;
  Clk_t m_bus_avail = 0;
  
  std::vector<int> m_open_row;
  std::vector<Clk_t> m_bank_ready;

  double m_miss_penalty_weight = 0.6;
  double m_nRC_weight = 0.6;
  double m_wr_drain_overhead = 20.0;

  void init_model() {
    m_nBL = m_device.m_spec->get_timing_value("nBL");
    m_nRP = m_device.m_spec->get_timing_value("nRP");
    m_nRCD = m_device.m_spec->get_timing_value("nRCD");
    m_nRC = m_device.m_spec->get_timing_value("nRC");
    m_nCCDL = m_device.m_spec->get_timing_value("nCCDL");

    if (m_device.m_spec->has_level("Rank")) {
      m_level_rank = m_device.m_spec->get_level_id("Rank");
      m_rank_count = m_device.m_spec->get_level_size("Rank");
    }
    if (m_device.m_spec->has_level("BankGroup")) {
      m_level_bg = m_device.m_spec->get_level_id("BankGroup");
      m_bg_count = m_device.m_spec->get_level_size("BankGroup");
    }
    if (m_device.m_spec->has_level("Bank")) {
      m_level_bank = m_device.m_spec->get_level_id("Bank");
      m_bank_count = m_device.m_spec->get_level_size("Bank");
    }
    if (m_device.m_spec->has_level("Row")) {
      m_level_row = m_device.m_spec->get_level_id("Row");
    }

    int total_banks = m_rank_count * m_bg_count * m_bank_count;
    m_open_row.assign(total_banks, -1);
    m_bank_ready.assign(total_banks, 0);

    ModelControllerConfig cfg = controller_config();
    m_wb_high = static_cast<int>(cfg.write_buffer_size * cfg.wr_high_watermark);
    m_wb_low = static_cast<int>(cfg.write_buffer_size * cfg.wr_low_watermark);
    
    m_virtual_writes = 0;
    m_bus_avail = 0;

    m_miss_penalty_weight = model_param("miss_penalty_weight", 0.6, 0.0, 2.0);
    m_nRC_weight = model_param("nRC_weight", 0.6, 0.0, 2.0);
    m_wr_drain_overhead = model_param("wr_drain_overhead", 20.0, 0.0, 200.0);
  }

  Clk_t predict_departure(const Request& req) {
    if (req.type_id == Request::Type::Read) {
        int rank = m_level_rank != -1 ? req.addr_vec[m_level_rank] : 0;
        int bg = m_level_bg != -1 ? req.addr_vec[m_level_bg] : 0;
        int bank = m_level_bank != -1 ? req.addr_vec[m_level_bank] : 0;
        
        int flat_bank = (rank * m_bg_count + bg) * m_bank_count + bank;
        if (!m_open_row.empty()) {
            flat_bank = flat_bank % m_open_row.size();
        } else {
            flat_bank = 0;
        }
        
        int row = m_level_row != -1 ? req.addr_vec[m_level_row] : 0;
        
        bool is_hit = false;
        if (!m_open_row.empty()) {
            is_hit = (m_open_row[flat_bank] == row);
            m_open_row[flat_bank] = row;
        }
        
        Clk_t miss_delay = is_hit ? 0 : static_cast<Clk_t>((m_nRP + m_nRCD) * m_miss_penalty_weight);

        Clk_t bank_ready_time = m_bank_ready.empty() ? 0 : m_bank_ready[flat_bank];
        Clk_t issue_time = std::max(m_clk, bank_ready_time);
        
        if (!m_bank_ready.empty()) {
            if (!is_hit) {
                m_bank_ready[flat_bank] = issue_time + static_cast<Clk_t>(m_nRC * m_nRC_weight);
            } else {
                m_bank_ready[flat_bank] = issue_time + static_cast<Clk_t>(m_nCCDL);
            }
        }

        Clk_t base_delay = static_cast<Clk_t>(std::max(0, m_latency - m_nBL));
        Clk_t ideal_bus_start = issue_time + base_delay + miss_delay;
        
        Clk_t bus_start = std::max(ideal_bus_start, m_bus_avail);
        Clk_t depart = bus_start + static_cast<Clk_t>(m_nBL);
        
        m_bus_avail = depart;
        return depart;
    } else {
        m_virtual_writes++;
        if (m_virtual_writes >= m_wb_high) {
            int drain_count = std::max(1, m_virtual_writes - m_wb_low);
            m_virtual_writes -= drain_count;
            Clk_t drain_start = std::max(m_bus_avail, m_clk);
            m_bus_avail = drain_start + static_cast<Clk_t>(drain_count * m_nBL) + static_cast<Clk_t>(m_wr_drain_overhead);
        }
        return m_clk + static_cast<Clk_t>(m_latency); 
    }
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
