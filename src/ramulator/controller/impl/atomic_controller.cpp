#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

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
    if (m_refresh != "none") {
      throw std::runtime_error(
          "Atomic skeleton supports refresh='none' only; refresh behavior is intentionally unspecified");
    }

    RAMULATOR_CREATE_CHILD(m_addr_mapper, IAddrMapper);
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
    req.depart = m_clk + m_latency;
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
