#ifndef CHAMPSIM_RAMULATOR_BRIDGE_H
#define CHAMPSIM_RAMULATOR_BRIDGE_H

// Ramulator2 memory backend for ChampSim's MEMORY_CONTROLLER.
//
// Selected at runtime via environment variables (unset = vanilla ChampSim
// DRAM model):
//   RAMULATOR_CONFIG = path to a generated config JSON (ramcfg/<model>_<std>.json)
//
// Both models run behind this same bridge with the same clocking (one
// ramulator controller tick per MEMORY_CONTROLLER::operate()), mirroring the
// SimpleO3 validation harness configs (refresh disabled on both sides, 64B
// transactions, RoBaRaCoCh).

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "channel.h"
#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"
#include "ramulator_bridge_identity.h"

class RamulatorBridge
{
  using channel_type = champsim::channel;
  using request_type = typename channel_type::request_type;
  using response_type = typename channel_type::response_type;

  struct Pending {
    request_type pkt;
    channel_type* ul;
  };

  std::unique_ptr<Ramulator::IFrontEnd> m_fe;
  std::unique_ptr<Ramulator::IMemorySystem> m_ms;
  std::unordered_map<uint64_t, Pending> m_inflight;
  std::deque<Pending> m_done;
  uint64_t m_next_token = 0;
  int m_tx_bytes;
  std::FILE* m_reqtrace = nullptr;
  uint64_t m_cycle = 0;
  struct Window {
    int64_t begin = 0, end = -1;
    uint64_t reads = 0, begin_reads = 0, measured_reads = 0;
  };
  std::vector<Window> m_windows = std::vector<Window>(NUM_CPUS);
  int64_t m_next_admission = 0;

  void note_admission(const Ramulator::Request& req)
  {
    if (req.admission_ordinal < m_next_admission || req.source_id < 0 ||
        static_cast<std::size_t>(req.source_id) >= NUM_CPUS)
      throw std::runtime_error("invalid admission identity for measurement window");
    m_next_admission = req.admission_ordinal + 1;
    if (req.type_id == Ramulator::Request::Type::Read)
      ++m_windows.at(req.source_id).reads;
  }

public:
  void measurement_begin()
  {
    for (auto& window : m_windows) {
      window.begin = m_next_admission;
      window.begin_reads = window.reads;
    }
  }

  void measurement_end(unsigned cpu)
  {
    auto& window = m_windows.at(cpu);
    if (window.end != -1)
      throw std::runtime_error("measurement window ended twice");
    window.end = m_next_admission;
    window.measured_reads = window.reads - window.begin_reads;
  }

  void measurement_report() const
  {
    for (unsigned cpu = 0; cpu < m_windows.size(); ++cpu) {
      const auto& w = m_windows[cpu];
      std::printf("CONTENTION_WINDOW {\"core\":%u,\"begin\":%lld,\"end\":%lld,\"admitted_reads\":%llu,\"total_admitted_reads\":%llu}\n",
          cpu, static_cast<long long>(w.begin), static_cast<long long>(w.end),
          static_cast<unsigned long long>(w.measured_reads), static_cast<unsigned long long>(w.reads));
    }
  }

  explicit RamulatorBridge(const std::string& cfg_path)
  {
    Ramulator::ConfigNode cfg = Ramulator::Config::parse_config_file(cfg_path);
    if (NUM_CPUS == 0 || NUM_CPUS > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("ChampSim NUM_CPUS does not fit Ramulator's source-id domain");
    }
    auto frontend_cfg = cfg["frontend"];
    frontend_cfg.set("num_cores", Ramulator::ConfigNode(static_cast<int>(NUM_CPUS)));
    cfg.set("frontend", std::move(frontend_cfg));
    m_fe.reset(Ramulator::Factory::create_frontend(cfg));
    m_ms.reset(Ramulator::Factory::create_memory_system(cfg));
    m_fe->connect_memory_system(m_ms.get());
    m_ms->connect_frontend(m_fe.get());
    m_tx_bytes = m_ms->get_tx_bytes();
    if (const char* rt = std::getenv("RAMULATOR_REQTRACE")) {
      m_reqtrace = std::fopen(rt, "w");
      std::fprintf(m_reqtrace, "cycle,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal\n");
    }
  }

  static RamulatorBridge* from_env()
  {
    // RAMULATOR_CONFIG = path to a python-wrapper-generated config (presets
    // are resolved python-side; the C++ layer takes only expanded configs).
    const char* path = std::getenv("RAMULATOR_CONFIG");
    if (path == nullptr)
      return nullptr;
    return new RamulatorBridge(std::string{path});
  }

  // Returns false on backpressure (leave the packet queued upstream).
  bool send_read(const request_type& pkt, channel_type* ul)
  {
    uint64_t addr = pkt.address.to<uint64_t>();
    const auto identity = ramulator_bridge::make_request_identity(pkt.cpu, pkt.instr_id, pkt.v_address.to<uint64_t>(), addr, pkt.type, m_tx_bytes);
    const uint64_t token = m_next_token++;
    Ramulator::Request req(static_cast<Ramulator::Addr_t>(addr), Ramulator::Request::Type::Read);
    req.source_id = identity.source_id;
    req.frontend_id = identity.frontend_id;
    req.frontend_sub_id = identity.frontend_sub_id;
    req.size_bytes = m_tx_bytes;
    req.callback = [this, token](Ramulator::Request&) {
      auto it = m_inflight.find(token);
      if (it != m_inflight.end()) {
        m_done.push_back(std::move(it->second));
        m_inflight.erase(it);
      }
    };
    if (!m_ms->send(req))
      return false;
    note_admission(req);
    if (m_reqtrace != nullptr) {
      std::fprintf(m_reqtrace, "%lu,0,%d,%lu,%lld,%lld,%lld\n", static_cast<unsigned long>(m_cycle), req.source_id, static_cast<unsigned long>(addr),
                   static_cast<long long>(req.frontend_id), static_cast<long long>(req.frontend_sub_id), static_cast<long long>(req.admission_ordinal));
    }
    m_inflight.emplace(token, Pending{pkt, ul});
    return true;
  }

  bool send_write(const request_type& pkt)
  {
    uint64_t addr = pkt.address.to<uint64_t>();
    const auto identity = ramulator_bridge::make_request_identity(pkt.cpu, pkt.instr_id, pkt.v_address.to<uint64_t>(), addr, pkt.type, m_tx_bytes);
    Ramulator::Request req(static_cast<Ramulator::Addr_t>(addr), Ramulator::Request::Type::Write);
    req.source_id = identity.source_id;
    // Writes reaching DRAM are cache writebacks. Their ChampSim instr_id is
    // the request that evicted the line, not the request that dirtied it.
    req.frontend_id = -1;
    req.frontend_sub_id = 0;
    req.size_bytes = m_tx_bytes;
    bool ok = m_ms->send(req);
    if (ok)
      note_admission(req);
    if (ok && m_reqtrace != nullptr) {
      std::fprintf(m_reqtrace, "%lu,1,%d,%lu,%lld,%lld,%lld\n", static_cast<unsigned long>(m_cycle), req.source_id, static_cast<unsigned long>(addr),
                   static_cast<long long>(req.frontend_id), static_cast<long long>(req.frontend_sub_id), static_cast<long long>(req.admission_ordinal));
    }
    return ok;
  }

  // Clock ratio: RAMULATOR_TICKS_PER_8 controller ticks per 8 MC cycles
  // (default 8 = 1:1; the SimpleO3 harness ratio is 3:8).
  int m_ticks_per_8 = [] {
    const char* e = std::getenv("RAMULATOR_TICKS_PER_8");
    return e ? std::atoi(e) : 8;
  }();
  int m_tick_acc = 0;

  void tick()
  {
    ++m_cycle;
    m_tick_acc += m_ticks_per_8;
    while (m_tick_acc >= 8) {
      m_tick_acc -= 8;
      m_ms->tick();
    }
    while (!m_done.empty()) {
      Pending& p = m_done.front();
      if (p.ul != nullptr) {
        response_type resp{p.pkt.address, p.pkt.v_address, p.pkt.data, p.pkt.pf_metadata, p.pkt.instr_depend_on_me};
        p.ul->returned.push_back(resp);
      }
      m_done.pop_front();
    }
  }

  void finalize() { m_ms->finalize(); }
};

#endif
