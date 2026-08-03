#include <fmt/format.h>
#include <fstream>
#include <string>

#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

/// Records one line per completed read/write request at final-command issue.
///
/// CSV per channel (path suffixed ".ch0", ".ch1", ...):
///   arrive, depart, type, source, addr_vec...
/// depart for reads = issue_clk + read_latency (same formula as
/// ControllerBase::retire_request); for writes = issue_clk (fire-and-forget).
///
/// Example config (Python):
///   ramulator.controller_plugin.ReqTraceRecorder(path="reqs.csv")
class ReqTraceRecorder : public IControllerPlugin, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, ReqTraceRecorder, "ReqTraceRecorder")

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_path, std::string, "path").required();
  }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    m_ctrl = cast_parent<ControllerBase>();
    const auto& spec = *m_ctrl->get_spec();
    m_level_count = spec.level_count;
    m_read_latency = spec.read_latency;

    m_file.open(fmt::format("{}.ch{}", m_path, m_ctrl->m_channel_id));
    m_file << "arrive,depart,type,source";
    for (const auto& name : spec.level_names) {
      m_file << "," << name;
    }
    m_file << "\n";
  }

  void on_issue(const Request& req) override {
    if (req.command != req.final_command || req.type_id < 0) {
      return;
    }
    Clk_t depart = (req.type_id == Request::Type::Read) ? m_ctrl->m_clk + m_read_latency : m_ctrl->m_clk;
    m_file << req.arrive << "," << depart << "," << req.type_id << "," << req.source_id;
    for (int i = 0; i < m_level_count; i++) {
      m_file << "," << req.addr_vec[i];
    }
    m_file << "\n";
  }

  void finalize() override {
    if (m_file.is_open()) {
      m_file.close();
    }
  }

 private:
  ControllerBase* m_ctrl = nullptr;
  std::string m_path;
  int m_level_count = 0;
  Clk_t m_read_latency = 0;
  std::ofstream m_file;
};

}  // namespace Ramulator
