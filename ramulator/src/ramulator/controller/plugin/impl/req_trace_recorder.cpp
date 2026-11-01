#include <fmt/format.h>
#include <fstream>
#include <string>

#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

/// Records one line per accepted read/write request when its departure is final.
///
/// CSV per channel (path suffixed ".ch0", ".ch1", ...):
///   arrive, depart, type, source, addr, frontend_id, frontend_sub_id,
///   admission_ordinal, addr_vec...
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

    m_file.open(fmt::format("{}.ch{}", m_path, m_ctrl->m_channel_id));
    m_file << "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal";
    for (const auto& name : spec.level_names) {
      m_file << "," << name;
    }
    m_file << "\n";
  }

  void on_request_departure_scheduled(const Request& req) override {
    if (req.type_id < 0) {
      return;
    }
    m_file << req.arrive << "," << req.depart << "," << req.type_id << ","
           << req.source_id << "," << req.addr << "," << req.frontend_id << ","
           << req.frontend_sub_id << "," << req.admission_ordinal;
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
  std::ofstream m_file;
};

}  // namespace Ramulator
