#include "ramulator_bridge_iface.h"

#include <functional>
#include <iterator>
#include <memory>

#include "channel.h"
#include "ramulator_bridge.h"

namespace
{
std::unique_ptr<RamulatorBridge> g_bridge;
bool g_checked = false;
} // namespace

namespace ramulator_bridge
{

bool active()
{
  if (!g_checked) {
    g_bridge.reset(RamulatorBridge::from_env());
    g_checked = true;
  }
  return g_bridge != nullptr;
}

long warmup_cycle(std::vector<champsim::channel*>& queues)
{
  long progress{0};
  for (auto* ul : queues) {
    for (auto q : {std::ref(ul->RQ), std::ref(ul->PQ)}) {
      for (auto& pkt : q.get()) {
        if (pkt.response_requested) {
          champsim::channel::response_type response{pkt.address, pkt.v_address, pkt.data, pkt.pf_metadata, pkt.instr_depend_on_me};
          ul->returned.push_back(response);
        }
        ++progress;
      }
      q.get().clear();
    }
    progress += static_cast<long>(std::size(ul->WQ));
    ul->WQ.clear();
  }
  return progress;
}

long cycle(std::vector<champsim::channel*>& queues)
{
  long progress{0};
  for (auto* ul : queues) {
    for (auto q : {std::ref(ul->RQ), std::ref(ul->PQ)}) {
      while (!q.get().empty() && g_bridge->send_read(q.get().front(), q.get().front().response_requested ? ul : nullptr)) {
        q.get().pop_front();
        ++progress;
      }
    }
    while (!ul->WQ.empty() && g_bridge->send_write(ul->WQ.front())) {
      ul->WQ.pop_front();
      ++progress;
    }
  }
  g_bridge->tick();
  return progress + 1;
}

} // namespace ramulator_bridge
