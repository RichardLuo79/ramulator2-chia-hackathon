#include <stdexcept>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

class ExternalFrontEnd : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, ExternalFrontEnd, "External")

 private:
  int m_num_cores = 1;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_num_cores, int, "num_cores").default_val(1);
    if (m_num_cores <= 0) {
      throw std::invalid_argument("External frontend num_cores must be positive");
    }
  }

  void tick() override {}

  bool is_finished() override { return false; }

  int get_num_cores() override { return m_num_cores; }

  bool receive_external_requests(int req_type_id, Addr_t addr, int source_id,
                                 std::function<void(Request&)> callback,
                                 int size_bytes) override {
    return receive_external_requests(req_type_id, addr, source_id, -1,
                                     std::move(callback), size_bytes);
  }

  bool receive_external_requests(int req_type_id, Addr_t addr, int source_id,
                                 int ingress_id,
                                 std::function<void(Request&)> callback,
                                 int size_bytes) override {
    return receive_external_requests(req_type_id, addr, source_id, ingress_id,
                                     -1, 0, std::move(callback), size_bytes);
  }

  bool receive_external_requests(int req_type_id, Addr_t addr, int source_id,
                                 int ingress_id, std::int64_t frontend_id,
                                 std::int64_t frontend_sub_id,
                                 std::function<void(Request&)> callback,
                                 int size_bytes) override {
    Request req(addr, req_type_id, source_id, std::move(callback));
    req.ingress_id = ingress_id;
    req.frontend_id = frontend_id;
    req.frontend_sub_id = frontend_sub_id;
    req.size_bytes = size_bytes;
    return m_memory_system->send(req);
  }
};

}  // namespace Ramulator
