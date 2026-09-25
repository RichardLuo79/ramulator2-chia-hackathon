// Trusted callback/admission bookkeeping, independent of prediction heuristics.
#include "ramulator/controller/impl/atomic_controller_base.h"
#include <iostream>

using namespace Ramulator;

static void check(bool condition) {
  if (!condition) throw std::runtime_error("atomic admission check failed");
}

class TestMapper final : public IAddrMapper {
 public:
  void apply(Request& request) override { request.addr_vec = {0, 1, 2}; }
};

class TestController final : public AtomicControllerBase {
 public:
  int predictions = 0;
  Clk_t next_latency = 5;
  TestController() : AtomicControllerBase(ConfigNode{}, nullptr) {
    m_addr_mapper = &mapper;
    m_channel_id = 0;
    m_read_buffer_size = 2;
    m_write_buffer_size = 1;
  }
  std::string get_name() const override { return "TestAtomic"; }
  std::string get_ifce_name() const override { return "controller"; }

 private:
  TestMapper mapper;
  void init_model() override {}
  Clk_t predict_departure(const Request&) override {
    ++predictions;
    return m_clk + next_latency;
  }
};

int main() {
  TestController controller;
  std::vector<std::pair<int, Clk_t>> completed;
  auto request = [&](int type, int id) {
    Request value;
    value.type_id = type;
    value.addr = id * 64;
    value.size_bytes = 64;
    value.frontend_id = id;
    value.callback = [&](Request& done) { completed.emplace_back(done.frontend_id, done.depart); };
    return value;
  };
  auto first = request(Request::Type::Read, 1);
  auto second = request(Request::Type::Read, 2);
  auto third = request(Request::Type::Read, 3);
  auto write = request(Request::Type::Write, 4);
  auto rejected_write = request(Request::Type::Write, 5);
  check(controller.send(first) && first.depart == 5);
  controller.next_latency = 2;
  check(controller.send(second) && second.depart == 2);
  check(!controller.send(third) && controller.predictions == 2);
  check(controller.send(write));
  check(!controller.send(rejected_write) && controller.predictions == 3);
  auto invalid = request(42, 6);
  bool rejected = false;
  try { controller.send(invalid); } catch (const std::runtime_error&) { rejected = true; }
  check(rejected && controller.predictions == 3);

  // Later caller mutations and predictor state cannot revise committed copies.
  second.frontend_id = 999;
  second.depart = 999;
  second.callback = nullptr;
  controller.next_latency = 999;
  check(controller.idle_ticks() == 1);
  controller.fast_forward(1);
  check(completed.empty() && controller.predictions == 3);
  controller.tick();
  check(completed == std::vector<std::pair<int, Clk_t>>{{2, 2}, {4, 2}});
  controller.reset_stats();
  check(controller.idle_ticks() == 2);
  controller.fast_forward(2);
  controller.tick();
  check(completed == std::vector<std::pair<int, Clk_t>>{{2, 2}, {4, 2}, {1, 5}});
  check(controller.predictions == 3);
  controller.next_latency = 1;
  check(controller.send(third));
  controller.tick();
  check(completed.back() == std::pair<int, Clk_t>{3, 6});
  check(controller.predictions == 4);

  TestController invalid_departure;
  invalid_departure.next_latency = 0;
  rejected = false;
  try { invalid_departure.send(first); } catch (const std::runtime_error&) { rejected = true; }
  check(rejected);
  std::cout << "atomic admission, immutable departures, capacity and callback checks passed\n";
}
