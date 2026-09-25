// Drive the actual shared-library controllers; no copied prediction formulas.
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include "ramulator/base/base.h"
#include "ramulator/base/config.h"
#include "ramulator/controller/i_controller.h"
#include "ramulator/dram/dram_spec.h"

int main(int argc, char** argv) {
  using namespace Ramulator;
  if (argc != 3) throw std::runtime_error("usage: baseline_trace_driver config.yaml requests.txt");
  auto config = Config::parse_config_file(argv[1]);
  std::unique_ptr<Implementation> impl(Factory::create_implementation("controller", config, nullptr));
  auto* controller = dynamic_cast<IController*>(impl.get());
  if (!controller) throw std::runtime_error("not a controller");
  std::cerr << "read_latency=" << controller->get_spec()->read_latency
            << " tCK_ps=" << controller->get_spec()->get_timing_value("tCK_ps") << '\n';
  impl->setup(nullptr, nullptr);
  std::ifstream file;
  if (std::string(argv[2]) != "-") file.open(argv[2]);
  std::istream& input = std::string(argv[2]) == "-" ? std::cin : file;
  std::cout << std::unitbuf;
  Clk_t clock = 0, arrival;
  int type, reset;
  long id = 0, callbacks = 0;
  std::vector<bool> completed;
  Clk_t previous_read_completion = -1;
  while (input >> arrival >> type >> reset) {
    if (arrival < clock) throw std::runtime_error("nonmonotonic input");
    while (clock < arrival) { controller->tick(); ++clock; }
    if (reset) impl->reset_stats();
    Request req(id * 64, type);
    req.source_id = 0; req.frontend_id = id++; req.size_bytes = 64;
    completed.push_back(false);
    req.callback = [&](Request& response) {
      if (completed.at(response.frontend_id)) throw std::runtime_error("duplicate callback");
      completed[response.frontend_id] = true;
      if (response.type_id == Request::Type::Read) {
        if (response.depart != clock + 1 || response.depart < previous_read_completion)
          throw std::runtime_error("incorrect read completion edge/order");
        previous_read_completion = response.depart;
      }
      ++callbacks;
    };
    const bool accepted = controller->send(req);
    if (!accepted) throw std::runtime_error("baseline unexpectedly rejected an admission");
    if (type == Request::Type::Write && !completed.at(req.frontend_id))
      throw std::runtime_error("posted write was not acknowledged at admission");
    if (type == Request::Type::Read && (completed.at(req.frontend_id) || req.depart <= arrival))
      throw std::runtime_error("read did not retain a future immutable completion");
    std::cout << arrival << ' ' << accepted << ' ' << req.depart << ' ' << callbacks << '\n';
  }
  while (callbacks < id) {
    auto skip = controller->idle_ticks();
    if (skip > 0 && skip < 1000000000) { controller->fast_forward(skip); clock += skip; }
    controller->tick(); ++clock;
    if (clock > 1000000000) throw std::runtime_error("callbacks failed to drain");
  }
  impl->finalize();
}
