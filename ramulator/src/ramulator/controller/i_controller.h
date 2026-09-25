#ifndef RAMULATOR_CONTROLLER_I_CONTROLLER_H
#define RAMULATOR_CONTROLLER_I_CONTROLLER_H

#include <list>

#include "ramulator/base/base.h"

namespace Ramulator {

struct DRAMSpec;

class IController {
  RAMULATOR_REGISTER_INTERFACE(IController, "controller");

 public:
  int m_channel_id = -1;
  Clk_t m_clk = 0;
  unsigned int m_clock_ratio = 1;

  void print_stats(std::ostream& os) { m_impl->print_stats(os); }
  ConfigNode collect_stats() const { return m_impl->collect_stats(); }

  virtual void set_channel_id(int channel_id) {
    m_channel_id = channel_id;
  }

  virtual bool send(Request& req) = 0;
  virtual bool priority_send(Request& req) = 0;
  virtual void tick() = 0;

  virtual int get_tx_bytes() const = 0;
  virtual int get_num_levels() const = 0;
  virtual float get_tCK() const = 0;

  // The DRAM spec this controller drives. Lets sub-components (e.g. addr
  // mappers) query spec data without assuming a ControllerBase parent.
  virtual const DRAMSpec* get_spec() const = 0;

  // Tick-elision support. idle_ticks(): number of upcoming ticks guaranteed
  // to perform no observable action (0 = must be ticked next cycle).
  // fast_forward(k): advance state by k ticks; requires k <= idle_ticks().
  // Cycle-level controllers keep the defaults and are ticked every cycle.
  virtual Clk_t idle_ticks() {
    return 0;
  }
  virtual void fast_forward(Clk_t ticks) {
  }
};

struct ReqBuffer {
  std::list<Request> buffer;
  size_t max_size;

  explicit ReqBuffer(size_t max_size = 32) : max_size(max_size) {
  }

  using iterator = std::list<Request>::iterator;
  iterator begin() {
    return buffer.begin();
  }
  iterator end() {
    return buffer.end();
  }

  size_t size() const {
    return buffer.size();
  }

  bool enqueue(const Request& request) {
    if (buffer.size() < max_size) {
      buffer.push_back(request);
      return true;
    }
    return false;
  }

  bool enqueue(Request&& request) {
    if (buffer.size() < max_size) {
      buffer.push_back(std::move(request));
      return true;
    }
    return false;
  }

  void remove(iterator it) {
    buffer.erase(it);
  }
};

}  // namespace Ramulator

#endif  // RAMULATOR_CONTROLLER_I_CONTROLLER_H
