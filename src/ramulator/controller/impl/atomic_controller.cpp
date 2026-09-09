// Historical editable-region seed. New framework candidates implement the
// separate atomic-model API instead of inheriting the trusted controller.
#include "ramulator/controller/impl/atomic_controller_base.h"

// CHIA_MODEL_INCLUDES_BEGIN
// CHIA_MODEL_INCLUDES_END

namespace Ramulator {
class AtomicController final : public AtomicControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, AtomicController, AtomicControllerBase, "Atomic")

 private:
  // CHIA_MODEL_BEGIN
  void init_model() override {}

  Clk_t predict_departure(const Request& req) override {
    return m_clk + m_latency;
  }
  // CHIA_MODEL_END
};
}  // namespace Ramulator
