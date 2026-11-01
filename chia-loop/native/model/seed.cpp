#include "ramulator/controller/atomic_model/api.h"

namespace {
// Deliberately no bank, bus, row, queueing or scheduling prediction logic.
class Seed final : public Ramulator::AtomicModel::Model {
 public:
  void initialize(Ramulator::AtomicModel::Hardware hardware,
                  Ramulator::AtomicModel::Parameters&) override {
    latency = hardware.read_latency;
  }

  Ramulator::AtomicModel::Cycle predict(Ramulator::AtomicModel::Access,
                                      Ramulator::AtomicModel::Cycle now) override {
    return now + latency;
  }

 private:
  Ramulator::AtomicModel::Cycle latency = 1;
};
}  // namespace

extern "C" std::uint32_t ramulator_atomic_model_api_version() {
  return Ramulator::AtomicModel::api_version;
}

extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model() {
  return new Seed;
}
