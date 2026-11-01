#include "ramulator/controller/atomic_model/loader.h"
#include "ramulator/controller/impl/atomic_controller_base.h"

namespace Ramulator {

// This entire adapter is trusted build input. Generated code implements only
// AtomicModel::Model in a separate library, not IController or this class.
class AtomicModelController final : public AtomicControllerBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IController, AtomicModelController, AtomicControllerBase, "Atomic")

 private:
  void init_model() override {
    // Only the trusted evaluator supplies this field. It is not a model
    // parameter and is never passed through Hardware or Parameters. SimpleO3's
    // isolated host installs the factory before construction instead.
    std::string library;
    RAMULATOR_PARSE_PARAM(library, std::string, "model_library").default_val("");
    if (!library.empty()) AtomicModel::load_library(library);
    const auto& spec = *m_device.m_spec;
    AtomicModel::Hardware hardware{
        spec.standard_name, spec.level_names, spec.organization.level_sizes, {},
        m_latency, m_tCK_ps, spec.get_tx_bytes(), spec.channel_width,
        spec.organization.dq, spec.internal_prefetch_size, m_read_buffer_size,
        m_write_buffer_size, m_wr_low_watermark, m_wr_high_watermark};
    for (const auto& [name, index] : spec.timings) hardware.timings.emplace(name, spec.timing_vals.at(index));
    AtomicModel::Parameters parameters(m_model_parameters);
    m_model = AtomicModel::create();
    m_model->initialize(std::move(hardware), parameters);
    parameters.require_all_used();
    // The base's historical parameter checker sees only overrides accepted by
    // the separate parameter object. No callback into the controller is given
    // to the model to implement parameter declarations.
    for (const auto& [name, value] : m_model_parameters) m_used_model_parameters.insert(name);
  }

  Clk_t predict_departure(const Request& request) override {
    return m_model->predict(
        AtomicModel::Access{request.addr,
            request.type_id == Request::Type::Read ? AtomicModel::AccessType::Read : AtomicModel::AccessType::Write,
            request.addr_vec, request.size_bytes}, m_clk);
  }

  std::unique_ptr<AtomicModel::Model> m_model;
};
}  // namespace Ramulator
