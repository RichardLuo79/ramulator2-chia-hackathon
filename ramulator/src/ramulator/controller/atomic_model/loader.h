#pragma once

#include <memory>
#include "ramulator/controller/atomic_model/api.h"

namespace Ramulator::AtomicModel {
// Trusted host API; not included in the model's source/headers grant.
void install_factory(void* library);
// External hosts use an explicit evaluator-owned path, never LD_PRELOAD.
// Loading the same library for another channel is idempotent; changing it is not.
void load_library(const std::string& path);
std::unique_ptr<Model> create();
}  // namespace Ramulator::AtomicModel
