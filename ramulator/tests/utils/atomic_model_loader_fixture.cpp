// Deliberately invalid DSOs used only by the trusted-loader infrastructure test.
#include "ramulator/controller/atomic_model/api.h"

#include <exception>

using namespace Ramulator::AtomicModel;

#if defined(LOADER_THROWING_VERSION)
class ModuleError final : public std::exception {
 public:
  ~ModuleError() noexcept override;
  const char* what() const noexcept override;
};

// Keep the virtual functions in the module. The host must be able to inspect
// and destroy this exception after load_library has unwound.
ModuleError::~ModuleError() noexcept = default;
const char* ModuleError::what() const noexcept { return "module-defined version failure"; }
#endif

extern "C" uint32_t ramulator_atomic_model_api_version() {
#if defined(LOADER_THROWING_VERSION)
  throw ModuleError();
#elif defined(LOADER_WRONG_VERSION)
  return api_version + 1;
#else
  return api_version;
#endif
}

#if !defined(LOADER_MISSING_FACTORY)
extern "C" Model* ramulator_create_atomic_model() { return nullptr; }
#endif
