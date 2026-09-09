#include "ramulator/controller/atomic_model/loader.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

namespace Ramulator::AtomicModel {
namespace {
CreateFunction factory = nullptr;
std::string loaded_path;
}

void install_factory(void* library) {
  if (factory) throw std::logic_error("An atomic model is already installed in this process");
  if (!library) throw std::invalid_argument("Cannot install a null atomic model library");
  auto version = reinterpret_cast<VersionFunction>(dlsym(library, "ramulator_atomic_model_api_version"));
  auto create_model = reinterpret_cast<CreateFunction>(dlsym(library, "ramulator_create_atomic_model"));
  if (!version || !create_model || version() != api_version) {
    throw std::runtime_error("Atomic candidate has missing or incompatible model-API entry points");
  }
  factory = create_model;
}

void load_library(const std::string& path) {
  if (!std::filesystem::path(path).is_absolute() || !std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument("Atomic model_library must name an absolute regular file");
  }
  const std::string canonical = std::filesystem::canonical(path).string();
  if (factory) {
    if (loaded_path == canonical) return;
    throw std::logic_error("Cannot replace the atomic model installed in this process");
  }
  void* library = dlopen(canonical.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(std::string("Cannot load atomic model: ") + dlerror());
  // Keep even a rejected library resident until process exit. Its version
  // entry point can throw a module-defined exception: unloading here would
  // unmap that exception's what(), destructor or type information while it is
  // still unwinding. Successful model objects also need their module's vtables.
  install_factory(library);
  loaded_path = canonical;
}

std::unique_ptr<Model> create() {
  if (!factory) throw std::logic_error("No atomic model library was installed by the trusted host");
  auto result = std::unique_ptr<Model>(factory());
  if (!result) throw std::runtime_error("Atomic candidate factory returned null");
  return result;
}
}  // namespace Ramulator::AtomicModel
