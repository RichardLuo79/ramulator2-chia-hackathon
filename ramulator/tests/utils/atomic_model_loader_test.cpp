// Infrastructure check, not a DRAM accuracy workload.
#include "ramulator/controller/atomic_model/loader.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace Ramulator::AtomicModel;

template <typename F> static void rejects(F test) {
  try { test(); } catch (const std::exception&) { return; }
  throw std::runtime_error("invalid model load was accepted");
}

int main(int argc, char** argv) {
  if (argc != 5) throw std::runtime_error("expected seed, missing, incompatible and throwing modules");
  rejects([] { create(); });
  rejects([] { load_library(""); });
  rejects([] { load_library("candidate.so"); });
  rejects([] { load_library("/dev/null"); });
  rejects([&] { load_library(std::filesystem::absolute(argv[0]).string()); });
  rejects([] { install_factory(nullptr); });
  rejects([&] { load_library(argv[2]); });
  rejects([&] { load_library(argv[3]); });
  bool caught_module_error = false;
  try {
    load_library(argv[4]);
  } catch (const std::exception& error) {
    if (std::string(error.what()) != "module-defined version failure") throw;
    caught_module_error = true;
  }
  if (!caught_module_error) throw std::runtime_error("module-defined exception was not preserved");
  // Rejected libraries must not install a factory or prevent a later valid load.
  rejects([] { create(); });
  load_library(argv[1]);
  load_library(argv[1]);
  rejects([&] { load_library(std::filesystem::absolute(argv[0]).string()); });
  auto model = create();
  Hardware hardware{};
  hardware.read_latency = 17;
  Parameters parameters({});
  model->initialize(std::move(hardware), parameters);
  if (model->predict(Access{}, 100) != 117) throw std::runtime_error("wrong model loaded");
  std::cout << "explicit model load, API validation and replacement checks passed\n";
}
