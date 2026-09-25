// Trusted runner: initialize the frontend before loading an untrusted Atomic DSO.
// Python and native entry points share the same interleave implementation.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/landlock.h>
#include "ramulator/base/config.h"
#include "ramulator/base/batch_simulation.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/simulation.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"
#ifdef RAMULATOR_CHIA_MODEL_API
#include "ramulator/controller/atomic_model/loader.h"
#endif

extern "C" {
void* seccomp_init(unsigned int);
int seccomp_syscall_resolve_name(const char*);
int seccomp_rule_add(void*, unsigned int, int, unsigned int, ...);
int seccomp_load(void*);
void seccomp_release(void*);
}

using namespace Ramulator;

static void isolate(const char* candidate, const char* output_dir) {
  if (syscall(SYS_landlock_create_ruleset, nullptr, 0, LANDLOCK_CREATE_RULESET_VERSION) < 3)
    throw std::runtime_error("Landlock ABI >= 3 required");
  landlock_ruleset_attr attr{};
  attr.handled_access_fs = (1ULL << 15) - 1;
  int fd = syscall(SYS_landlock_create_ruleset, &attr, sizeof(uint64_t), 0);
  if (fd < 0) throw std::runtime_error("Landlock create failed");
  auto allow = [&](const char* path, uint64_t mask) {
    int parent = open(path, O_PATH | O_CLOEXEC);
    landlock_path_beneath_attr rule{mask, parent};
    if (parent < 0 || syscall(SYS_landlock_add_rule, fd, LANDLOCK_RULE_PATH_BENEATH, &rule, 0))
      throw std::runtime_error("Landlock rule failed");
    close(parent);
  };
  // No trace inputs, source files, config, credentials, or result labels.
  allow(candidate, LANDLOCK_ACCESS_FS_READ_FILE);
  allow(output_dir, LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_MAKE_REG |
                    LANDLOCK_ACCESS_FS_TRUNCATE);
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) || syscall(SYS_landlock_restrict_self, fd, 0))
    throw std::runtime_error("Landlock restriction failed");
  close(fd);
  void* ctx = seccomp_init(0x7fff0000U);
  for (const char* name : {"socket", "connect", "ptrace", "process_vm_readv", "process_vm_writev",
                           "execve", "execveat", "fork", "vfork", "clone", "clone3",
                           "open_by_handle_at", "io_uring_setup", "bpf", "userfaultfd"}) {
    int number = seccomp_syscall_resolve_name(name);
    if (number >= 0 && seccomp_rule_add(ctx, 0x50000U | 1U, number, 0))
      throw std::runtime_error("seccomp rule failed");
  }
  if (seccomp_load(ctx)) throw std::runtime_error("seccomp load failed");
  seccomp_release(ctx);
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: isolated_sim config candidate-or-dash output_dir stats");
    auto config = Config::parse_config_file(argv[1]);
    std::vector<BatchRequest> batch;
    const auto loading_start = std::chrono::steady_clock::now();
    const bool compact_input = static_cast<bool>(config["batch_speed_input"]);
    const bool replay = static_cast<bool>(config["batch_trace"]) || compact_input;
    if (compact_input && config["batch_trace"])
      throw std::invalid_argument("choose one batch input format");
    if (replay) {
      std::ifstream input(config[compact_input ? "batch_speed_input" : "batch_trace"].as<std::string>(),
                          std::ios::binary);
      batch = compact_input ? read_speed_batch(input, config["batch_interval"].as<Clk_t>()) : read_batch(input);
    }
    const auto loading_end = std::chrono::steady_clock::now();
    std::ofstream stats(argv[4]);
    if (!stats) throw std::runtime_error("cannot open stats output");
    stats << std::setprecision(17);
    auto start = std::chrono::steady_clock::now();
    std::unique_ptr<IFrontEnd> frontend(Factory::create_frontend(config));
    if (std::string(argv[2]) != "-") {
      isolate(argv[2], argv[3]);
      void* library = dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL);
      if (!library) throw std::runtime_error(dlerror());
#ifdef RAMULATOR_CHIA_MODEL_API
      AtomicModel::install_factory(library);
#endif
    }
    std::unique_ptr<IMemorySystem> memory(Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
    auto sim_start = std::chrono::steady_clock::now();
    if (replay) {
      BatchState state;
      BatchOptions options;
      const auto settings = config["batch_options"];
      if (settings) {
        if (settings["record_requests"]) options.record_requests = settings["record_requests"].as<bool>();
        if (settings["skip_idle_retries"]) options.skip_idle_retries = settings["skip_idle_retries"].as<bool>();
        if (settings["tick_only"]) options.tick_only = settings["tick_only"].as<bool>();
        if (settings["warmup_requests"]) options.warmup_requests = settings["warmup_requests"].as<size_t>();
      }
      auto result = run_batch(*memory, batch, state, options);
      if (options.record_requests) {
        std::ofstream observations(std::string(argv[3]) + "/replay.csv");
        write_batch(observations, batch, result, 0);
      }
      stats << "batch:\n  cycles: " << result.elapsed
            << "\n  reads_completed: " << result.completed_reads
            << "\n  writes_completed: " << result.completed_writes
            << "\n  measured_reads: " << result.measured_reads
            << "\n  measured_writes: " << result.measured_writes
            << "\n  measured_reads_completed: " << result.measured_reads_completed
            << "\n  measured_writes_completed: " << result.measured_writes_completed
            << "\n  warmup_reads_outstanding: " << result.warmup_reads_outstanding
            << "\n  warmup_writes_outstanding: " << result.warmup_writes_outstanding
            << "\n  measurement_start_cycle: " << result.measurement_start_cycle
            << "\n  final_admission_cycle: " << result.final_admission_cycle
            << "\n  measured_admission_wait_sum: " << result.measured_admission_wait_sum
            << "\n  measured_admission_wait_max: " << result.measured_admission_wait_max
            << "\n  admission_digest: " << result.admission_digest
            << "\n  completion_digest: " << result.completion_digest
            << "\n  warmup_wall_s: " << result.warmup_wall_s
            << "\n  measured_wall_s: " << result.measured_wall_s
            << "\n  drain_wall_s: " << result.drain_wall_s
            << "\n  warmup_cpu_s: " << result.warmup_cpu_s
            << "\n  measured_cpu_s: " << result.measured_cpu_s
            << "\n  drain_cpu_s: " << result.drain_cpu_s << '\n';
    } else {
      run_simulation(*frontend, *memory);
    }
    const auto sim_end = std::chrono::steady_clock::now();
    frontend->update_stats_recursive();
    memory->update_stats_recursive();
    frontend->print_stats(stats);
    memory->print_stats(stats);
    stats << "simulation_wall_s: " << std::chrono::duration<double>(sim_end - sim_start).count() << '\n';
    stats << "input_loading_wall_s: " << std::chrono::duration<double>(loading_end - loading_start).count() << '\n';
    stats << "construction_and_simulation_wall_s: " << std::chrono::duration<double>(sim_end - start).count() << '\n';
    frontend->finalize();
    memory->finalize();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
