// Trusted runner: initialize the frontend before loading an untrusted Atomic DSO.
// The interleave body is extracted verbatim from the pinned Python binding by
// prepare_runtime(), not reimplemented or optimized here.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/landlock.h>
#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

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

static void run(IFrontEnd* m_frontend, IMemorySystem* m_memory_system) {
  // CHIA_INTERLEAVE_BODY
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: isolated_sim config candidate-or-dash output_dir stats");
    auto config = Config::parse_config_file(argv[1]);
    std::ofstream stats(argv[4]);
    if (!stats) throw std::runtime_error("cannot open stats output");
    auto start = std::chrono::steady_clock::now();
    std::unique_ptr<IFrontEnd> frontend(Factory::create_frontend(config));
    if (std::string(argv[2]) != "-") {
      isolate(argv[2], argv[3]);
      if (!dlopen(argv[2], RTLD_NOW | RTLD_GLOBAL))
        throw std::runtime_error(dlerror());
    }
    std::unique_ptr<IMemorySystem> memory(Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
    auto sim_start = std::chrono::steady_clock::now();
    run(frontend.get(), memory.get());
    const auto sim_end = std::chrono::steady_clock::now();
    frontend->update_stats_recursive();
    memory->update_stats_recursive();
    frontend->print_stats(stats);
    memory->print_stats(stats);
    stats << "simulation_wall_s: " << std::chrono::duration<double>(sim_end - sim_start).count() << '\n';
    stats << "construction_and_simulation_wall_s: " << std::chrono::duration<double>(sim_end - start).count() << '\n';
    frontend->finalize();
    memory->finalize();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
