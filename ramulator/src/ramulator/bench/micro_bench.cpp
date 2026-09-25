/*
 * Micro-benchmark harness for profiling critical hot paths
 * Directly measures:
 * - FRFCFS scheduler double-scan overhead
 * - DRAMNode check_timing tree walk cost
 * - m_row_state map lookup overhead vs unordered_map
 * - Request buffer iteration (std::list vs std::deque)
 * - get_target_banks allocation cost
 */

#include <benchmark/benchmark.h>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <unordered_map>
#include <array>
#include <memory>
#include <algorithm>

// Simulate relevant data structures
struct Request {
    std::vector<int> addr_vec;  // This allocation is the target of A1
    int command = -1;
    int final_command = 0;
};

using ReqList = std::list<Request>;
using ReqDeque = std::deque<Request>;

// Benchmark 1: Iterate a buffer of requests
static void BM_iterate_list_of_requests(benchmark::State& state) {
    ReqList buffer;
    // Create 32 requests (typical buffer size)
    for (int i = 0; i < 32; ++i) {
        Request req;
        req.addr_vec.resize(7);  // DDR4 address levels
        buffer.push_back(req);
    }

    for (auto _ : state) {
        for (auto& req : buffer) {
            benchmark::DoNotOptimize(req.command);
        }
    }
    state.SetItemsProcessed(state.iterations() * 32);
}
BENCHMARK(BM_iterate_list_of_requests);

static void BM_iterate_deque_of_requests(benchmark::State& state) {
    ReqDeque buffer;
    for (int i = 0; i < 32; ++i) {
        Request req;
        req.addr_vec.resize(7);
        buffer.push_back(req);
    }

    for (auto _ : state) {
        for (auto& req : buffer) {
            benchmark::DoNotOptimize(req.command);
        }
    }
    state.SetItemsProcessed(state.iterations() * 32);
}
BENCHMARK(BM_iterate_deque_of_requests);

// Benchmark 2: Row state lookup (map vs unordered_map)
static void BM_row_state_map_lookup(benchmark::State& state) {
    std::map<int, int> row_state;
    // Populate with a few open rows (typical case: 0-1, worst case: many)
    for (int i = 0; i < 5; ++i) {
        row_state[i * 1000] = 1;  // Open
    }

    for (auto _ : state) {
        auto it = row_state.find(2000);  // Typical lookup
        benchmark::DoNotOptimize(it);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_row_state_map_lookup);

static void BM_row_state_unordered_map_lookup(benchmark::State& state) {
    std::unordered_map<int, int> row_state;
    for (int i = 0; i < 5; ++i) {
        row_state[i * 1000] = 1;
    }

    for (auto _ : state) {
        auto it = row_state.find(2000);
        benchmark::DoNotOptimize(it);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_row_state_unordered_map_lookup);

// Benchmark 3: get_target_banks allocation (returning vector by value)
static std::vector<int> get_target_banks_current(bool single, int bank_id) {
    if (single) {
        return {bank_id};  // Allocation here!
    }
    std::vector<int> result;
    for (int i = 0; i < 16; ++i) result.push_back(i);
    return result;
}

static void BM_get_target_banks_vector_return(benchmark::State& state) {
    for (auto _ : state) {
        auto banks = get_target_banks_current(true, 5);
        benchmark::DoNotOptimize(banks);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_get_target_banks_vector_return);

// Benchmark 4: Address vector copy in Request
static void BM_request_address_vec_copy(benchmark::State& state) {
    Request source;
    source.addr_vec.resize(7);
    for (int i = 0; i < 7; ++i) source.addr_vec[i] = i;

    for (auto _ : state) {
        Request dest = source;  // Copy constructor - this copies addr_vec
        benchmark::DoNotOptimize(dest);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_request_address_vec_copy);

// Benchmark 5: Two-pass scheduler pattern (O(N) + O(N))
static void BM_frfcfs_double_scan(benchmark::State& state) {
    ReqList buffer;
    for (int i = 0; i < 32; ++i) {
        Request req;
        req.addr_vec.resize(7);
        req.final_command = i % 4;
        buffer.push_back(req);
    }

    for (auto _ : state) {
        // Pass 1: update command for every entry
        for (auto& req : buffer) {
            req.command = (req.final_command + 1) % 4;  // Simulates get_preq_command
        }
        // Pass 2: find best
        auto best_it = buffer.begin();
        for (auto it = buffer.begin(); it != buffer.end(); ++it) {
            if (it->command > best_it->command) best_it = it;
        }
        benchmark::DoNotOptimize(*best_it);
    }
    state.SetItemsProcessed(state.iterations() * 64);  // Two passes
}
BENCHMARK(BM_frfcfs_double_scan);

// Benchmark 6: Write-forwarding check (std::find_if over list)
static void BM_write_forwarding_find(benchmark::State& state) {
    ReqList write_buffer;
    for (int i = 0; i < 16; ++i) {
        Request req;
        req.addr_vec.resize(7);
        req.addr_vec[0] = i;
        write_buffer.push_back(req);
    }

    for (auto _ : state) {
        auto it = std::find_if(
            write_buffer.begin(),
            write_buffer.end(),
            [](const Request& r) { return r.addr_vec[0] == 8; }
        );
        benchmark::DoNotOptimize(it);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_write_forwarding_find);

static void BM_write_forwarding_hash(benchmark::State& state) {
    std::unordered_set<int> write_addrs;
    for (int i = 0; i < 16; ++i) {
        write_addrs.insert(i);
    }

    for (auto _ : state) {
        auto found = write_addrs.count(8) > 0;
        benchmark::DoNotOptimize(found);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_write_forwarding_hash);

BENCHMARK_MAIN();
