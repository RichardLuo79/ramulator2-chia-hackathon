#ifndef RAMULATOR_FRONTEND_PROCESSOR_SIMPLEO3_LLC_H
#define RAMULATOR_FRONTEND_PROCESSOR_SIMPLEO3_LLC_H

#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ramulator/base/debug.h"
#include "ramulator/base/request.h"
#include "ramulator/base/type.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace Ramulator {

class SimpleO3LLC {
  friend class SimpleO3;
  enum class LogicalPath : int {
    Hit = 0,
    MSHRMerge = 1,
    MissOwner = 2,
  };

  struct LogicalKey {
    int source_id;
    std::int64_t frontend_id;
    std::int64_t frontend_sub_id;

    bool operator==(const LogicalKey& other) const {
      return source_id == other.source_id && frontend_id == other.frontend_id &&
             frontend_sub_id == other.frontend_sub_id;
    }
  };

  struct LogicalKeyHash {
    size_t operator()(const LogicalKey& key) const;
  };

  struct LogicalRequest {
    Request req;
    Clk_t arrive;
    int original_type;
    std::int64_t admission_ordinal;
    LogicalPath path;
  };

  struct Line {
    Addr_t addr = -1;
    Addr_t tag = -1;
    bool dirty = false;
    bool ready = false;  // Whether this line is ready (i.e., is still inflight?)
  };

 private:
  const Clk_t& m_clk;
  using CacheSet_t = std::list<Line>;  // LRU queue for the set. The head of the list is the least-recently-used way.
  std::unordered_map<int, CacheSet_t> m_cache_sets;

  using MSHREntry_t = std::pair<Addr_t, CacheSet_t::iterator>;
  using MSHR_t = std::vector<MSHREntry_t>;
  MSHR_t m_mshrs;
  std::unordered_map<Addr_t, std::vector<LogicalRequest>> m_receive_requests;
  std::unordered_set<LogicalKey, LogicalKeyHash> m_live_logical_ids;

  // Request that miss in the LLC with the clock cycle (current cycle + llc latency) that they
  // should be sent to the memory system
  std::list<std::pair<Clk_t, Request>> m_miss_list;

  // Enqueue a full-line memory request, splitting it into linesize/tx_bytes
  // transactions when the memory transaction is smaller than the cache line
  // (e.g. 64B lines over 32B-tx LPDDR/HBM). The original completion fires
  // once, when the last split transaction returns.
  void enqueue_miss(Request req);

  // Request that hit in the LLC with the clock cycle (current cycle + llc latency) that they
  // should be sent back to the core (calls the callback)
  std::list<std::pair<Clk_t, LogicalRequest>> m_hit_list;

  std::ofstream m_request_trace_file;
  std::function<void(const LogicalRequest&)> m_hit_completion_callback;
  std::int64_t m_next_logical_admission_ordinal = 0;

  IMemorySystem* m_memory_system;

  Logger m_logger;

  int m_latency;

  size_t m_size_bytes;
  size_t m_linesize_bytes;
  int m_associativity;
  int m_set_size;
  int m_num_mshrs;

  Addr_t m_index_mask;
  int m_index_offset;
  int m_tag_offset;

  int s_llc_read_access = 0;
  int s_llc_write_access = 0;
  int s_llc_read_misses = 0;
  int s_llc_write_misses = 0;
  int s_llc_eviction = 0;
  int s_llc_mshr_unavailable = 0;
  std::int64_t s_logical_requests_completed = 0;
  std::int64_t s_logical_requests_live = 0;
  std::int64_t s_logical_requests_peak = 0;
  std::int64_t s_logical_requests_hit = 0;
  std::int64_t s_logical_requests_mshr_merge = 0;
  std::int64_t s_logical_requests_miss_owner = 0;
  std::int64_t s_internal_writebacks_generated = 0;
  std::int64_t s_internal_writebacks_completed = 0;
  std::int64_t s_internal_writebacks_live = 0;

 public:
  SimpleO3LLC(const Clk_t& clk, int latency, int size_bytes, int linesize_bytes, int associativity, int num_mshrs);
  SimpleO3LLC(const Clk_t& clk, int latency, int size_bytes, int linesize_bytes, int associativity, int num_mshrs,
              const std::string& request_trace_path);
  void connect_memory_system(IMemorySystem* memory_system) {
    m_memory_system = memory_system;
  };

  void tick();

  // Tick-elision support: earliest cycle at which tick() would do work.
  // 0 = busy now (a memory-rejected miss is retrying every cycle).
  Clk_t next_event() const {
    Clk_t next = std::numeric_limits<Clk_t>::max();
    for (const auto& e : m_miss_list) {
      if (e.first <= m_clk) {
        return 0;
      }
      next = std::min(next, e.first);
    }
    for (const auto& e : m_hit_list) {
      if (e.first <= m_clk) {
        return 0;
      }
      next = std::min(next, e.first);
    }
    return next;
  }
  bool send(Request& req);
  void receive(Request& req);
  bool is_quiescent() const;
  void finalize();

  void serialize(std::string serialization_filename);
  void deserialize(std::string serialization_filename);
  void dump_llc();

 private:
  int get_index(Addr_t addr) {
    return (addr >> m_index_offset) & m_index_mask;
  };
  Addr_t get_tag(Addr_t addr) {
    return (addr >> m_tag_offset);
  };
  Addr_t align(Addr_t addr) {
    return (addr & ~(m_linesize_bytes - 1l));
  };

  CacheSet_t& get_set(Addr_t addr);
  CacheSet_t::iterator allocate_line(CacheSet_t& set, Addr_t addr);
  bool need_eviction(const CacheSet_t& set, Addr_t addr);
  void evict_line(CacheSet_t& set, CacheSet_t::iterator victim_it);

  CacheSet_t::iterator check_set_hit(CacheSet_t& set, Addr_t addr);
  MSHR_t::iterator check_mshr_hit(Addr_t addr);

  LogicalKey logical_key(const Request& req) const;
  void validate_logical_identity(const Request& req) const;
  LogicalRequest begin_logical_request(const Request& req, int original_type, LogicalPath path);
  void complete_logical_request(const LogicalRequest& logical, Clk_t depart);
  std::vector<LogicalRequest> take_receive_requests(Addr_t addr);
};

}  // namespace Ramulator

#endif  // RAMULATOR_FRONTEND_PROCESSOR_SIMPLEO3_LLC_H
