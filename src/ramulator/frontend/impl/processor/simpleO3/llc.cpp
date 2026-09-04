#include "ramulator/frontend/impl/processor/simpleO3/llc.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <stdexcept>

namespace Ramulator {

size_t SimpleO3LLC::LogicalKeyHash::operator()(const LogicalKey& key) const {
  size_t seed = std::hash<int>{}(key.source_id);
  auto combine = [&seed](size_t value) { seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2); };
  combine(std::hash<std::int64_t>{}(key.frontend_id));
  combine(std::hash<std::int64_t>{}(key.frontend_sub_id));
  return seed;
}

SimpleO3LLC::SimpleO3LLC(const Clk_t& clk, int latency, int size_bytes, int linesize_bytes, int associativity,
                         int num_mshrs)
    : SimpleO3LLC(clk, latency, size_bytes, linesize_bytes, associativity, num_mshrs, "") {
}

SimpleO3LLC::SimpleO3LLC(const Clk_t& clk, int latency, int size_bytes, int linesize_bytes, int associativity,
                         int num_mshrs, const std::string& request_trace_path)
    : m_clk(clk),
      m_latency(latency),
      m_size_bytes(size_bytes),
      m_linesize_bytes(linesize_bytes),
      m_associativity(associativity),
      m_num_mshrs(num_mshrs) {
  m_logger = Logger("SimpleO3LLC");

  if (m_latency < 0) {
    throw std::runtime_error("SimpleO3 LLC latency must be non-negative");
  }
  if (size_bytes <= 0 || linesize_bytes <= 0 || associativity <= 0) {
    throw std::runtime_error("SimpleO3 LLC size, line size, and associativity must be positive");
  }
  if ((linesize_bytes & (linesize_bytes - 1)) != 0) {
    throw std::runtime_error("SimpleO3 LLC line size must be a power of two");
  }
  const std::int64_t bytes_per_set =
      static_cast<std::int64_t>(linesize_bytes) * static_cast<std::int64_t>(associativity);
  if (size_bytes < bytes_per_set || size_bytes % bytes_per_set != 0) {
    throw std::runtime_error("SimpleO3 LLC capacity must contain an integral number of complete sets");
  }
  const std::int64_t num_sets = size_bytes / bytes_per_set;
  if ((num_sets & (num_sets - 1)) != 0) {
    throw std::runtime_error("SimpleO3 LLC set count must be a power of two");
  }
  if (num_mshrs <= 0) {
    throw std::runtime_error("SimpleO3 LLC MSHR count must be positive");
  }

  m_set_size = static_cast<int>(num_sets);
  m_index_mask = m_set_size - 1;
  m_index_offset = calc_log2(m_linesize_bytes);
  m_tag_offset = calc_log2(m_set_size) + m_index_offset;

  if (!request_trace_path.empty()) {
    m_request_trace_file.open(request_trace_path);
    if (!m_request_trace_file.is_open()) {
      throw std::runtime_error("SimpleO3 could not open logical request trace: " + request_trace_path);
    }
    m_request_trace_file << "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal,llc_path\n";
  }

  DEBUG_LOG(m_logger, "Index mask: {0:x}", m_index_mask);
  DEBUG_LOG(m_logger, "Index offset: {}", m_index_offset);
  DEBUG_LOG(m_logger, "Tag offset: {}", m_tag_offset);
};

SimpleO3LLC::LogicalKey SimpleO3LLC::logical_key(const Request& req) const {
  return {req.source_id, req.frontend_id, req.frontend_sub_id};
}

void SimpleO3LLC::validate_logical_identity(const Request& req) const {
  if (req.source_id < 0 || req.frontend_id < 0 || req.frontend_sub_id < 0) {
    throw std::runtime_error("SimpleO3 LLC request is missing a valid stable frontend identity");
  }
  if (req.type_id != Request::Type::Read && req.type_id != Request::Type::Write) {
    throw std::runtime_error("SimpleO3 LLC received an unsupported logical request type");
  }
  if (m_live_logical_ids.contains(logical_key(req))) {
    throw std::runtime_error("SimpleO3 LLC received a duplicate live stable frontend identity");
  }
}

SimpleO3LLC::LogicalRequest SimpleO3LLC::begin_logical_request(const Request& req, int original_type,
                                                               LogicalPath path) {
  const auto key = logical_key(req);
  if (!m_live_logical_ids.insert(key).second) {
    throw std::runtime_error("SimpleO3 LLC failed to register a unique live frontend identity");
  }
  s_logical_requests_live++;
  s_logical_requests_peak = std::max(s_logical_requests_peak, s_logical_requests_live);
  switch (path) {
    case LogicalPath::Hit:
      s_logical_requests_hit++;
      break;
    case LogicalPath::MSHRMerge:
      s_logical_requests_mshr_merge++;
      break;
    case LogicalPath::MissOwner:
      s_logical_requests_miss_owner++;
      break;
  }
  return {req, m_clk, original_type, m_next_logical_admission_ordinal++, path};
}

void SimpleO3LLC::complete_logical_request(const LogicalRequest& logical, Clk_t depart) {
  if (depart < logical.arrive) {
    throw std::runtime_error("SimpleO3 logical request completed before it arrived at the LLC");
  }
  if (m_live_logical_ids.erase(logical_key(logical.req)) != 1) {
    throw std::runtime_error("SimpleO3 logical request completion has no unique live identity");
  }
  if (s_logical_requests_live <= 0) {
    throw std::runtime_error("SimpleO3 logical live-request count underflow");
  }
  s_logical_requests_live--;
  s_logical_requests_completed++;
  if (m_request_trace_file.is_open()) {
    m_request_trace_file << logical.arrive << ',' << depart << ',' << logical.original_type << ','
                         << logical.req.source_id << ',' << logical.req.addr << ',' << logical.req.frontend_id << ','
                         << logical.req.frontend_sub_id << ',' << logical.admission_ordinal << ','
                         << static_cast<int>(logical.path) << '\n';
  }
}

std::vector<SimpleO3LLC::LogicalRequest> SimpleO3LLC::take_receive_requests(Addr_t addr) {
  auto it = m_receive_requests.find(addr);
  if (it == m_receive_requests.end() || it->second.empty()) {
    throw std::runtime_error("SimpleO3 LLC completion has no logical request bucket");
  }
  auto requests = std::move(it->second);
  m_receive_requests.erase(it);
  return requests;
}

void SimpleO3LLC::tick() {
  // Send miss requests to the memory system when LLC latency is met
  // TODO: Optimization by assuming in-order issue?
  auto miss_it = m_miss_list.begin();
  while (miss_it != m_miss_list.end()) {
    if (m_clk >= miss_it->first) {
      if (!m_memory_system->send(miss_it->second)) {
        miss_it++;
      } else {
        miss_it = m_miss_list.erase(miss_it);
      }
    } else {
      miss_it++;
    }
  }

  // call hit request callback when LLC latency is met
  auto hit_it = m_hit_list.begin();
  while (hit_it != m_hit_list.end()) {
    if (m_clk >= hit_it->first) {
      LogicalRequest logical = std::move(hit_it->second);
      if (!m_hit_completion_callback) {
        throw std::runtime_error("SimpleO3 logical LLC hit is missing its frontend completion callback");
      }
      m_hit_completion_callback(logical);
      hit_it = m_hit_list.erase(hit_it);
    } else {
      hit_it++;
    }
  }
};

bool SimpleO3LLC::send(Request& req) {
  validate_logical_identity(req);
  const int original_type = req.type_id;
  CacheSet_t& set = get_set(req.addr);

  if (req.type_id == Request::Type::Read) {
    s_llc_read_access++;
  } else if (req.type_id == Request::Type::Write) {
    s_llc_write_access++;
  }

  if (auto line_it = check_set_hit(set, req.addr); line_it != set.end()) {
    // Hit in the set
    DEBUG_LOG(m_logger,
              "[Clk={}] Request Source: {}, Type: {}, Addr: {}, Index: {}, Tag: {}. Hit, will finish at Clk={}", m_clk,
              req.source_id, req.type_id, req.addr, get_index(req.addr), get_tag(req.addr), m_clk + m_latency);

    // Update the LRU status
    set.push_back({req.addr, get_tag(req.addr), line_it->dirty || (req.type_id == Request::Type::Write), true});
    set.erase(line_it);

    // Add to the hit list to callback when finished
    m_hit_list.push_back(
        std::make_pair(m_clk + m_latency, begin_logical_request(req, original_type, LogicalPath::Hit)));
    return true;
  } else {
    // Miss in the set
    DEBUG_LOG(m_logger, "[Clk={}] Request Source: {}, Type: {}, Addr: {}, Index: {}, Tag: {}. Miss.", m_clk,
              req.source_id, req.type_id, req.addr, get_index(req.addr), get_tag(req.addr));

    if (req.type_id == Request::Type::Read) {
      s_llc_read_misses++;
    } else if (req.type_id == Request::Type::Write) {
      s_llc_write_misses++;
    }

    bool dirty = (req.type_id == Request::Type::Write);
    if (req.type_id == Request::Type::Write) {
      req.type_id = Request::Type::Read;
    }

    // MSHR lookup
    auto mshr_it = check_mshr_hit(req.addr);
    if (mshr_it != m_mshrs.end()) {
      DEBUG_LOG(m_logger, "MSHR Hit.", m_clk);
      // Add new req to MSHR_requests
      auto bucket_it = m_receive_requests.find(mshr_it->first);
      if (bucket_it == m_receive_requests.end()) {
        throw std::runtime_error("SimpleO3 MSHR has no logical owner bucket");
      }
      bucket_it->second.push_back(begin_logical_request(req, original_type, LogicalPath::MSHRMerge));

      mshr_it->second->dirty = dirty || mshr_it->second->dirty;
      return true;
    }

    // MSHR miss
    // Check if there is available MSHR entry
    if (m_mshrs.size() == m_num_mshrs) {
      DEBUG_LOG(m_logger, "No MSHR entry available.", m_clk);
      s_llc_mshr_unavailable++;
      return false;
    }

    // Check if there is available cache line in the set
    bool line_available = false;
    if (set.size() < m_associativity) {
      line_available = true;
    } else {
      for (const auto& line : set) {
        if (line.ready) {
          line_available = true;
        }
      }
    }
    if (!line_available) {
      DEBUG_LOG(m_logger, "No cache line available in the set.", m_clk);
      return false;
    }

    // Allocate a new cache line
    auto newline_it = allocate_line(set, req.addr);
    if (newline_it == set.end()) {
      throw std::runtime_error("Failed to allocate new line when there is available entry.");
    }
    newline_it->dirty = dirty;

    // Add to MSHR entries
    m_mshrs.push_back(std::make_pair(req.addr, newline_it));
    // Add Request to MSHR_requests
    auto [bucket_it, inserted] = m_receive_requests.emplace(req.addr, std::vector<LogicalRequest>{});
    if (!inserted) {
      throw std::runtime_error("SimpleO3 new miss collided with a live completion bucket");
    }
    bucket_it->second.push_back(begin_logical_request(req, original_type, LogicalPath::MissOwner));

    // Add to the miss request list
    req.size_bytes = static_cast<int>(m_linesize_bytes);
    enqueue_miss(req);

    return true;
  }
};

void SimpleO3LLC::enqueue_miss(Request req) {
  const int tx_bytes = m_memory_system->get_tx_bytes();
  if (tx_bytes >= m_linesize_bytes) {
    m_miss_list.push_back(std::make_pair(m_clk + m_latency, req));
    return;
  }

  const int n = m_linesize_bytes / tx_bytes;
  const Addr_t base = align(req.addr);
  Request original = req;
  std::function<void(Request&)> sub_callback = nullptr;
  if (original.callback) {
    auto remaining = std::make_shared<int>(n);
    sub_callback = [remaining, original](Request&) mutable {
      if (--(*remaining) == 0) {
        Request completed = original;
        original.callback(completed);
      }
    };
  }
  for (int i = 0; i < n; i++) {
    Request sub = req;
    sub.addr = base + static_cast<Addr_t>(i) * tx_bytes;
    sub.size_bytes = tx_bytes;
    sub.frontend_sub_id = i;
    sub.callback = sub_callback;
    m_miss_list.push_back(std::make_pair(m_clk + m_latency, sub));
  }
}

void SimpleO3LLC::receive(Request& req) {
  auto it = std::find_if(m_mshrs.begin(), m_mshrs.end(),
                         [&req, this](MSHREntry_t mshr_entry) { return (align(mshr_entry.first) == align(req.addr)); });

  DEBUG_LOG(m_logger, "[Clk={}] Request {} received.", m_clk, req.addr);

  if (it == m_mshrs.end()) {
    throw std::runtime_error("SimpleO3 memory completion has no matching live MSHR");
  }
  it->second->ready = true;
  m_mshrs.erase(it);
};

bool SimpleO3LLC::is_quiescent() const {
  if (s_logical_requests_live != static_cast<std::int64_t>(m_live_logical_ids.size())) {
    throw std::runtime_error("SimpleO3 LLC live-request accounting is inconsistent");
  }
  if (m_receive_requests.empty() && m_hit_list.empty() && !m_live_logical_ids.empty()) {
    throw std::runtime_error("SimpleO3 LLC lost a live logical request");
  }
  if (s_internal_writebacks_live < 0 || s_internal_writebacks_completed > s_internal_writebacks_generated ||
      s_internal_writebacks_live + s_internal_writebacks_completed != s_internal_writebacks_generated) {
    throw std::runtime_error("SimpleO3 LLC internal-writeback accounting is inconsistent");
  }
  return m_miss_list.empty() && m_hit_list.empty() && m_mshrs.empty() && m_receive_requests.empty() &&
         m_live_logical_ids.empty() && s_internal_writebacks_live == 0;
}

void SimpleO3LLC::finalize() {
  if (!is_quiescent()) {
    throw std::runtime_error("SimpleO3 finalized with live logical LLC requests");
  }
  if (m_request_trace_file.is_open()) {
    m_request_trace_file.close();
  }
}

SimpleO3LLC::CacheSet_t& SimpleO3LLC::get_set(Addr_t addr) {
  int set_index = get_index(addr);
  if (m_cache_sets.find(set_index) == m_cache_sets.end()) {
    m_cache_sets.insert(make_pair(set_index, std::list<Line>()));
  }
  return m_cache_sets[set_index];
}

SimpleO3LLC::CacheSet_t::iterator SimpleO3LLC::allocate_line(CacheSet_t& set, Addr_t addr) {
  // Check if we need to evict any line
  if (need_eviction(set, addr)) {
    // Get a victim to evict
    auto victim = std::find_if(set.begin(), set.end(), [this](Line line) { return line.ready; });
    if (victim == set.end()) {
      return victim;  // doesn't exist a line that's already unlocked in each level
    }
    evict_line(set, victim);
  }

  // Allocate new cache line and return an iterator to it
  set.push_back({addr, get_tag(addr)});
  return --set.end();
}

bool SimpleO3LLC::need_eviction(const CacheSet_t& set, Addr_t addr) {
  if (std::find_if(set.begin(), set.end(), [addr, this](Line l) { return (get_tag(addr) == l.tag); }) != set.end()) {
    // Due to MSHR, the program can't reach here. Just for checking
    assert(false);
    return false;
  } else {
    if (set.size() < m_associativity) {
      return false;
    } else {
      return true;
    }
  }
}

void SimpleO3LLC::evict_line(CacheSet_t& set, CacheSet_t::iterator victim_it) {
  DEBUG_LOG(m_logger, "Evicting {}.", victim_it->addr);
  s_llc_eviction++;

  // Generate writeback request if victim line is dirty
  if (victim_it->dirty) {
    Request writeback_req(victim_it->addr, Request::Type::Write);
    writeback_req.size_bytes = static_cast<int>(m_linesize_bytes);
    s_internal_writebacks_generated++;
    s_internal_writebacks_live++;
    writeback_req.callback = [this](Request&) {
      if (s_internal_writebacks_live <= 0) {
        throw std::runtime_error("SimpleO3 LLC internal-writeback completion underflow");
      }
      s_internal_writebacks_live--;
      s_internal_writebacks_completed++;
    };
    enqueue_miss(writeback_req);

    DEBUG_LOG(m_logger, "Writeback Request will be issued at Clk={}.", m_clk + m_latency);
  }

  set.erase(victim_it);
}

SimpleO3LLC::CacheSet_t::iterator SimpleO3LLC::check_set_hit(CacheSet_t& set, Addr_t addr) {
  auto line_it = std::find_if(set.begin(), set.end(), [addr, this](Line l) { return (l.tag == get_tag(addr)); });
  if (line_it == set.end() || !line_it->ready) {
    return set.end();
  }
  return line_it;
}

SimpleO3LLC::MSHR_t::iterator SimpleO3LLC::check_mshr_hit(Addr_t addr) {
  auto mshr_it = std::find_if(m_mshrs.begin(), m_mshrs.end(), [addr, this](MSHREntry_t mshr_entry) {
    return (align(mshr_entry.first) == align(addr));
  });
  return mshr_it;
}

void SimpleO3LLC::serialize(std::string serialization_filename) {
  std::ofstream serialization_file;
  serialization_file.open(serialization_filename, std::ios::out);

  serialization_file << "index,addr,tag,dirty" << std::endl;
  for (auto it1 = m_cache_sets.begin(); it1 != m_cache_sets.end(); it1++) {
    for (auto it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
      serialization_file << it1->first << "," << it2->addr << "," << it2->tag << "," << it2->dirty << std::endl;
    }
  }
  serialization_file.close();
}

void SimpleO3LLC::deserialize(std::string serialization_filename) {
  std::ifstream serialization_file;
  serialization_file.open(serialization_filename, std::ios::in);

  std::string file_line;
  std::getline(serialization_file, file_line);  // Skip the first line, which is the header
  while (std::getline(serialization_file, file_line)) {
    std::string index_str = file_line.substr(0, file_line.find(","));
    file_line = file_line.substr(file_line.find(",") + 1);
    std::string addr_str = file_line.substr(0, file_line.find(","));
    file_line = file_line.substr(file_line.find(",") + 1);
    std::string tag_str = file_line.substr(0, file_line.find(","));
    file_line = file_line.substr(file_line.find(",") + 1);
    std::string dirty_str = file_line.substr(0, file_line.find(","));

    int index = std::stoi(index_str);
    Addr_t addr = std::stoll(addr_str);
    Addr_t tag = std::stoll(tag_str);
    bool dirty = std::stoi(dirty_str);
    if (m_cache_sets.find(index) == m_cache_sets.end()) {
      m_cache_sets.insert({index, std::list<SimpleO3LLC::Line>()});
    }
    m_cache_sets[index].push_back({addr, tag, dirty, 1});
  }
  serialization_file.close();
}

void SimpleO3LLC::dump_llc() {
  DEBUG_LOG(m_logger, "Dumping LLC");
  DEBUG_LOG(m_logger, "index,addr,tag,dirty,ready");
  for (auto it1 = m_cache_sets.begin(); it1 != m_cache_sets.end(); it1++) {
    for (auto it2 = it1->second.begin(); it2 != it1->second.end(); it2++) {
      DEBUG_LOG(m_logger, "{},{},{},{},{}", it1->first, it2->addr, it2->tag, it2->dirty, it2->ready);
    }
  }
}

}  // namespace Ramulator
