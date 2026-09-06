#include <deque>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"

namespace Ramulator {

/// Closed-loop parametric memory-pattern generator for controller response
/// characterization. N independent stream engines (composition axis), each
/// with a deterministic address sequence (all stochastic choices are
/// index-seeded, never timing-dependent) in its own address region, and
/// closed-loop timing (MLP cap + dependence + think time). Per stream:
/// spatial structure (row runs, bank spread), R/W structure (write fraction,
/// row relation, interleave granularity via write release batching), phase
/// structure (fixed or index-seeded random OFF gaps), and a two-regime
/// alternation (mode_switch: parameter set B every other block — regime
/// transitions). Reads match 1:1 across oracle/atomic runs per stream
/// (source_id = stream index) via occurrence-keyed matching between the
/// cycle-level oracle and the candidate model.
class SyntheticPattern : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, SyntheticPattern, "SyntheticPattern")

 private:
  struct Axes {
    int mlp = 16, think_time = 0, jitter = 0, row_run = 32, bank_spread = 1;
    bool bank_random = false, row_random = false;
    float dep_frac = 0.0f;
    int wfrac_pct = 0, wb_mode = 0, wb_batch = 1;
    int phase_on = 0, phase_off = 0, phase_off_random = 0;
    int pp_lead = 40;  // wb_mode 4: write to the alternate row issued this many cycles before each read group
  };

  struct Stream {
    int id = 0;
    Axes a, b;            // regime A / regime B (mode_switch alternates)
    int mode_switch = 0;  // reads per regime block; 0 = regime A only
    int num_requests = 0;
    uint64_t seed = 1;
    int bank_base = 0;    // stream's own bank window (disjoint per stream)
    int row_base = 0;     // stream's own row region

    int reads_issued = 0, outstanding = 0;
    bool dep_wait = false, retry_read = false, retry_write = false, pp_sent = false;
    Clk_t next_issue = 0, phase_resume = 0;
    std::deque<AddrVec_t> wb_queue;
    Request pending_read{{}, 0}, pending_write{{}, 0};
    size_t reads_done = 0, writes_sent = 0;
    int64_t total_lat = 0;

    const Axes& cur() const {
      if (mode_switch <= 0) return a;
      return ((reads_issued / mode_switch) % 2 == 0) ? a : b;
    }
    bool finished() const {
      return reads_issued >= num_requests && outstanding == 0 && wb_queue.empty() && !retry_write;
    }
  };

  // ── Config ──
  int m_streams = 1;
  int m_num_requests = 0;
  std::string m_stream_params;  // JSON-ish "k=v;k=v|k=v" per-stream overrides (see parse)
  Axes m_base;                  // stream-0 / default axes from flat params
  int m_p_mlp = 16;
  float m_p_dep_frac = 0.0f;
  int m_p_think_time = 0;
  int m_p_jitter = 0;
  int m_p_row_run = 32;
  int m_p_bank_spread = 1;
  bool m_p_bank_random = false;
  bool m_p_row_random = false;
  int m_p_wfrac_pct = 0;
  int m_p_wb_mode = 0;
  int m_p_wb_batch = 1;
  int m_p_phase_on = 0;
  int m_p_phase_off = 0;
  int m_p_phase_off_random = 0;
  int m_p_pp_lead = 40;
  int m_mode_switch = 0;
  std::string m_mode_b;         // "k=v;k=v" overrides for regime B
  uint64_t m_seed = 1;
  bool m_share_region = false;  // streams share the same bank/row region (aliasing/contention)

  // ── Address recipe (injected by Python from the DRAM object) ──
  int m_addr_vec_size;
  std::vector<int> m_bank_positions;
  std::vector<int> m_bank_counts;
  int m_total_bank_units;
  int m_row_pos, m_col_pos, m_num_rows, m_num_cols, m_internal_prefetch_size, m_num_cls;

  std::vector<Stream> m_s;
  size_t m_rr = 0;  // round-robin issue pointer

  // ── Stats ──
  size_t s_reads_sent = 0, s_writes_sent = 0;
  int64_t s_total_read_latency = 0;
  float s_avg_read_latency = 0.0f;
  size_t s_reads_done = 0;
  size_t s_writes_done = 0;

 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_num_requests, int, "num_requests").required();
    RAMULATOR_PARSE_PARAM(m_streams, int, "streams").default_val(1);
    RAMULATOR_PARSE_PARAM(m_stream_params, std::string, "stream_params").default_val("");
    RAMULATOR_PARSE_PARAM(m_mode_switch, int, "mode_switch").default_val(0);
    RAMULATOR_PARSE_PARAM(m_mode_b, std::string, "mode_b").default_val("");
    RAMULATOR_PARSE_PARAM(m_share_region, bool, "share_region").default_val(false);
    RAMULATOR_PARSE_PARAM(m_p_mlp, int, "mlp").default_val(16);
    RAMULATOR_PARSE_PARAM(m_p_dep_frac, float, "dep_frac").default_val(0.0f);
    RAMULATOR_PARSE_PARAM(m_p_think_time, int, "think_time").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_jitter, int, "jitter").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_row_run, int, "row_run").default_val(32);
    RAMULATOR_PARSE_PARAM(m_p_bank_spread, int, "bank_spread").default_val(1);
    RAMULATOR_PARSE_PARAM(m_p_bank_random, bool, "bank_random").default_val(false);
    RAMULATOR_PARSE_PARAM(m_p_row_random, bool, "row_random").default_val(false);
    RAMULATOR_PARSE_PARAM(m_p_wfrac_pct, int, "wfrac_pct").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_wb_mode, int, "wb_mode").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_wb_batch, int, "wb_batch").default_val(1);
    RAMULATOR_PARSE_PARAM(m_p_phase_on, int, "phase_on").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_phase_off, int, "phase_off").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_phase_off_random, int, "phase_off_random").default_val(0);
    RAMULATOR_PARSE_PARAM(m_p_pp_lead, int, "pp_lead").default_val(40);
    RAMULATOR_PARSE_PARAM(m_seed, uint64_t, "seed").default_val(1ULL);

    RAMULATOR_PARSE_PARAM(m_addr_vec_size, int, "addr_vec_size").required();
    RAMULATOR_PARSE_PARAM(m_total_bank_units, int, "total_bank_units").required();
    RAMULATOR_PARSE_PARAM(m_row_pos, int, "row_pos").required();
    RAMULATOR_PARSE_PARAM(m_col_pos, int, "col_pos").required();
    RAMULATOR_PARSE_PARAM(m_num_rows, int, "num_rows").required();
    RAMULATOR_PARSE_PARAM(m_num_cols, int, "num_cols").required();
    RAMULATOR_PARSE_PARAM(m_internal_prefetch_size, int, "internal_prefetch_size").required();
    RAMULATOR_PARSE_PARAM(m_num_cls, int, "num_cls").required();
    RAMULATOR_PARSE_PARAM(m_bank_positions, std::vector<int>, "bank_positions").required();
    RAMULATOR_PARSE_PARAM(m_bank_counts, std::vector<int>, "bank_counts").required();

    m_base.mlp = m_p_mlp;
    m_base.dep_frac = m_p_dep_frac;
    m_base.think_time = m_p_think_time;
    m_base.jitter = m_p_jitter;
    m_base.row_run = m_p_row_run;
    m_base.bank_spread = m_p_bank_spread;
    m_base.bank_random = m_p_bank_random;
    m_base.row_random = m_p_row_random;
    m_base.wfrac_pct = m_p_wfrac_pct;
    m_base.wb_mode = m_p_wb_mode;
    m_base.wb_batch = m_p_wb_batch;
    m_base.phase_on = m_p_phase_on;
    m_base.phase_off = m_p_phase_off;
    m_base.phase_off_random = m_p_phase_off_random;
    m_base.pp_lead = m_p_pp_lead;
    if (m_streams <= 0) {
      throw std::runtime_error("SyntheticPattern: streams must be >= 1");
    }
    // Per-stream overrides: "k=v;k=v|k=v;..." — segment i applies to stream i.
    std::vector<std::string> segs = split(m_stream_params, '|');
    const int banks_per_stream = m_share_region ? m_total_bank_units
                                                : std::max(m_total_bank_units / m_streams, 1);
    const int rows_per_stream = m_share_region ? m_num_rows / 2
                                               : std::max((m_num_rows / 2) / m_streams, 1);
    for (int i = 0; i < m_streams; i++) {
      Stream st;
      st.id = i;
      st.a = m_base;
      if (i < static_cast<int>(segs.size()) && !segs[i].empty()) {
        apply(st.a, segs[i]);
      }
      st.b = st.a;
      if (!m_mode_b.empty()) {
        apply(st.b, m_mode_b);
      }
      st.mode_switch = m_mode_switch;
      st.num_requests = m_num_requests;
      st.seed = m_seed + 7919ULL * static_cast<uint64_t>(i);
      st.bank_base = m_share_region ? 0 : i * banks_per_stream;
      st.row_base = m_share_region ? 0 : i * rows_per_stream;
      for (Axes* ax : {&st.a, &st.b}) {
        if (ax->bank_spread <= 0 || ax->bank_spread > banks_per_stream) {
          ax->bank_spread = banks_per_stream;
        }
        if (ax->wb_mode == 3 && ax->bank_spread * 2 > banks_per_stream) {
          throw std::runtime_error("SyntheticPattern: wb_mode=3 needs bank_spread <= half the stream's banks");
        }
        if (ax->row_run <= 0 || ax->row_run > m_num_cls) {
          throw std::runtime_error("SyntheticPattern: row_run must be in [1, num_cls]");
        }
      }
      m_s.push_back(st);
    }
    m_rows_per_stream = rows_per_stream;

    m_stats.add("cycles", m_clk);
    m_stats.add("reads_sent", s_reads_sent);
    m_stats.add("writes_sent", s_writes_sent);
    m_stats.add("reads_completed", s_reads_done);
    m_stats.add("writes_completed", s_writes_done);
    m_stats.add("total_read_latency", s_total_read_latency);
    m_stats.add("avg_read_latency", s_avg_read_latency);
  }

  int get_num_cores() override { return m_streams; }

  void tick() override {
    m_clk++;
    // One send attempt per tick, streams round-robin (fair arbitration).
    for (size_t k = 0; k < m_s.size(); k++) {
      Stream& st = m_s[(m_rr + k) % m_s.size()];
      if (try_stream(st)) {
        m_rr = (m_rr + k + 1) % m_s.size();
        return;
      }
    }
  }

  bool is_finished() override {
    for (const auto& st : m_s) {
      if (!st.finished()) return false;
    }
    // The last read callback can enqueue a writeback. Sending that write is
    // not its completion: keep ticking until every admitted callback returns.
    return s_writes_done == s_writes_sent;
  }

  void update_stats() override {
    if (s_reads_done > 0) {
      s_avg_read_latency = static_cast<float>(s_total_read_latency) / s_reads_done;
    }
  }

  void finalize() override { update_stats(); }

 private:
  int m_rows_per_stream = 1;

  // Returns true if a send was attempted (successful or backpressured).
  bool try_stream(Stream& st) {
    if (m_clk < st.phase_resume) {
      return false;
    }
    const Axes& ax = st.cur();
    const bool drain_tail = (st.reads_issued >= st.num_requests);
    if (st.retry_write || static_cast<int>(st.wb_queue.size()) >= ax.wb_batch ||
        (drain_tail && !st.wb_queue.empty())) {
      if (!st.retry_write) {
        st.pending_write = make_request(st.wb_queue.front(), Request::Type::Write, st.id);
        st.wb_queue.pop_front();
        st.retry_write = true;
      }
      if (m_memory_system->send(st.pending_write)) {
        st.writes_sent++;
        s_writes_sent++;
        st.retry_write = false;
      }
      return true;
    }
    if (drain_tail || st.outstanding >= ax.mlp || m_clk < st.next_issue) {
      return false;
    }
    if (ax.wb_mode == 4 && !st.pp_sent && st.reads_issued % ax.row_run == 0 &&
        st.reads_issued < st.num_requests) {
      // Ping-pong: the alternate-row write leads the read group by pp_lead.
      Request w = make_request(write_addr_vec(st, ax, st.reads_issued), Request::Type::Write, st.id);
      if (m_memory_system->send(w)) {
        st.writes_sent++;
        s_writes_sent++;
        st.pp_sent = true;
        st.next_issue = m_clk + ax.pp_lead;
      }
      return true;
    }
    if (st.dep_wait) {
      if (st.outstanding > 0) return false;
      st.dep_wait = false;
    }
    if (!st.retry_read) {
      const int i = st.reads_issued;
      st.pending_read = make_request(read_addr_vec(st, ax, i), Request::Type::Read, st.id);
      Stream* sp = &st;
      st.pending_read.callback = [this, sp](Request& completed) {
        sp->outstanding--;
        sp->reads_done++;
        s_reads_done++;
        const Clk_t lat = completed.depart - completed.arrive;
        sp->total_lat += lat;
        s_total_read_latency += lat;
        const Axes& cx = sp->cur();
        if (cx.wb_mode == 1 && want_writeback(*sp, cx, static_cast<int>(sp->reads_done))) {
          sp->wb_queue.push_back(completed.addr_vec);  // RMW: write back the same line
        }
      };
      st.retry_read = true;
    }
    if (m_memory_system->send(st.pending_read)) {
      const int i = st.reads_issued;
      st.retry_read = false;
      st.outstanding++;
      st.reads_issued++;
      s_reads_sent++;
      st.pp_sent = false;
      st.next_issue = m_clk + ax.think_time + jitter_of(st, ax, i);
      if (is_dependent(st, ax, i + 1)) {
        st.dep_wait = true;
      }
      if (ax.wb_mode >= 2 && want_writeback(st, ax, i)) {
        st.wb_queue.push_back(write_addr_vec(st, ax, i));
      }
      if (ax.phase_on > 0 && st.reads_issued % ax.phase_on == 0) {
        Clk_t off = ax.phase_off;
        if (ax.phase_off_random > 0) {
          off += static_cast<Clk_t>(hash_of(st, st.reads_issued / ax.phase_on, 6) %
                                    static_cast<uint64_t>(ax.phase_off_random + 1));
        }
        st.phase_resume = m_clk + off;
      }
    }
    return true;
  }

  static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
      if (c == sep) { out.push_back(cur); cur.clear(); } else { cur += c; }
    }
    if (!cur.empty() || !out.empty()) out.push_back(cur);
    return out;
  }
  static void apply(Axes& ax, const std::string& kv) {
    for (const auto& item : split(kv, ';')) {
      auto eq = item.find('=');
      if (eq == std::string::npos) continue;
      const std::string k = item.substr(0, eq), v = item.substr(eq + 1);
      if (k == "mlp") ax.mlp = std::stoi(v);
      else if (k == "think_time") ax.think_time = std::stoi(v);
      else if (k == "jitter") ax.jitter = std::stoi(v);
      else if (k == "row_run") ax.row_run = std::stoi(v);
      else if (k == "bank_spread") ax.bank_spread = std::stoi(v);
      else if (k == "bank_random") ax.bank_random = (std::stoi(v) != 0);
      else if (k == "row_random") ax.row_random = (std::stoi(v) != 0);
      else if (k == "dep_frac") ax.dep_frac = std::stof(v);
      else if (k == "wfrac_pct") ax.wfrac_pct = std::stoi(v);
      else if (k == "wb_mode") ax.wb_mode = std::stoi(v);
      else if (k == "wb_batch") ax.wb_batch = std::stoi(v);
      else if (k == "phase_on") ax.phase_on = std::stoi(v);
      else if (k == "phase_off") ax.phase_off = std::stoi(v);
      else if (k == "phase_off_random") ax.phase_off_random = std::stoi(v);
      else if (k == "pp_lead") ax.pp_lead = std::stoi(v);
      else throw std::runtime_error(fmt::format("SyntheticPattern: unknown axis '{}'", k));
    }
  }

  // Index-seeded hash → all stochastic axes are timing-invariant.
  static uint64_t hash_of(const Stream& st, int i, uint64_t salt) {
    uint64_t x = st.seed * 0x9E3779B97F4A7C15ULL + salt * 0xBF58476D1CE4E5B9ULL +
                 static_cast<uint64_t>(i) * 0x94D049BB133111EBULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
  }
  static bool is_dependent(const Stream& st, const Axes& ax, int i) {
    return ax.dep_frac > 0.0f &&
           (hash_of(st, i, 1) % 10000) < static_cast<uint64_t>(ax.dep_frac * 10000.0f);
  }
  static int jitter_of(const Stream& st, const Axes& ax, int i) {
    return ax.jitter > 0 ? static_cast<int>(hash_of(st, i, 2) % (ax.jitter + 1)) : 0;
  }
  static bool want_writeback(const Stream& st, const Axes& ax, int i) {
    return ax.wfrac_pct > 0 && (hash_of(st, i, 3) % 100) < static_cast<uint64_t>(ax.wfrac_pct);
  }

  AddrVec_t read_addr_vec(const Stream& st, const Axes& ax, int i) const {
    const int bank = ax.bank_random ? static_cast<int>(hash_of(st, i, 4) % ax.bank_spread)
                                    : (i % ax.bank_spread);
    const int p = i / ax.bank_spread;
    const int cl = p % ax.row_run;
    const int region = std::max(m_rows_per_stream / 2, 1);
    int row = ax.row_random
                  ? static_cast<int>(hash_of(st, p / ax.row_run * ax.bank_spread + bank, 5) % region)
                  : (p / ax.row_run) % region;
    return compose(st.bank_base + bank, st.row_base + row, cl);
  }
  AddrVec_t write_addr_vec(const Stream& st, const Axes& ax, int i) const {
    int bank = ax.bank_random ? static_cast<int>(hash_of(st, i, 4) % ax.bank_spread)
                              : (i % ax.bank_spread);
    if (ax.wb_mode == 3) bank += ax.bank_spread;
    const int p = i / ax.bank_spread;
    const int cl = p % ax.row_run;
    const int region = std::max(m_rows_per_stream / 2, 1);
    const int row = (p / ax.row_run) % region + region;  // disjoint half of the stream's region
    return compose(st.bank_base + bank, st.row_base + row, cl);
  }

  AddrVec_t compose(int flat_bank, int row, int cl) const {
    AddrVec_t av(m_addr_vec_size, 0);
    flat_bank %= m_total_bank_units;
    row %= m_num_rows;
    for (int i = static_cast<int>(m_bank_positions.size()) - 1; i >= 0; i--) {
      av[m_bank_positions[i]] = flat_bank % m_bank_counts[i];
      flat_bank /= m_bank_counts[i];
    }
    av[m_row_pos] = row;
    av[m_col_pos] = cl * m_internal_prefetch_size;
    return av;
  }

  Request make_request(const AddrVec_t& av, int type, int source_id) {
    Request req(av, type);
    int bank_flat = 0;
    for (size_t i = 0; i < m_bank_positions.size(); i++) {
      bank_flat = bank_flat * m_bank_counts[i] + av[m_bank_positions[i]];
    }
    const int cls = av[m_col_pos] / m_internal_prefetch_size;
    req.addr = static_cast<Addr_t>(static_cast<int64_t>(bank_flat) * m_num_rows * m_num_cls +
                                   static_cast<int64_t>(av[m_row_pos]) * m_num_cls + cls);
    req.source_id = source_id;
    req.size_bytes = m_memory_system->get_tx_bytes();
    if (type == Request::Type::Write) {
      // A coalesced write may call back synchronously inside send(). Counting
      // cumulative completions also handles that convention without underflow.
      req.callback = [this](Request&) { s_writes_done++; };
    }
    return req;
  }
};

}  // namespace Ramulator
