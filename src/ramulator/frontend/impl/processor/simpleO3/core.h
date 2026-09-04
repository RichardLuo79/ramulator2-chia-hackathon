#ifndef RAMULATOR_FRONTEND_PROCESSOR_CORE_H
#define RAMULATOR_FRONTEND_PROCESSOR_CORE_H

#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ramulator/base/request.h"
#include "ramulator/base/type.h"
#include "ramulator/translation/i_translation.h"

namespace Ramulator {

class SimpleO3LLC;

class SimpleO3Core {
  friend class SimpleO3;
  class Trace {
    friend class SimpleO3Core;
    struct Inst {
      int bubble_count = 0;
      Addr_t load_addr = -1;
      Addr_t store_addr = -1;
    };

    std::vector<Inst> m_trace;
    size_t m_trace_length = 0;
    size_t m_curr_trace_idx = 0;

   public:
    Trace(std::string file_path_str);
    const Inst& get_next_inst();
  };

  /**
   * @brief   Simplified ROB of an O3 processor.
   * @details
   * We model
   */
  class InstWindow {
    friend class SimpleO3Core;

   private:
    int m_ipc = 4;      // How many instructions we can retire in a cycle
    int m_depth = 128;  // How many inflight instructions we can keep track of

    int m_load = 0;      // The current load
    int m_head_idx = 0;  // Head index. New instructions are inserted at the head index.
    int m_tail_idx = 0;  // Tail index. The instruction at the tail will be retired first.

    std::vector<bool> m_ready_list;  // Bitvector to mark whether each instruction is ready to be retired.
    std::vector<Addr_t>
        m_addr_list;  // Which address is each LD/ST instruction targeting? TODO: Perf. optimization with unordered map?
    // Readiness must use the stable logical request identity, not address:
    // multiple same-address LLC hits can complete on different cycles.
    std::vector<std::int64_t> m_frontend_id_list;

   public:
    InstWindow(int ipc = 4, int depth = 128);
    // The load the commit head is blocked on this cycle (-1 if none):
    // criticality attribution for per-request validation metrics.
    Addr_t blocked_addr() const {
      return (m_load > 0 && !m_ready_list[m_tail_idx]) ? m_addr_list[m_tail_idx] : -1;
    }

    bool is_full() const;
    bool tail_ready() const {
      return m_load > 0 && m_ready_list[m_tail_idx];
    }

    /**
     * @brief   Inserts an instruction to the window.
     *
     * @param ready True if instruction is a non-memory instruction. False otherwise.
     * @param addr  -1 if non-memory, the actual LD/ST address otherwise.
     */
    void insert(bool ready, Addr_t addr, std::int64_t frontend_id = -1);

    /**
     * @brief   Tries to retire instructions from the tail of the window
     *
     * @return int The number of instructions retired.
     */
    int retire();

    /**
     * @brief   Set a memory instruction to ready. Called by the callback when a request is served by the memory
     *
     */
    bool set_ready(std::int64_t frontend_id);
  };

 private:
  const Clk_t& m_clk;
  int m_id = -1;

  Trace m_trace;
  InstWindow m_window;
  // Criticality attribution (blocked-on-load cycles per address, flushed
  // per completion in occurrence order); enabled via crit_trace_path.
  std::ofstream m_crit_file;
  std::unordered_map<Addr_t, int> m_blocked_cycles;
  ITranslation* m_translation;
  SimpleO3LLC* m_llc;

  std::function<void(Request&)> m_callback;

  int m_num_bubbles = 0;
  Addr_t m_load_addr = -1;
  Addr_t m_writeback_addr = -1;
  std::int64_t m_load_frontend_id = -1;
  std::int64_t m_writeback_frontend_id = -1;
  std::int64_t m_next_frontend_id = 0;
  bool m_roi_issue_complete = false;
  bool m_issue_current_writeback_after_roi = false;

  // Fetch one trace instance and assign request identities once. These IDs
  // remain attached to the pending operations across all retry cycles.
  void fetch_next_trace_inst();
  void finish_issue_roi(bool keep_current_writeback);

  size_t m_num_expected_insts = 0;
  Clk_t m_last_mem_cycle = 0;  // The last cycle that a memory request departs from mc

  /************************************************
   *              Core Statistics
   ***********************************************/
 public:
  bool reached_expected_num_insts = false;
  size_t s_insts_issued = 0;
  size_t s_insts_retired = 0;
  size_t s_cycles_recorded = 0;
  Clk_t s_mem_access_cycles = 0;

 public:
  SimpleO3Core(const Clk_t& clk, int id, int ipc, int depth, size_t num_expected_insts, std::string trace_path,
               ITranslation* translation, SimpleO3LLC* llc);

  /**
   * @brief   Ticks the core.
   *
   */
  void tick();

  bool issue_roi_quiescent() const {
    return m_roi_issue_complete && !m_issue_current_writeback_after_roi;
  }

  // Tick-elision support. A stalled core's tick is a pure blocked-cycle
  // count: retirement, issue, and trace advance are all unreachable. A core
  // whose fixed ROI has fully retired is also inert while another core drains.
  bool is_stalled() const {
    // Once the fixed issue ROI is complete, a not-ready tail is a pure stall
    // even when the ROB is not full.  The sole exception is the write operand
    // attached to the last admitted trace record, which must still be sent.
    if (m_roi_issue_complete) {
      if (m_issue_current_writeback_after_roi || m_window.tail_ready()) {
        return false;
      }
      return true;
    }
    if (!m_window.is_full() || m_window.tail_ready()) {
      return false;
    }
    // A pending writeback bypasses the window-full checks in tick() (the
    // bubble and load branches return first; the writeback branch sends and
    // advances the trace regardless of window state).
    if (m_num_bubbles == 0 && m_load_addr == -1 && m_writeback_addr != -1) {
      return false;
    }
    return true;
  }
  void fast_forward(Clk_t ticks) {
    if (m_crit_file.is_open()) {
      Addr_t b = m_window.blocked_addr();
      if (b != -1) {
        m_blocked_cycles[b] += ticks;
      }
    }
  }
  void open_crit_trace(const std::string& path) {
    m_crit_file.open(path);
  }

  /**
   * @brief   Called when a request is served by the memory.
   *
   */
  void receive(Request& req);
};

}  // namespace Ramulator

#endif  // RAMULATOR_FRONTEND_PROCESSOR_CORE_H
