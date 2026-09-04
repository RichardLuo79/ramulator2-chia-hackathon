#include "ramulator/frontend/impl/processor/simpleO3/core.h"

#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>

#include "ramulator/base/utils.h"
#include "ramulator/frontend/impl/processor/simpleO3/llc.h"

namespace Ramulator {

namespace fs = std::filesystem;

SimpleO3Core::Trace::Trace(std::string file_path_str) {
  fs::path trace_path(file_path_str);
  if (!fs::exists(trace_path)) {
    throw std::runtime_error(fmt::format("Trace {} does not exist!", file_path_str));
  }

  std::ifstream trace_file(trace_path);
  if (!trace_file.is_open()) {
    throw std::runtime_error(fmt::format("Trace {} cannot be opened!", file_path_str));
  }

  std::string line;
  size_t line_number = 0;
  while (std::getline(trace_file, line)) {
    line_number++;
    std::vector<std::string> tokens;
    tokenize(tokens, line, " ");

    int num_tokens = tokens.size();
    if (num_tokens != 2 && num_tokens != 3) {
      throw std::runtime_error(fmt::format("Trace {} format invalid!", file_path_str));
    }
    size_t parsed = 0;
    int bubble_count = 0;
    Addr_t load_addr = -1;
    try {
      bubble_count = std::stoi(tokens[0], &parsed, 10);
      if (parsed != tokens[0].size()) {
        throw std::invalid_argument("trailing characters");
      }
    } catch (const std::exception&) {
      throw std::runtime_error(
          fmt::format("Trace {}:{} has an invalid instruction bubble count", file_path_str, line_number));
    }
    try {
      const int base = tokens[1].size() > 2 && tokens[1][0] == '0' &&
                               (tokens[1][1] == 'x' || tokens[1][1] == 'X')
                           ? 16
                           : 10;
      load_addr = std::stoll(tokens[1], &parsed, base);
      if (parsed != tokens[1].size()) {
        throw std::invalid_argument("trailing characters");
      }
    } catch (const std::exception&) {
      throw std::runtime_error(fmt::format("Trace {}:{} has an invalid load address", file_path_str, line_number));
    }
    if (bubble_count < 0) {
      throw std::runtime_error(
          fmt::format("Trace {}:{} has a negative instruction bubble count", file_path_str, line_number));
    }
    if (load_addr < -1) {
      throw std::runtime_error(fmt::format("Trace {}:{} has an invalid load address", file_path_str, line_number));
    }

    bool has_store = num_tokens == 2 ? false : true;
    if (has_store) {
      Addr_t store_addr = -1;
      try {
        const int base = tokens[2].size() > 2 && tokens[2][0] == '0' &&
                                 (tokens[2][1] == 'x' || tokens[2][1] == 'X')
                             ? 16
                             : 10;
        store_addr = std::stoll(tokens[2], &parsed, base);
        if (parsed != tokens[2].size()) {
          throw std::invalid_argument("trailing characters");
        }
      } catch (const std::exception&) {
        throw std::runtime_error(fmt::format("Trace {}:{} has an invalid store address", file_path_str, line_number));
      }
      if (store_addr < -1) {
        throw std::runtime_error(fmt::format("Trace {}:{} has an invalid store address", file_path_str, line_number));
      }
      m_trace.push_back({bubble_count, load_addr, store_addr});
    } else {
      m_trace.push_back({bubble_count, load_addr, -1});
    }
    if (bubble_count == 0 && load_addr == -1) {
      throw std::runtime_error(
          fmt::format("Trace {}:{} contributes no instruction to the fixed ROI", file_path_str, line_number));
    }
  }

  trace_file.close();
  m_trace_length = m_trace.size();
  if (m_trace_length == 0) {
    throw std::runtime_error(fmt::format("Trace {} contains no instruction records", file_path_str));
  }
}

const SimpleO3Core::Trace::Inst& SimpleO3Core::Trace::get_next_inst() {
  const Inst& inst = m_trace[m_curr_trace_idx];
  m_curr_trace_idx = (m_curr_trace_idx + 1) % m_trace_length;
  return inst;
}

SimpleO3Core::InstWindow::InstWindow(int ipc, int depth)
    : m_ipc(ipc), m_depth(depth), m_ready_list(depth, false), m_addr_list(depth, -1), m_frontend_id_list(depth, -1){};

bool SimpleO3Core::InstWindow::is_full() const {
  return m_load == m_depth;
}

void SimpleO3Core::InstWindow::insert(bool ready, Addr_t addr, std::int64_t frontend_id) {
  m_ready_list.at(m_head_idx) = ready;
  m_addr_list.at(m_head_idx) = addr;
  m_frontend_id_list.at(m_head_idx) = frontend_id;

  m_head_idx = (m_head_idx + 1) % m_depth;
  m_load++;
}

int SimpleO3Core::InstWindow::retire() {
  if (m_load == 0) {
    return 0;
  }

  int num_retired = 0;
  while (m_load > 0 && num_retired < m_ipc) {
    if (!m_ready_list.at(m_tail_idx)) {
      break;
    }

    m_tail_idx = (m_tail_idx + 1) % m_depth;
    m_load--;
    num_retired++;
  }
  return num_retired;
}

bool SimpleO3Core::InstWindow::set_ready(std::int64_t frontend_id) {
  if (frontend_id < 0) {
    throw std::runtime_error("SimpleO3 completion is missing its stable frontend identity");
  }
  if (m_load == 0) {
    return false;
  }

  int index = m_tail_idx;
  for (int i = 0; i < m_load; i++) {
    if (m_frontend_id_list[index] == frontend_id) {
      m_ready_list[index] = true;
      return true;
    }
    index++;
    if (index == m_depth) {
      index = 0;
    }
  }
  return false;
}

SimpleO3Core::SimpleO3Core(const Clk_t& clk, int id, int ipc, int depth, size_t num_expected_insts,
                           std::string trace_path, ITranslation* translation, SimpleO3LLC* llc)
    : m_clk(clk),
      m_id(id),
      m_window(ipc, depth),
      m_trace(trace_path),
      m_num_expected_insts(num_expected_insts),
      m_translation(translation),
      m_llc(llc) {
  // Fetch the instructions and addresses for tick 0.
  fetch_next_trace_inst();
}

void SimpleO3Core::fetch_next_trace_inst() {
  if (m_roi_issue_complete) {
    throw std::runtime_error("SimpleO3 attempted to fetch beyond its fixed instruction ROI");
  }
  auto inst = m_trace.get_next_inst();
  m_num_bubbles = inst.bubble_count;
  m_load_addr = inst.load_addr;
  m_writeback_addr = inst.store_addr;
  m_load_frontend_id = (m_load_addr != -1) ? m_next_frontend_id++ : -1;
  m_writeback_frontend_id = (m_writeback_addr != -1) ? m_next_frontend_id++ : -1;
}

void SimpleO3Core::finish_issue_roi(bool keep_current_writeback) {
  if (s_insts_issued != m_num_expected_insts) {
    throw std::runtime_error("SimpleO3 fixed issue ROI ended at an inconsistent instruction count");
  }
  m_roi_issue_complete = true;
  m_issue_current_writeback_after_roi = keep_current_writeback && m_writeback_addr != -1;
  m_num_bubbles = 0;
  m_load_addr = -1;
  if (!m_issue_current_writeback_after_roi) {
    m_writeback_addr = -1;
  }
}

void SimpleO3Core::tick() {
  s_insts_retired += m_window.retire();
  if (m_crit_file.is_open()) {
    Addr_t b = m_window.blocked_addr();
    if (b != -1) {
      m_blocked_cycles[b]++;
    }
  }
  if (!reached_expected_num_insts) {
    if (s_insts_retired >= m_num_expected_insts) {
      reached_expected_num_insts = true;
      s_cycles_recorded = m_clk;
    }
  }

  // A write operand attached to the final in-ROI trace record is part of
  // that record even though SimpleO3 sends it one cycle after its load.  No
  // other trace work may be issued once the fixed instruction budget ends.
  if (m_roi_issue_complete) {
    if (!m_issue_current_writeback_after_roi) {
      return;
    }
    Request writeback_request(m_writeback_addr, Request::Type::Write, m_id, m_callback);
    writeback_request.frontend_id = m_writeback_frontend_id;
    if (!m_translation->translate(writeback_request)) {
      return;
    }
    if (!m_llc->send(writeback_request)) {
      return;
    }
    m_writeback_addr = -1;
    m_issue_current_writeback_after_roi = false;
    return;
  }

  // First, issue the non-memory instructions
  int num_inserted_insts = 0;
  while (m_num_bubbles > 0) {
    if (num_inserted_insts == m_window.m_ipc) {
      return;
    }
    if (m_window.is_full()) {
      return;
    };
    m_window.insert(true, -1);
    s_insts_issued++;
    num_inserted_insts++;
    m_num_bubbles--;
    if (s_insts_issued == m_num_expected_insts) {
      // The memory operands following these bubbles lie outside the fixed
      // instruction ROI and must not leak into the logical request cohort.
      finish_issue_roi(false);
      return;
    }
  }

  // Second, try to send the load to the LLC
  if (m_load_addr != -1) {
    if (num_inserted_insts == m_window.m_ipc) {
      return;
    }
    if (m_window.is_full()) {
      return;
    };

    Request load_request(m_load_addr, Request::Type::Read, m_id, m_callback);
    load_request.frontend_id = m_load_frontend_id;
    if (!m_translation->translate(load_request)) {
      return;
    };

    if (m_llc->send(load_request)) {
      m_window.insert(false, load_request.addr, load_request.frontend_id);
      s_insts_issued++;
      m_load_addr = -1;
      if (s_insts_issued == m_num_expected_insts) {
        finish_issue_roi(true);
        return;
      }
      if (m_writeback_addr != -1) {
        // If there is still writeback, return without getting the next trace line
        // The write back will be issued in the next cycle
        // TODO: Should we allow both load and writeback to issue at the same cycle?
        return;
      }
    } else {
      return;
    }
  }

  // Third, try to send the writeback to the LLC
  if (m_writeback_addr != -1) {
    Request writeback_request(m_writeback_addr, Request::Type::Write, m_id, m_callback);
    writeback_request.frontend_id = m_writeback_frontend_id;
    if (!m_translation->translate(writeback_request)) {
      return;
    };
    if (!m_llc->send(writeback_request)) {
      return;
    }
    m_writeback_addr = -1;
  }

  fetch_next_trace_inst();
}

void SimpleO3Core::receive(Request& req) {
  if (req.type_id != Request::Type::Read && req.type_id != Request::Type::Write) {
    throw std::runtime_error("SimpleO3 core received an unsupported logical request type");
  }
  if (req.type_id == Request::Type::Read && m_crit_file.is_open()) {
    auto it = m_blocked_cycles.find(req.addr);
    int blocked = (it != m_blocked_cycles.end()) ? it->second : 0;
    if (it != m_blocked_cycles.end()) {
      m_blocked_cycles.erase(it);
    }
    m_crit_file << req.addr << ',' << blocked << '\n';
  }
  if (req.type_id == Request::Type::Read && !m_window.set_ready(req.frontend_id)) {
    throw std::runtime_error("SimpleO3 read completion does not match an in-flight ROB identity");
  }

  if (req.type_id == Request::Type::Read && req.arrive != -1 && req.depart > m_last_mem_cycle) {
    if (!reached_expected_num_insts) {
      s_mem_access_cycles += (req.depart - std::max(m_last_mem_cycle, req.arrive));
      m_last_mem_cycle = req.depart;
    }
  }
}

}  // namespace Ramulator
