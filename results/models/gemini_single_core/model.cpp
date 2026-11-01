#include "ramulator/controller/atomic_model/api.h"
#include <algorithm>
#include <vector>

namespace {

struct BusBurst {
  Ramulator::AtomicModel::Cycle depart;
  int bank_id;
  int bg;
  bool is_write;
};

struct QueuedWrite {
  int bank_id;
  int bg;
  int row;
  Ramulator::AtomicModel::Cycle arrive;
};

// Generic immediate-response DRAM timing model for DDR5 memory subsystems.
// Accounts for:
// - Physical 32-bank open-page state and row conflicts
// - Data bus burst arbitration with complete 4-way turnarounds (CCDS, CCDL, RTW, WTR)
// - Direction-aware bus turnaround scheduling (gap_to_b vs gap_from_b)
// - Bounded high-watermark write drain modeling with calibrated physical drain cycles
// - Opportunistic write execution during bank/controller idle gaps
// - Multi-level hierarchical queue congestion arbitration:
//   * Global CA bus serialization (linear)
//   * Global M/G/1 queue depth backpressure (quadratic)
//   * BankGroup-specific structural clustering contention (quadratic)
//   * Bank-specific structural queue serialization (linear)
class GenericDRAMModel final : public Ramulator::AtomicModel::Model {
 public:
  void initialize(Ramulator::AtomicModel::Hardware hw,
                  Ramulator::AtomicModel::Parameters& params) override {
    hardware = hw;
    num_channels = hw.level_sizes[hw.level_index("Channel")];
    num_ranks = hw.level_sizes[hw.level_index("Rank")];
    num_bankgroups = hw.level_sizes[hw.level_index("BankGroup")];
    num_banks_per_group = hw.level_sizes[hw.level_index("Bank")];
    total_banks = num_channels * num_ranks * num_bankgroups * num_banks_per_group;

    t_hit = hw.read_latency + 1;

    nRCD = hw.timing("nRCD");
    nRP = hw.timing("nRP");
    nRAS = hw.timing("nRAS");
    nRC = hw.timing("nRC");
    nRTP = hw.timing("nRTP");
    nCCDS = hw.timing("nCCDS");
    nCCDL = hw.timing("nCCDL");

    auto get_t = [&](const std::string& name, Ramulator::AtomicModel::Cycle fallback) {
      auto it = hw.timings.find(name);
      return it != hw.timings.end() ? it->second : fallback;
    };
    nRTW = get_t("nRTW", 14);
    nWTRL = get_t("nWTRL", 24);
    nWTRS = get_t("nWTRS", 10);
    nWR = get_t("nWR", 48);
    nCWL = get_t("nCWL", 32);
    nBL = get_t("nBL", 8);

    t_write_hit = nCWL + nBL; // 40 cycles

    write_high_wm = static_cast<int>(hw.write_high_watermark * hw.write_capacity);
    write_low_wm = static_cast<int>(hw.write_low_watermark * hw.write_capacity);
    if (write_high_wm <= 0) write_high_wm = 51;
    if (write_low_wm <= 0) write_low_wm = 32;

    int writes_to_drain = write_high_wm - write_low_wm;
    double drain_cycles_per_write = params.number("drain_cycles_per_write", 11.0, 8.0, 20.0);
    drain_duration = nRTW + static_cast<Ramulator::AtomicModel::Cycle>(writes_to_drain * drain_cycles_per_write) + nWTRL;

    ca_linear_delay = params.number("ca_linear_delay", 0.50, 0.0, 5.0);
    ca_quad_delay = params.number("ca_quad_delay", 0.02, 0.0, 0.5);
    ca_bg_delay = params.number("ca_bg_delay", 0.15, 0.0, 1.0);
    ca_bank_delay = params.number("ca_bank_delay", 1.2, 0.0, 20.0);
    write_congestion_weight = params.number("write_congestion_weight", 0.75, 0.0, 2.0);

    bank_open_row.assign(total_banks, -1);
    bank_row_ready.assign(total_banks, 0);
    bank_col_ready.assign(total_banks, 0);
    bank_pre_ready.assign(total_banks, 0);
    bank_act_ready.assign(total_banks, 0);

    scheduled_bursts.clear();
    write_queue.clear();
    last_bus_depart = 0;
    last_was_read = false;
    write_drain_until = 0;
  }

  Ramulator::AtomicModel::Cycle predict(Ramulator::AtomicModel::Access access,
                                      Ramulator::AtomicModel::Cycle now) override {
    int ch = access.levels[0];
    int rk = access.levels[1];
    int bg = access.levels[2];
    int bk = access.levels[3];
    int row = access.levels[4];
    int bank_id = ((ch * num_ranks + rk) * num_bankgroups + bg) * num_banks_per_group + bk;

    if (access.type == Ramulator::AtomicModel::AccessType::Write) {
      write_queue.push_back({bank_id, bg, row, now});
      if (static_cast<int>(write_queue.size()) > 64) {
        write_queue.erase(write_queue.begin());
      }
      if (now >= write_drain_until && static_cast<int>(write_queue.size()) >= write_high_wm) {
        write_drain_until = now + drain_duration;
        int to_remove = static_cast<int>(write_queue.size()) - write_low_wm;
        for (int i = 0; i < to_remove && i < static_cast<int>(write_queue.size()); ++i) {
          const auto& w = write_queue[i];
          bank_open_row[w.bank_id] = w.row;
          bank_pre_ready[w.bank_id] = std::max(bank_pre_ready[w.bank_id], write_drain_until + nWR);
        }
        if (to_remove > 0 && to_remove <= static_cast<int>(write_queue.size())) {
          write_queue.erase(write_queue.begin(), write_queue.begin() + to_remove);
        }
      }
      return now + hardware.read_latency;
    }

    // Read access:
    // Process any pending writes that could drain opportunistically before now
    drain_writes_opportunistic(now);

    int inflight = 0;
    int bg_inflight = 0;
    int bank_inflight = 0;
    for (const auto& b : scheduled_bursts) {
      if (b.depart > now) {
        inflight++;
        if (b.bg == bg) {
          bg_inflight++;
          if (b.bank_id == bank_id) {
            bank_inflight++;
          }
        }
      }
    }

    int bg_writes = 0;
    int bank_writes = 0;
    for (const auto& w : write_queue) {
      if (w.bg == bg) {
        bg_writes++;
        if (w.bank_id == bank_id) {
          bank_writes++;
        }
      }
    }

    // Global Command/Address bus load across all BankGroups
    int excess_writes = std::max(0, static_cast<int>(write_queue.size()) - num_bankgroups);
    double total_load = inflight + excess_writes * write_congestion_weight;
    double excess_global = std::max(0.0, total_load - num_bankgroups);

    // Local structural BankGroup contention (intra-BankGroup column and bank conflicts)
    double bg_load = bg_inflight + bg_writes * write_congestion_weight;
    double excess_bg = std::max(0.0, bg_load - 1.0);

    // Local structural Bank contention (serialization on single physical bank)
    double bank_load = bank_inflight + bank_writes * write_congestion_weight;
    double excess_bank = std::max(0.0, bank_load - 1.0);

    // Multi-level hierarchical queue arbitration delay:
    // 1. Base CA bus command pin serialization across the memory channel
    // 2. Global M/G/1 queue depth backpressure under deep controller saturation
    // 3. Local BankGroup structural clustering delay when accesses crowd a single bank group
    // 4. Local Bank structural serialization delay when multiple requests crowd the same bank
    double ca_wait_cycles = excess_global * ca_linear_delay 
                          + excess_global * excess_global * ca_quad_delay
                          + excess_bg * excess_bg * ca_bg_delay
                          + excess_bank * ca_bank_delay;
    Ramulator::AtomicModel::Cycle ca_wait = static_cast<Ramulator::AtomicModel::Cycle>(ca_wait_cycles);
    Ramulator::AtomicModel::Cycle base_now = std::max(now + ca_wait, write_drain_until);

    while (!scheduled_bursts.empty() && scheduled_bursts.front().depart < base_now - 200) {
      scheduled_bursts.erase(scheduled_bursts.begin());
    }

    int open_row = bank_open_row[bank_id];
    Ramulator::AtomicModel::Cycle cmd_time = 0;
    bool is_hit = (open_row == row);

    if (is_hit) {
      cmd_time = std::max(base_now, std::max(bank_row_ready[bank_id], bank_col_ready[bank_id]));
    } else if (open_row == -1) {
      Ramulator::AtomicModel::Cycle act_time = std::max(base_now, bank_act_ready[bank_id]);
      cmd_time = act_time + nRCD;
      bank_open_row[bank_id] = row;
    } else {
      Ramulator::AtomicModel::Cycle pre_time = std::max(base_now, bank_pre_ready[bank_id]);
      Ramulator::AtomicModel::Cycle act_time = std::max(pre_time + nRP, bank_act_ready[bank_id]);
      cmd_time = act_time + nRCD;
      bank_open_row[bank_id] = row;
    }

    Ramulator::AtomicModel::Cycle data_ready = cmd_time + t_hit;
    Ramulator::AtomicModel::Cycle t = schedule_bus_burst(data_ready, bank_id, bg, /*is_write=*/false);

    Ramulator::AtomicModel::Cycle actual_cmd_time = t - t_hit;
    bank_row_ready[bank_id] = actual_cmd_time;
    bank_col_ready[bank_id] = actual_cmd_time + nCCDL;
    bank_pre_ready[bank_id] = std::max(bank_pre_ready[bank_id], actual_cmd_time + nRTP);

    if (!is_hit) {
      Ramulator::AtomicModel::Cycle actual_act_time = actual_cmd_time - nRCD;
      bank_act_ready[bank_id] = actual_act_time + nRC;
      bank_pre_ready[bank_id] = std::max(bank_pre_ready[bank_id], actual_act_time + nRAS);
    }

    last_was_read = true;
    last_bus_depart = std::max(last_bus_depart, t);

    return t;
  }

 private:
  Ramulator::AtomicModel::Cycle get_bus_gap(bool prev_is_write, bool curr_is_write,
                                           int prev_bg, int curr_bg) const {
    if (!prev_is_write && !curr_is_write) {
      return (prev_bg == curr_bg) ? nCCDL : nCCDS;
    } else if (prev_is_write && curr_is_write) {
      return (prev_bg == curr_bg) ? nCCDL : nCCDS;
    } else if (!prev_is_write && curr_is_write) {
      return nRTW;
    } else {
      return (prev_bg == curr_bg) ? nWTRL : nWTRS;
    }
  }

  Ramulator::AtomicModel::Cycle find_bus_slot(Ramulator::AtomicModel::Cycle earliest_depart,
                                              int bg, bool is_write) const {
    Ramulator::AtomicModel::Cycle t = std::max(earliest_depart, write_drain_until);
    for (const auto& b : scheduled_bursts) {
      Ramulator::AtomicModel::Cycle gap_to_b = get_bus_gap(is_write, b.is_write, bg, b.bg);
      if (t + gap_to_b <= b.depart) {
        break;
      }
      Ramulator::AtomicModel::Cycle gap_from_b = get_bus_gap(b.is_write, is_write, b.bg, bg);
      if (t < b.depart + gap_from_b) {
        t = b.depart + gap_from_b;
      }
    }
    return t;
  }

  Ramulator::AtomicModel::Cycle schedule_bus_burst(Ramulator::AtomicModel::Cycle earliest_depart,
                                                  int bank_id, int bg, bool is_write) {
    Ramulator::AtomicModel::Cycle t = find_bus_slot(earliest_depart, bg, is_write);
    BusBurst new_burst{t, bank_id, bg, is_write};
    auto it = std::upper_bound(
        scheduled_bursts.begin(), scheduled_bursts.end(), t,
        [](Ramulator::AtomicModel::Cycle val, const BusBurst& b) { return val < b.depart; });
    scheduled_bursts.insert(it, new_burst);
    return t;
  }

  Ramulator::AtomicModel::Cycle execute_write(const QueuedWrite& qw,
                                             Ramulator::AtomicModel::Cycle earliest_start) {
    int bank_id = qw.bank_id;
    int bg = qw.bg;
    int row = qw.row;

    int open_row = bank_open_row[bank_id];
    Ramulator::AtomicModel::Cycle cmd_time = 0;
    bool is_hit = (open_row == row);

    if (is_hit) {
      cmd_time = std::max(earliest_start, std::max(bank_row_ready[bank_id], bank_col_ready[bank_id]));
    } else if (open_row == -1) {
      Ramulator::AtomicModel::Cycle act_time = std::max(earliest_start, bank_act_ready[bank_id]);
      cmd_time = act_time + nRCD;
      bank_open_row[bank_id] = row;
    } else {
      Ramulator::AtomicModel::Cycle pre_time = std::max(earliest_start, bank_pre_ready[bank_id]);
      Ramulator::AtomicModel::Cycle act_time = std::max(pre_time + nRP, bank_act_ready[bank_id]);
      cmd_time = act_time + nRCD;
      bank_open_row[bank_id] = row;
    }

    Ramulator::AtomicModel::Cycle data_ready = cmd_time + t_write_hit;
    Ramulator::AtomicModel::Cycle t = schedule_bus_burst(data_ready, bank_id, bg, /*is_write=*/true);
    Ramulator::AtomicModel::Cycle actual_cmd_time = t - t_write_hit;

    bank_row_ready[bank_id] = actual_cmd_time;
    bank_col_ready[bank_id] = actual_cmd_time + nCCDL;
    bank_pre_ready[bank_id] = std::max(bank_pre_ready[bank_id], t + nWR);

    if (!is_hit) {
      Ramulator::AtomicModel::Cycle actual_act_time = actual_cmd_time - nRCD;
      bank_act_ready[bank_id] = actual_act_time + nRC;
      bank_pre_ready[bank_id] = std::max(bank_pre_ready[bank_id], actual_act_time + nRAS);
    }

    last_was_read = false;
    last_bus_depart = std::max(last_bus_depart, t);
    return t;
  }

  void drain_writes_opportunistic(Ramulator::AtomicModel::Cycle now) {
    while (!write_queue.empty()) {
      const auto& qw = write_queue.front();
      if (qw.arrive >= now) {
        break;
      }

      int bank_id = qw.bank_id;
      int open_row = bank_open_row[bank_id];

      // Check when this write could start its first command
      Ramulator::AtomicModel::Cycle start_time = qw.arrive;
      if (open_row == qw.row) {
        start_time = std::max(start_time, std::max(bank_row_ready[bank_id], bank_col_ready[bank_id]));
      } else if (open_row == -1) {
        start_time = std::max(start_time, bank_act_ready[bank_id]);
      } else {
        start_time = std::max(start_time, bank_pre_ready[bank_id]);
      }

      if (start_time >= now) {
        break;
      }

      Ramulator::AtomicModel::Cycle t = execute_write(qw, qw.arrive);
      write_queue.erase(write_queue.begin());

      if (t > now) {
        break;
      }
    }
  }

  Ramulator::AtomicModel::Hardware hardware;
  int num_channels = 1;
  int num_ranks = 1;
  int num_bankgroups = 8;
  int num_banks_per_group = 4;
  int total_banks = 32;

  Ramulator::AtomicModel::Cycle t_hit = 43;
  Ramulator::AtomicModel::Cycle t_write_hit = 40;
  Ramulator::AtomicModel::Cycle nRCD = 34;
  Ramulator::AtomicModel::Cycle nRP = 34;
  Ramulator::AtomicModel::Cycle nRAS = 77;
  Ramulator::AtomicModel::Cycle nRC = 111;
  Ramulator::AtomicModel::Cycle nRTP = 18;
  Ramulator::AtomicModel::Cycle nCCDS = 8;
  Ramulator::AtomicModel::Cycle nCCDL = 12;
  Ramulator::AtomicModel::Cycle nRTW = 14;
  Ramulator::AtomicModel::Cycle nWTRL = 24;
  Ramulator::AtomicModel::Cycle nWTRS = 10;
  Ramulator::AtomicModel::Cycle nWR = 48;
  Ramulator::AtomicModel::Cycle nCWL = 32;
  Ramulator::AtomicModel::Cycle nBL = 8;
  Ramulator::AtomicModel::Cycle drain_duration = 247;

  double ca_linear_delay = 0.50;
  double ca_quad_delay = 0.02;
  double ca_bg_delay = 0.15;
  double ca_bank_delay = 1.2;
  double write_congestion_weight = 0.75;

  int write_high_wm = 51;
  int write_low_wm = 32;

  std::vector<int> bank_open_row;
  std::vector<Ramulator::AtomicModel::Cycle> bank_row_ready;
  std::vector<Ramulator::AtomicModel::Cycle> bank_col_ready;
  std::vector<Ramulator::AtomicModel::Cycle> bank_pre_ready;
  std::vector<Ramulator::AtomicModel::Cycle> bank_act_ready;

  std::vector<BusBurst> scheduled_bursts;
  std::vector<QueuedWrite> write_queue;
  Ramulator::AtomicModel::Cycle last_bus_depart = 0;
  bool last_was_read = false;
  Ramulator::AtomicModel::Cycle write_drain_until = 0;
};

}  // namespace

extern "C" std::uint32_t ramulator_atomic_model_api_version() {
  return Ramulator::AtomicModel::api_version;
}

extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model() {
  return new GenericDRAMModel;
}
