#include "ramulator/controller/atomic_model/api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

using Ramulator::AtomicModel::Access;
using Ramulator::AtomicModel::AccessType;
using Ramulator::AtomicModel::Cycle;
using Ramulator::AtomicModel::Hardware;
using Ramulator::AtomicModel::Parameters;

// ---------------------------------------------------------------------------
// Immediate-response DRAM model: work-conserving data bus + per-bank row state.
//
// Physical picture.  The channel has two resources a transaction must clear:
//
//   * the data bus.  Every admitted transaction (read or posted write drain)
//     owns one burst on the bus.  A read burst lasts `burst_cycles` cycles of
//     data plus `bus_gap` cycles of bus recovery; a write drain owns
//     `bus_interval + write_extra` cycles because it pays bus turnaround on
//     both sides.  The bus is *work conserving*: a transaction is placed in the
//     earliest free window at or after the moment its bank can deliver data.  A
//     read waiting on a row activation therefore does not block reads whose
//     banks are already ready.  Only the most recent `kWindow` reservations are
//     remembered, which bounds both the model state and how far the model looks
//     ahead; this is a bus-occupancy calendar, not a command scheduler.
//
//   * the target bank.  A DRAM bank serves one row at a time.  A transaction
//     whose row is already open is a row hit and needs only a column command
//     (`t_cmd` from command to data for a read; nothing extra for a posted
//     write, whose data is driven at the column command).  A transaction whose
//     row differs must first close the held row (`t_precharge`, only if a row
//     is actually held) and activate the new one (`t_activate`); the bank
//     cannot start anything else until its own data burst has been delivered.
//     Rows persist: controller traces of this platform show a bank keeping its
//     row open across long idle gaps.
//
// Write path.  A drain is *posted*: it does not wait for the frontend, and at
// admission time the model can only know that the write will have to move its
// bank's row.  Two bounded, reported-only corrections stand in for the
// controller behaviour that a pure arrival-order model cannot see:
//
//   `write_wait_scale`  -- the drain waits for the burst its bank is already
//       delivering (it cannot precharge mid-burst), scaled 0..1.
//   `drain_conflict_extra` -- a read that has to move the row a *drain* just
//       installed, while that drain is still recent, pays the drain's own bank
//       service as well: the controller serialises the deferred drain and the
//       read at that bank (two row cycles), which is what trace captures of a
//       bank alternating read-row / write-row show (reads completing at about
//       two row services while the model charges one).  The recency window
//       `drain_window_cycles` expires the term once the drain can be assumed
//       absorbed; both terms are added to the reported completion only and
//       never enter the bank or bus timelines, so they cannot compound.
//
// Reported completion = bus slot + burst, plus `queue_extra` times the time the
// request spent waiting for a free bus window.  These reported-only terms stand
// in for controller delays not modelled explicitly (refresh, command-bus
// limits, arbitration).  They never feed back into the reservation timeline, so
// the model's own schedule cannot run away.
//
// Address decoding (verified against controller traces of this platform):
//   column        = address bits 11:6        (64B lines within a 4KB stripe)
//   bank group    = bits 14:12, bank = bits 16:15, so the 32-way bank index
//                   is (address >> 12) & 31 exactly
//   row           = address bits 32:17       (128KB per row)
// ---------------------------------------------------------------------------
class WorkConservingBus final : public Ramulator::AtomicModel::Model {
 public:
  void initialize(Hardware hardware, Parameters& parameters) override {
    (void)hardware;
    t_cmd = parameters.number("t_cmd_to_data", 35.0, 1.0, 400.0);
    t_activate = parameters.number("t_activate", 34.0, 0.0, 400.0);
    t_precharge = parameters.number("t_precharge", 33.0, 0.0, 400.0);
    precharge_scale = parameters.number("precharge_scale", 1.0, 0.0, 1.0);
    burst = parameters.number("burst_cycles", 8.0, 1.0, 128.0);
    bus_gap = parameters.number("bus_gap", 0.0, 0.0, 64.0);
    bus_interval = parameters.number("bus_interval", 8.0, 1.0, 128.0);
    write_extra = parameters.number("write_extra", 6.0, 0.0, 256.0);
    write_wait_scale = parameters.number("write_wait_scale", 0.0, 0.0, 1.0);
    drain_conflict_extra =
        parameters.number("drain_conflict_extra", 0.0, 0.0, 400.0);
    drain_window_cycles =
        parameters.number("drain_window_cycles", 0.0, 0.0, 100000.0);
    same_bank_interval =
        parameters.number("same_bank_interval", 12.0, 1.0, 128.0);
    queue_extra = parameters.number("queue_extra", 0.0, 0.0, 4.0);
    fixed_offset = parameters.number("fixed_offset", 0.0, -128.0, 128.0);
  }

  Cycle predict(Access access, Cycle now) override {
    const std::int64_t address = access.address;
    const std::int64_t bank = (address >> 12) & 31;
    const std::int64_t row = address >> 17;

    Bank& state = banks_[bank];
    const double t_now = std::max(static_cast<double>(now), 0.0);

    if (access.type == AccessType::Read) {
      double data_ready;
      double drain_extra = 0.0;
      if (state.open && state.row == row) {
        // Row hit: only a column command is needed; the row stays open and the
        // bank pipelines column commands at its column-to-column spacing.
        const double column = std::max(
            t_now, static_cast<double>(state.column_last) + same_bank_interval);
        data_ready = column + t_cmd;
      } else {
        // Row (re)activation: the bank must finish the burst it is running,
        // close the row it holds (if any) and activate the new row.
        const double begin =
            std::max(t_now, static_cast<double>(state.busy_until));
        const double close_row =
            state.open ? precharge_scale * t_precharge : 0.0;
        data_ready = begin + close_row + t_activate + t_cmd;
        // A drain that installed the held row and has not yet been absorbed
        // still owns the bank for one row service; the read waits for it.
        if (state.row_from_write && drain_conflict_extra > 0.0 &&
            t_now - static_cast<double>(state.row_from_write_time) <=
                drain_window_cycles) {
          drain_extra = drain_conflict_extra;
        }
      }
      const double slot = reserve(data_ready, read_span());
      const double wait = slot - data_ready;

      // The bank now holds this row; its column pipeline and its occupancy are
      // anchored to the schedule that was actually handed out.
      state.open = true;
      state.row = row;
      state.row_from_write = false;
      state.column_last = snapped(slot - t_cmd);
      state.busy_until = snapped(slot + burst);

      return snapped(slot + burst + drain_extra + queue_extra * wait +
                     fixed_offset);
    }

    // Posted writeback: it waits for the bus, not for the frontend.  It still
    // has to move its bank's row, so it leaves that row open for the reads that
    // follow; `write_wait_scale` lets it also wait for the burst its bank is
    // already delivering (0 = the historical model, 1 = full bank pacing).
    const double ready =
        write_wait_scale > 0.0
            ? std::max(t_now,
                       t_now + write_wait_scale *
                                    (static_cast<double>(state.busy_until) -
                                     t_now))
            : t_now;
    const double slot = reserve(ready, write_span());
    state.open = true;
    state.row = row;
    state.row_from_write = true;
    state.row_from_write_time = snapped(slot);
    state.column_last = snapped(slot);
    state.busy_until = snapped(slot + burst);
    return snapped(slot + burst + fixed_offset);
  }

 private:
  static constexpr std::int64_t kIdle = -1000000000;
  static constexpr int kWindow = 128;  // remembered bus reservations

  static std::int64_t snapped(double value) {
    return static_cast<std::int64_t>(std::llround(value));
  }

  double read_span() const { return burst + bus_gap; }
  double write_span() const { return bus_interval + write_extra; }

  // Earliest free window at or after `ready` that fits a burst of `length`
  // cycles, packed against the reservations still in the window.
  double reserve(double ready, double length) {
    double t = ready;
    for (int i = 0; i < count_; ++i) {
      const double start = slots_[i];
      if (t < start + lengths_[i] && t + length > start) {
        t = start + lengths_[i];
      }
    }
    insert(t, length);
    return t;
  }

  void insert(double start, double length) {
    if (count_ == kWindow) {
      // Drop the least relevant reservation: the one that starts earliest,
      // because new admissions can never be scheduled before it.
      for (int i = 0; i + 1 < count_; ++i) {
        slots_[i] = slots_[i + 1];
        lengths_[i] = lengths_[i + 1];
      }
      --count_;
    }
    int pos = count_;
    while (pos > 0 && slots_[pos - 1] > start) {
      slots_[pos] = slots_[pos - 1];
      lengths_[pos] = lengths_[pos - 1];
      --pos;
    }
    slots_[pos] = start;
    lengths_[pos] = length;
    ++count_;
  }

  struct Bank {
    bool open = false;
    std::int64_t row = 0;
    std::int64_t column_last = 0;     // last column command accepted here
    std::int64_t busy_until = kIdle;  // bank free again after this cycle
    bool row_from_write = false;      // held row was installed by a drain
    std::int64_t row_from_write_time = kIdle;  // when that drain was posted
  };

  Bank banks_[32];
  double slots_[kWindow];
  double lengths_[kWindow];
  int count_ = 0;

  double t_cmd = 35.0;
  double t_activate = 34.0;
  double t_precharge = 33.0;
  double precharge_scale = 1.0;
  double burst = 8.0;
  double bus_gap = 0.0;
  double bus_interval = 8.0;
  double write_extra = 6.0;
  double write_wait_scale = 0.0;
  double drain_conflict_extra = 0.0;
  double drain_window_cycles = 0.0;
  double same_bank_interval = 12.0;
  double queue_extra = 0.0;
  double fixed_offset = 0.0;
};

}  // namespace

extern "C" std::uint32_t ramulator_atomic_model_api_version() {
  return Ramulator::AtomicModel::api_version;
}

extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model() {
  return new WorkConservingBus;
}
