#include "ramulator/controller/atomic_model/api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <set>
#include <vector>

// Immediate-response DRAM model: a resource-reservation timeline.
//
// Every admitted request is placed once, at admission, onto a timeline of the
// DRAM resources it needs, and its completion is returned immediately. Read
// predictions are never revised. There is no per-cycle command selection: the
// timeline only answers "given everything booked so far, when is the earliest
// legal time for this request's commands?".
//
// Physical state
//  * Bank: row left open by the last booked access and the earliest legal
//    ACT / PRE / column times implied by tRC, tRP, tRAS, tRCD, tRTP and write
//    recovery. A bank also remembers its recent row windows [ACT, PRE): a
//    request that hits a row whose window is still open when it could issue
//    is served inside that window, ahead of the conflicting request that will
//    close it (the row-hit-first part of FR-FCFS). Otherwise accesses to a
//    bank are booked first-come-first-served. A row conflict found at
//    admission on an idle bank activates tRP - 1 after the PRE's earliest
//    slot (rp_adjust): the controller can issue that PRE in the admission
//    cycle itself, one cycle before the model's earliest command slot. A PRE
//    that instead waits for the bank's own tRAS or tRTP limit gains nothing
//    from the admission boundary: it issues pre_wait cycles after that limit,
//    so its ACT follows the limit by a full tRP. After write recovery the PRE
//    keeps the earlier slot (see pre_wait below).
//  * Column calendar: booked RD/WR commands of the channel. It enforces
//    tCCD_S/L (read and write variants), write-to-read turnaround
//    (CWL + BL + tWTR_S/L) and read-to-write turnaround (tRTW). A later request
//    may use a gap left between earlier bookings, which is how independent
//    banks overlap (the ready-first part of FR-FCFS). Reads of one bank group
//    are tCCD_L apart, leaving a bubble shorter than two bursts; a ready read
//    of another bank group is issued into that bubble and the booked read
//    behind it slips by up to tCCD_L - tCCD_S. Booked predictions cannot move,
//    so the newcomer is allowed that much overlap instead of queueing behind
//    the whole same-bank-group run. With col_slip the calendar entries of the
//    slipped reads move as well (their issued predictions do not), so the
//    channel never carries more bursts than its data bus allows.
//    slip_sync = 1 keeps the rest of the timeline consistent with a slip: the
//    slipped read's bank cannot precharge before its RD + tRTP, and a slipped
//    row hit leaves the read queue only at its moved RD, so the read queue is
//    not seen empty (and write mode is not entered) while it still waits.
//  * ACT calendar: booked activations, enforcing tRRD_S/L.
//  * Write buffer: posted writes wait here and are booked onto the same
//    timeline only when the controller would be in write mode: when the read
//    queue is empty, or after the buffer crosses the high watermark (forced
//    drain down to the low watermark, booked when the next read arrives, which
//    waits for it). Reads that match a buffered write are forwarded. A write
//    whose PRE has issued but whose ACT has not when a read arrives leaves its
//    bank precharged and stays buffered.
//  * Read-queue horizon: the latest time a booked read leaves the read queue
//    (its ACT, or its RD for a row hit). After that time the controller sees an
//    empty read queue and drains writes. The latest booked read column command
//    and the latest booked read ACT are kept as well (see the forced drain
//    below).
//  * Read arrival estimate: when a read's ACT empties the read queue, buffered
//    writes are drained while it waits for tRCD, until the next read arrives.
//    The expected time to that arrival is taken from the reads still in
//    flight and their mean latency (Little's law). Reads arrive in bursts, so
//    the recent admission-to-next-admission gaps observed with the same
//    number of reads in flight are a sharper estimate. A read admitted with
//    no other read in flight is either the head of a burst of independent
//    misses, followed within a few cycles, or a link of a dependent chain,
//    whose next read only arrives after this one returns (no estimate).
//    arrival_model = 3 applies the same distinction with reads in flight: the
//    core's window of outstanding misses is bounded (reorder buffer, miss
//    registers), so a read either continues the current burst (the next read
//    follows after the usual gap) or fills the window and ends it; the next
//    read is then only issued after the oldest outstanding miss returns (the
//    earliest predicted completion in flight, plus the usual delay after
//    it). Which of the two holds is learned per in-flight count from the
//    model's own recent admissions and predictions; arrival_buckets must
//    exceed the window's typical miss count for its end to be recognised.
//
// Admission boundary: a read admitted at cycle t issues its first command at
// t + 1 at the earliest, so the controller has already chosen whatever it
// issues at t without it. By default the write-mode replay only commits write
// commands issued before t, which lets the new read overtake a write the
// controller started in the very cycle the read arrived (typically the
// writeback of the previous read's fill, followed one cycle later by the
// dependent read). same_cycle_writes = 1 also commits the write commands
// issued at t: the read then pays that write's turnaround, and a read of the
// same line is no longer forwarded from a write that already left the buffer.
//
// Forced drains: a forced drain also freezes reads that were admitted before
// the buffer crossed the high watermark and have not yet left the read queue.
// Their predictions were already issued, so the model serves them first and
// books the drain after them: the two phases trade places but keep their
// lengths. The frozen reads are under-predicted, and the reads after the drain
// over-predicted, by about the drain length. By default the drain's PRE/ACT
// may overlap those reads' remaining commands. drain_after_reads = 1 makes
// write mode exclusive instead: the drain starts after their column commands.
// drain_after_reads = 2 only keeps the drain's commands behind their ACTs,
// the part of their service that write mode really blocks: reads that have
// activated keep priority over the drain, so their column commands overlap
// its bank preparation. A booked row hit, however, leaves the read queue at
// its RD, so under option 2 the drain's bank preparation also waits for the
// last booked row-hit RD. drain_after_reads = 3 keeps the drain behind the
// booked ACTs only: the frozen row hits are served (in the controller: after
// the drain) on the data bus alone, and the drain's PRE/ACT overlaps them as
// it overlaps the column commands of the activated reads.
// While a booked drain is in progress the read queue is not served;
// drain_block = 1 makes reads admitted before its end wait for it.
//
// Writebacks during a forced drain: the controller leaves write mode only once
// the buffer is below the low watermark, counting the writes that arrive while
// it drains. Reads still waiting in the read queue are not served, so the only
// reads completing are those that had left it (activated, or issued the column
// command of a row hit) when the buffer crossed the high watermark. Each of
// their fills may evict a dirty line whose writeback joins the buffer before
// the drain ends. Those writes arrive after the drain has been booked (at the
// next read's admission). drain_pending > 0 makes the booked drain take the
// expected number of them (those reads still in flight, times the recent
// writes-per-read ratio, times drain_pending) from the writes already
// buffered: the drain gets its true length, and the buffer level it leaves
// behind (which decides when the next forced drain comes) is the
// controller's. Not every such fill evicts in time (the writeback leaves the
// cache hierarchy some time after the fill), which drain_pending < 1 accounts
// for. In the controller an activated read has priority and returns within
// about tRCD + read latency, so at most (tRCD + read latency) / tCCD_S reads
// are in that state at once; pending_cap = 1 limits the count to that
// pipeline depth, since the timeline may book ACTs further ahead.
// pending_unforced = 1 applies the same expectation to write mode entered on
// an empty read queue: a read arriving while the buffer is at or above the low
// watermark does not end it, so the drain continues until the buffer, with
// the writebacks of the reads in flight at entry, falls below it. Without
// this, those writebacks stay buffered and cost the next read a whole second
// drain (PRE, ACT, write recovery, PRE, ACT) that the controller never pays.
namespace {

using Ramulator::AtomicModel::Access;
using Ramulator::AtomicModel::AccessType;
using Ramulator::AtomicModel::Cycle;
using Ramulator::AtomicModel::Hardware;
using Ramulator::AtomicModel::Parameters;

constexpr Cycle kNever = std::numeric_limits<Cycle>::max() / 4;

struct Window {
  std::int64_t row;
  Cycle act;  // ACT that opened the row
  Cycle pre;  // PRE that closes it; kNever while it is the open row
};

struct Bank {
  std::int64_t row = -1;  // row left open by the last booked access; -1 = precharged
  Cycle act_ok = 0;       // earliest ACT (previous ACT + tRC, PRE + tRP)
  Cycle pre_ok = 0;       // earliest PRE (ACT + tRAS, RD + tRTP, WR + CWL + BL + tWR)
  bool pre_after_write = false;  // pre_ok is set by write recovery
  Cycle col_ok = 0;       // earliest column command (ACT + tRCD, bank order)
  std::deque<Window> windows;
};

struct Command {
  Cycle t;
  int bank_group;
  bool write;
  int bank = -1;        // bank of a column command (-1: not recorded)
  bool leaves = false;  // a row-hit read: it leaves the read queue with this command
};

struct PendingWrite {
  std::int64_t address;
  int bank;
  int bank_group;
  std::int64_t row;
  Cycle arrive;
};

struct Booking {
  Cycle first = 0;   // first command issued for the request (PRE, ACT or column)
  Cycle leave = 0;   // leaves the request queue: its ACT, or its column command on a row hit
  Cycle column = 0;  // RD/WR command
  bool opened = false;
  bool inserted = false;  // served inside an earlier window of its row
};

// A read that has not returned: when it left the read queue and its prediction.
struct OutstandingRead {
  Cycle leave;
  Cycle done;
};

// The most recent samples of a signed interval (DRAM cycles).
struct RecentSamples {
  static constexpr int kSize = 16;
  std::array<Cycle, kSize> ring{};
  int count = 0;
  int pos = 0;

  void add(Cycle v) {
    ring[static_cast<std::size_t>(pos)] = v;
    pos = (pos + 1) % kSize;
    count = std::min(count + 1, kSize);
  }
  Cycle quantile(double q) const {
    std::array<Cycle, kSize> s = ring;
    const auto end = s.begin() + count;
    const auto k = s.begin() + std::min(count - 1, static_cast<int>(q * count));
    std::nth_element(s.begin(), k, end);
    return *k;
  }
};

class ReservationModel final : public Ramulator::AtomicModel::Model {
 public:
  void initialize(Hardware hw, Parameters& p) override {
    auto timing = [&](const char* name, Cycle fallback) {
      const auto it = hw.timings.find(name);
      return (it == hw.timings.end() || it->second < 0) ? fallback : it->second;
    };
    tCL = timing("nCL", 34);
    tBL = timing("nBL", 8);
    tRCD = timing("nRCD", 34);
    tRP = timing("nRP", 34);
    tRAS = timing("nRAS", 77);
    tRC = timing("nRC", tRAS + tRP);
    tRTP = timing("nRTP", 18);
    tWR = timing("nWR", 72);
    tCWL = timing("nCWL", tCL - 2);
    tCCDS = timing("nCCDS", tBL);
    tCCDL = timing("nCCDL", tCCDS);
    tCCDS_WR = timing("nCCDS_WR", tCCDS);
    tCCDL_WR = timing("nCCDL_WR", tCCDL);
    tRRDS = timing("nRRDS", 4);
    tRRDL = timing("nRRDL", tRRDS);
    tWTRS = timing("nWTRS", 4);
    tWTRL = timing("nWTRL", tWTRS);
    tRTW = timing("nRTW", tBL);
    read_latency = hw.read_latency;
    max_col_gap = std::max({tCCDL, tCCDL_WR, tCWL + tBL + tWTRL, tRTW});
    max_act_gap = std::max(tRRDS, tRRDL);
    mean_latency = static_cast<double>(tRCD + read_latency);

    // A row conflict on an idle bank was observed to cost one cycle less than
    // tRP + tRCD over a row hit, so the PRE-to-ACT spacing is adjusted by this
    // many cycles.
    rp_adjust = static_cast<Cycle>(p.number("rp_adjust", -1, -4, 4));
    // A conflict whose PRE waits for the bank's tRAS or a read's tRTP was
    // observed to activate a full tRP after that limit: the PRE issues this
    // many cycles after the bank allows it. Not applied after write recovery:
    // there it improves individual reads in replay but worsened core-cycle
    // accuracy in closed loop.
    pre_wait = static_cast<Cycle>(p.number("pre_wait", 0, 0, 4));
    // Latency of a read served from a buffered write to the same line.
    forward_latency = static_cast<Cycle>(p.number("forward_latency", 1, 1, 64));
    // Acknowledgement latency returned for a posted write.
    write_ack = static_cast<Cycle>(p.number("write_ack", 1, 1, 100000));
    // 1: posted writes occupy banks and the bus (write-mode drains); 0: ignored.
    model_writes = p.number("model_writes", 1, 0, 1) > 0.5;
    // 1: writes drained while a read waits for tRCD may push that read's RD.
    starve_reads = p.number("starve_reads", 1, 0, 1) > 0.5;
    // 1: that starvation ends when the next read is expected to arrive.
    arrival_gate = p.number("arrival_gate", 1, 0, 1) > 0.5;
    // Next-arrival estimate: 0 Little's law only (none with no read in
    // flight); 1 plus recent gaps for reads admitted alone; 2 recent gaps
    // for every in-flight count; 3 burst-or-completion for every count.
    arrival_model = static_cast<int>(p.number("arrival_model", 0, 0, 3));
    // Quantile of the recent gaps used as the expected next-arrival time.
    arrival_quantile = p.number("arrival_quantile", 0.5, 0.0, 1.0);
    // Number of in-flight-count histories (the last one collects all larger counts).
    n_buckets = static_cast<std::size_t>(p.number("arrival_buckets", 8, 1, static_cast<double>(kGapBuckets)));
    // 1: a forced drain that begins while booked reads still wait to leave the
    // read queue starts after their column commands (exclusive write mode);
    // 2: after their departures (ACTs, or RDs of row hits); 3: after their ACTs.
    drain_after_reads = static_cast<int>(p.number("drain_after_reads", 0, 0, 3));
    // 1: reads admitted during a booked drain wait for its end.
    drain_block = static_cast<int>(p.number("drain_block", 0, 0, 1));
    // 1: write commands issued in a read's admission cycle precede that read.
    same_cycle_writes = p.number("same_cycle_writes", 0, 0, 1) > 0.5;
    // 1: row hits may be served inside an earlier, still-open window of their row.
    hit_windows = p.number("hit_windows", 1, 0, 1) > 0.5;
    // Cycles by which a new read may slip a booked later read (default: the
    // tCCD_L bubble minus one burst).
    col_squeeze = static_cast<Cycle>(p.number("col_squeeze", static_cast<double>(std::max<Cycle>(0, tCCDL - tCCDS)), 0, 16));
    // 1: the slipped reads' calendar entries move later (bandwidth conserved).
    col_slip = p.number("col_slip", 0, 0, 1) > 0.5;
    // 1: a slip also delays the slipped read's bank (PRE, next column) and, for
    // a row hit, its departure from the read queue.
    slip_sync = p.number("slip_sync", 0, 0, 1) > 0.5;
    // Writebacks expected per in-flight activated read during a drain, as a
    // multiple of the recent writes-per-read ratio (0: not modelled).
    drain_pending = p.number("drain_pending", 0, 0, 2);
    // 1: at most (tRCD + read latency) / tCCD_S activated reads are counted.
    const bool pending_cap = p.number("pending_cap", 0, 0, 1) > 0.5;
    pending_limit = pending_cap
                        ? static_cast<std::size_t>((tRCD + read_latency + tCCDS - 1) / std::max<Cycle>(1, tCCDS))
                        : std::numeric_limits<std::size_t>::max();
    // 1: those writebacks also extend a write mode entered on an empty read
    // queue, when a read finds the buffer at or above the low watermark.
    pending_unforced = p.number("pending_unforced", 0, 0, 1) > 0.5;

    ch_idx = hw.level_index("Channel");
    rank_idx = hw.level_index("Rank");
    bg_idx = hw.level_index("BankGroup");
    bank_idx = hw.level_index("Bank");
    row_idx = hw.level_index("Row");
    n_ranks = hw.level_size("Rank");
    n_bgs = hw.level_size("BankGroup");
    n_banks = hw.level_size("Bank");
    banks.assign(static_cast<std::size_t>(hw.level_size("Channel") * n_ranks * n_bgs * n_banks), Bank{});

    // Controller semantics: enter write mode when size > high * capacity; leave
    // it when size < low * capacity and reads are waiting.
    const double cap = static_cast<double>(hw.write_capacity);
    wq_high = static_cast<std::size_t>(std::floor(hw.write_high_watermark * cap)) + 1;
    wq_low = static_cast<std::size_t>(std::ceil(hw.write_low_watermark * cap));
  }

  Cycle predict(Access a, Cycle now) override {
    prune(now);
    const int bank = bank_id(a);
    const int bg = a.levels[bg_idx];
    const std::int64_t row = a.levels[row_idx];
    write_share += ((a.type == AccessType::Write ? 1.0 : 0.0) - write_share) / 256.0;

    if (a.type == AccessType::Write) {
      if (!model_writes) return now + write_ack;
      advance(now);
      wq.push_back({a.address, bank, bg, row, now});
      if (wq.size() >= wq_high && !(write_mode && forced)) {
        write_mode = true;
        forced = true;
        forced_at = now;
        drain_start = now + 1;
        // Reads still waiting to leave the read queue are frozen by the
        // drain; they were booked first, so the drain follows them.
        if (read_horizon > now) {
          if (drain_after_reads == 1) drain_start = std::max(drain_start, read_col_horizon + 1);
          if (drain_after_reads == 2) drain_start = std::max(drain_start, read_horizon + 1);
          if (drain_after_reads == 3) drain_start = std::max(drain_start, read_act_horizon + 1);
        }
      }
      return now + write_ack;
    }

    if (model_writes) {
      advance(now);
      for (const auto& w : wq) {
        if (w.address == a.address) return now + forward_latency;
      }
    }

    outstanding.erase(std::remove_if(outstanding.begin(), outstanding.end(),
                                     [now](const OutstandingRead& r) { return r.done <= now; }),
                      outstanding.end());
    Cycle earliest = now + 1;
    if (write_mode) {
      // Write mode persists until the buffer falls below the low watermark,
      // counting the writebacks still expected meanwhile. An unforced write
      // mode ends at once if the buffer is already below it.
      Cycle exit = now;
      bool drained = false;
      const std::size_t expected = (forced || wq.size() >= wq_low) ? pending_writebacks(now) : 0;
      while (!wq.empty() && wq.size() + expected >= wq_low) {
        exit = std::max(exit, drain_one(drain_start).leave);
        drained = true;
      }
      earliest = std::max(earliest, exit + 1);
      if (drained) drain_exit = exit;
      write_mode = false;
      forced = false;
    } else if (drain_block >= 1 && now <= drain_exit) {
      earliest = std::max(earliest, drain_exit + 1);
    }

    // Expected arrival of the next read.
    while (!in_flight.empty() && *in_flight.begin() <= now) in_flight.erase(in_flight.begin());
    const std::size_t k = std::min(in_flight.size(), n_buckets - 1);
    const Cycle first_done = in_flight.empty() ? kNever : *in_flight.begin();
    if (last_k >= 0) {
      gaps[static_cast<std::size_t>(last_k)].add(now - last_read);
      if (last_k == 0) {
        alone_after_done.add(now - last_done);
      } else {
        after_first[static_cast<std::size_t>(last_k)].add(now - last_first_done);
      }
    }
    const RecentSamples& gap = gaps[k];
    Cycle next_read = kNever;
    if (arrival_gate && !in_flight.empty()) {
      // Little's law over the reads in flight.
      next_read = now + static_cast<Cycle>(std::ceil(mean_latency / static_cast<double>(in_flight.size())));
      if (arrival_model == 2 && gap.count >= 4) next_read = now + gap.quantile(arrival_quantile);
      if (arrival_model == 3 && k > 0 && gap.count >= 4 && after_first[k].count >= 4) {
        // Burst continues: the usual gap. Burst ends: the next read waits
        // for the oldest outstanding miss to return.
        const Cycle after = after_first[k].quantile(arrival_quantile);
        next_read = after < 0 ? now + gap.quantile(arrival_quantile) : std::max(now + 1, first_done + after);
      }
    } else if (arrival_gate && arrival_model >= 1 && gap.count >= 4 &&
               alone_after_done.quantile(arrival_quantile) < 0) {
      // Reads admitted alone are usually followed by an independent read
      // before they return (a burst head): expect it after the usual gap.
      next_read = now + gap.quantile(arrival_quantile);
    }

    Booking b = book(bank, bg, row, false, earliest, false);
    if (b.opened && starve_reads && model_writes && !wq.empty() && read_horizon <= b.leave && b.leave < next_read) {
      // Once this read's ACT issues, the read queue is empty and the
      // controller drains writes while the read waits for tRCD. Every WR that
      // issues first pushes the RD by the write-to-read turnaround. Writes stop
      // starting once the next read is expected to arrive.
      const Cycle act = b.leave;
      open_row(bank, bg, row, b);
      const Cycle col_ready = banks[static_cast<std::size_t>(bank)].col_ok;
      Cycle rd = col_slot(col_ready, bg, false);
      for (bool progress = true; progress && !wq.empty();) {
        progress = false;
        for (std::size_t i = 0; i < wq.size(); ++i) {
          const PendingWrite& w = wq[i];
          const Booking pk = book(w.bank, w.bank_group, w.row, true, std::max(act + 1, w.arrive + 1), false);
          if (pk.column < rd && pk.first < next_read) {
            commit_write(i, act + 1);
            rd = col_slot(col_ready, bg, false);
            progress = true;
            break;
          }
        }
      }
      b.column = rd;
      finish_column(bank, bg, false, rd, false);
    } else {
      b = book(bank, bg, row, false, earliest, true);
    }
    read_horizon = std::max(read_horizon, b.leave);
    read_col_horizon = std::max(read_col_horizon, b.column);
    if (b.opened) read_act_horizon = std::max(read_act_horizon, b.leave);
    const Cycle done = b.column + read_latency;
    in_flight.insert(done);
    outstanding.push_back({b.leave, done});
    mean_latency += (static_cast<double>(done - now) - mean_latency) / 32.0;
    last_k = static_cast<int>(k);
    last_read = now;
    last_done = done;
    last_first_done = first_done;
    return done;
  }

 private:
  // In-flight-count histories available (the controller holds at most 64 reads).
  static constexpr std::size_t kGapBuckets = 65;

  int bank_id(const Access& a) const {
    return ((a.levels[ch_idx] * n_ranks + a.levels[rank_idx]) * n_bgs + a.levels[bg_idx]) * n_banks +
           a.levels[bank_idx];
  }

  // Writebacks expected to join the buffer after `t` and before the current
  // write mode ends: one per recent writes-per-read for every read that had
  // left the read queue when write mode began (the buffer crossed the high
  // watermark, or the read queue emptied) and has not returned by `t` (reads
  // still queued only return after the drain).
  std::size_t pending_writebacks(Cycle t) const {
    if (drain_pending <= 0.0 || !write_mode || (!forced && !pending_unforced)) return 0;
    const Cycle since = forced ? forced_at : drain_start;
    std::size_t n = 0;
    for (const OutstandingRead& r : outstanding) {
      if (r.leave <= since && r.done > t) ++n;
    }
    n = std::min(n, pending_limit);
    const double per_read = std::min(4.0, write_share / std::max(1e-3, 1.0 - write_share));
    const auto expected = static_cast<std::size_t>(std::lround(drain_pending * per_read * static_cast<double>(n)));
    return std::min(expected, wq_low - 1);
  }

  // Replay the write-mode policy up to `now`. The controller drains writes
  // whenever its read queue is empty, and after a forced entry until the
  // buffer falls below the low watermark while reads are waiting. Only
  // commands issued before `now` (with same_cycle_writes, up to and including
  // `now`) are committed: a write whose PRE has issued but whose ACT has not
  // leaves its bank precharged and stays buffered.
  void advance(Cycle now) {
    const Cycle until = same_cycle_writes ? now + 1 : now;
    while (!wq.empty()) {
      if (!write_mode) {
        const Cycle s = std::max(read_horizon + 1, wq.front().arrive + 1);
        if (s >= until) return;
        write_mode = true;
        forced = false;
        drain_start = s;
      }
      if (forced && wq.size() + pending_writebacks(until) < wq_low && read_horizon > forced_at) {
        write_mode = false;
        forced = false;
        continue;
      }
      std::size_t pick = 0;
      Booking best;
      best.first = kNever;
      for (std::size_t i = 0; i < wq.size(); ++i) {
        const PendingWrite& w = wq[i];
        const Booking pk = book(w.bank, w.bank_group, w.row, true, std::max(drain_start, w.arrive + 1), false);
        if (pk.first < best.first) {
          best = pk;
          pick = i;
        }
      }
      if (best.first >= until) return;
      if (best.leave < until) {
        commit_write(pick, drain_start);
      } else {
        precharge(wq[pick].bank, best.first);
      }
    }
    write_mode = false;
    forced = false;
  }

  // Drain the buffered write that can start earliest (ready-first order).
  Booking drain_one(Cycle from) {
    std::size_t pick = 0;
    Cycle first = kNever;
    for (std::size_t i = 0; i < wq.size(); ++i) {
      const PendingWrite& w = wq[i];
      const Booking pk = book(w.bank, w.bank_group, w.row, true, std::max(from, w.arrive + 1), false);
      if (pk.first < first) {
        first = pk.first;
        pick = i;
      }
    }
    return commit_write(pick, from);
  }

  Booking commit_write(std::size_t i, Cycle from) {
    const PendingWrite w = wq[i];
    wq.erase(wq.begin() + static_cast<std::ptrdiff_t>(i));
    return book(w.bank, w.bank_group, w.row, true, std::max(from, w.arrive + 1), true);
  }

  // Place one access: PRE/ACT if the row is not open, then its column command.
  // With commit = false the timeline is left unchanged (a probe).
  Booking book(int bank_i, int bg, std::int64_t row, bool write, Cycle earliest, bool commit) {
    Bank& bank = banks[static_cast<std::size_t>(bank_i)];
    if (hit_windows && bank.row != row) {
      // Row hit inside an earlier window that is still open when the request can issue.
      for (auto it = bank.windows.rbegin(); it != bank.windows.rend(); ++it) {
        if (it->pre <= earliest) break;
        if (it->row != row) continue;
        const Cycle col = col_slot(std::max(earliest, it->act + tRCD), bg, write);
        if (col >= it->pre) break;
        Booking out;
        out.first = out.leave = out.column = col;
        out.inserted = true;
        if (commit) {
          insert(cols, {col, bg, write, bank_i, !write});
          if (!write) slip_after(col);
        }
        return out;
      }
    }
    const Bank saved = bank;
    Booking out = plan_open(saved, bg, row, earliest);
    if (commit && out.opened) open_row(bank_i, bg, row, out);
    const Bank& b = commit ? banks[static_cast<std::size_t>(bank_i)] : saved;
    const Cycle col_ready = out.opened ? out.leave + tRCD : std::max(earliest, b.col_ok);
    out.column = col_slot(col_ready, bg, write);
    if (!out.opened) out.first = out.leave = out.column;
    if (commit) finish_column(bank_i, bg, write, out.column, !out.opened);
    return out;
  }

  Booking plan_open(const Bank& b, int bg, std::int64_t row, Cycle earliest) const {
    Booking out;
    if (b.row == row) return out;
    out.opened = true;
    Cycle act_ready = std::max(earliest, b.act_ok);
    if (b.row >= 0) {
      const Cycle pre = std::max(earliest, b.pre_ok + (b.pre_after_write ? 0 : pre_wait));
      out.first = pre;
      act_ready = std::max(act_ready, pre + tRP + rp_adjust);
    }
    out.leave = act_slot(act_ready, bg);
    if (b.row < 0) out.first = out.leave;
    return out;
  }

  void open_row(int bank_i, int bg, std::int64_t row, const Booking& b) {
    Bank& bank = banks[static_cast<std::size_t>(bank_i)];
    if (bank.row >= 0 && !bank.windows.empty()) bank.windows.back().pre = b.first;
    bank.windows.push_back({row, b.leave, kNever});
    if (bank.windows.size() > 8) bank.windows.pop_front();
    bank.row = row;
    bank.act_ok = b.leave + tRC;
    bank.pre_ok = b.leave + tRAS;
    bank.pre_after_write = false;
    bank.col_ok = b.leave + tRCD;
    insert(acts, {b.leave, bg, false});
  }

  // A PRE issued for a buffered write that has not yet activated its row.
  void precharge(int bank_i, Cycle pre) {
    Bank& bank = banks[static_cast<std::size_t>(bank_i)];
    if (bank.row >= 0 && !bank.windows.empty()) bank.windows.back().pre = pre;
    bank.row = -1;
    bank.act_ok = std::max(bank.act_ok, pre + tRP + rp_adjust);
  }

  // `leaves`: the column command is also the request's departure from its
  // queue (a row hit).
  void finish_column(int bank_i, int bg, bool write, Cycle col, bool leaves) {
    Bank& bank = banks[static_cast<std::size_t>(bank_i)];
    bank.col_ok = std::max(bank.col_ok, col);
    const Cycle pre = write ? col + tCWL + tBL + tWR : col + tRTP;
    if (pre >= bank.pre_ok) {
      bank.pre_ok = pre;
      bank.pre_after_write = write;
    }
    insert(cols, {col, bg, write, bank_i, leaves && !write});
    if (!write) slip_after(col);
  }

  // A read booked at t into the bubble ahead of booked reads delays them in
  // the controller. Their predictions stand, but their calendar entries move
  // to the earliest legal spacing, so later bookings see the true bus use.
  void slip_after(Cycle t) {
    if (!col_slip || col_squeeze == 0) return;
    auto it = std::upper_bound(cols.begin(), cols.end(), t, [](Cycle v, const Command& x) { return v < x.t; });
    bool moved = false;
    for (; it != cols.end() && !it->write; ++it) {
      Cycle need = it->t;
      for (auto jt = it; jt != cols.begin();) {
        --jt;
        if (jt->t + max_col_gap <= need) break;
        need = std::max(need, jt->t + col_gap(jt->write, jt->bank_group, false, it->bank_group));
      }
      if (need == it->t) break;
      it->t = need;
      read_col_horizon = std::max(read_col_horizon, need);
      if (slip_sync && it->bank >= 0) {
        // The slipped RD still precedes its bank's next PRE and column
        // command; a slipped row hit waits in the read queue until it issues.
        Bank& bk = banks[static_cast<std::size_t>(it->bank)];
        bk.col_ok = std::max(bk.col_ok, need);
        if (need + tRTP > bk.pre_ok) {
          bk.pre_ok = need + tRTP;
          bk.pre_after_write = false;
        }
        if (it->leaves) read_horizon = std::max(read_horizon, need);
      }
      moved = true;
    }
    if (moved && !std::is_sorted(cols.begin(), cols.end(), [](const Command& x, const Command& y) { return x.t < y.t; })) {
      std::stable_sort(cols.begin(), cols.end(), [](const Command& x, const Command& y) { return x.t < y.t; });
    }
  }

  // Minimum spacing when column command `first` precedes column command `second`.
  Cycle col_gap(bool first_write, int first_bg, bool second_write, int second_bg) const {
    const bool same = first_bg == second_bg;
    if (!first_write && !second_write) return same ? tCCDL : tCCDS;
    if (first_write && second_write) return same ? tCCDL_WR : tCCDS_WR;
    if (first_write) return tCWL + tBL + (same ? tWTRL : tWTRS);
    return tRTW;
  }

  // Earliest column command time >= t compatible with every booked column
  // command. A read placed ahead of a booked read may overlap it by up to
  // col_squeeze cycles (that booked read slips in the controller instead).
  Cycle col_slot(Cycle t, int bg, bool write) const {
    for (;;) {
      bool moved = false;
      auto it = std::lower_bound(cols.begin(), cols.end(), t - max_col_gap,
                                 [](const Command& c, Cycle v) { return c.t < v; });
      for (; it != cols.end() && it->t < t + max_col_gap; ++it) {
        const Cycle before = col_gap(it->write, it->bank_group, write, bg);
        if (it->t <= t) {
          if (t < it->t + before) { t = it->t + before; moved = true; break; }
        } else {
          const Cycle slack = (!write && !it->write) ? col_squeeze : 0;
          if (it->t - t < col_gap(write, bg, it->write, it->bank_group) - slack) {
            t = it->t + before;
            moved = true;
            break;
          }
        }
      }
      if (!moved) return t;
    }
  }

  Cycle act_slot(Cycle t, int bg) const {
    for (;;) {
      bool moved = false;
      auto it = std::lower_bound(acts.begin(), acts.end(), t - max_act_gap,
                                 [](const Command& c, Cycle v) { return c.t < v; });
      for (; it != acts.end() && it->t < t + max_act_gap; ++it) {
        const Cycle gap = it->bank_group == bg ? tRRDL : tRRDS;
        if ((it->t <= t && t < it->t + gap) || (it->t > t && it->t - t < gap)) {
          t = it->t + gap;
          moved = true;
          break;
        }
      }
      if (!moved) return t;
    }
  }

  static void insert(std::vector<Command>& cal, Command c) {
    auto it = std::upper_bound(cal.begin(), cal.end(), c.t, [](Cycle v, const Command& x) { return v < x.t; });
    cal.insert(it, c);
  }

  void prune(Cycle now) {
    auto drop = [&](std::vector<Command>& cal, Cycle gap) {
      auto it = std::lower_bound(cal.begin(), cal.end(), now - gap,
                                 [](const Command& c, Cycle v) { return c.t < v; });
      cal.erase(cal.begin(), it);
    };
    drop(cols, max_col_gap + 1);
    drop(acts, max_act_gap + 1);
  }

  Cycle tCL = 34, tBL = 8, tRCD = 34, tRP = 34, tRAS = 77, tRC = 111, tRTP = 18, tWR = 72, tCWL = 32;
  Cycle tCCDS = 8, tCCDL = 12, tCCDS_WR = 8, tCCDL_WR = 24, tRRDS = 8, tRRDL = 12;
  Cycle tWTRS = 6, tWTRL = 24, tRTW = 14, read_latency = 42;
  Cycle max_col_gap = 64, max_act_gap = 12;
  Cycle rp_adjust = -1, pre_wait = 0, forward_latency = 1, col_squeeze = 4, write_ack = 1;
  bool model_writes = true, starve_reads = true, arrival_gate = true, hit_windows = true;
  bool col_slip = false;
  bool slip_sync = false;
  bool same_cycle_writes = false;
  bool pending_unforced = false;
  int drain_after_reads = 0;
  int drain_block = 0;
  int arrival_model = 0;
  double arrival_quantile = 0.5;
  double drain_pending = 0.0;
  std::size_t pending_limit = std::numeric_limits<std::size_t>::max();  // activated reads counted at most
  std::size_t n_buckets = 8;

  std::size_t ch_idx = 0, rank_idx = 1, bg_idx = 2, bank_idx = 3, row_idx = 4;
  int n_ranks = 1, n_bgs = 8, n_banks = 4;
  std::vector<Bank> banks;
  std::vector<Command> cols;
  std::vector<Command> acts;

  std::deque<PendingWrite> wq;
  std::size_t wq_high = 52, wq_low = 32;
  bool write_mode = false;
  bool forced = false;         // write mode entered through the high watermark
  Cycle forced_at = 0;         // admission time of the write that forced it
  Cycle drain_start = 0;       // earliest time buffered writes may be booked in this write mode
  Cycle drain_exit = -1;       // last booked drain's final ACT: reads resume after it
  Cycle read_horizon = 0;      // latest time a booked read leaves the read queue
  Cycle read_col_horizon = 0;  // latest booked read column command
  Cycle read_act_horizon = 0;  // latest booked read ACT

  std::multiset<Cycle> in_flight;            // predicted completion times of reads not yet returned
  std::vector<OutstandingRead> outstanding;  // the same reads, with the time each left the read queue
  double mean_latency = 76.0;                // running mean of predicted read latency (DRAM cycles)
  double write_share = 0.0;                  // recent fraction of admissions that are writes (EWMA over 256)
  // Gap from a read's admission to the next read's admission, by the number
  // of other reads in flight at the first one (capped). For reads admitted
  // alone, also the next admission relative to the alone read's completion
  // (negative: the next read did not wait for it); for the others, relative
  // to the earliest predicted completion in flight (negative: the burst went
  // on without waiting for the oldest outstanding miss).
  std::array<RecentSamples, kGapBuckets> gaps;
  std::array<RecentSamples, kGapBuckets> after_first;
  RecentSamples alone_after_done;
  int last_k = -1;  // in-flight bucket of the previous read (-1: none yet)
  Cycle last_read = 0;
  Cycle last_done = 0;
  Cycle last_first_done = 0;
};

}  // namespace

extern "C" std::uint32_t ramulator_atomic_model_api_version() {
  return Ramulator::AtomicModel::api_version;
}

extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model() {
  return new ReservationModel;
}
