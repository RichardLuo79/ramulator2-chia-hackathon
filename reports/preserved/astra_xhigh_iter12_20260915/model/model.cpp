#include "ramulator/controller/atomic_model/api.h"

// Admission-time queueing / max-plus approximation. State consists only of
// resource work, direction, and bounded observed locality/density history. There
// are no pending requests, command events, ticks, completion processing, or
// revised departures. All times and service quantities are DRAM cycles.
namespace {
using namespace Ramulator::AtomicModel;

class ResourceModel final : public Model {
  struct Bank {
    double projected = -1;  // Latest column-time estimate, excluding read return.
    int row[2] = {-1,-1};     // Recent row locality in read and write batches.
    struct Recent { int row=-1; Cycle seen=-1; };
    Recent recent[2][4];      // Bounded locality history, not queued requests.
    double recent_writes = 0; // Writes to this bank available to share row exchange.
    Cycle observed = 0;       // Last local admission for age decay.
    bool used = false;
    bool write = false;       // Last admitted class, for amortized row exchange.
    double column = 0;        // Tail of this bank's column pipeline.
    double class_column[2] = {0,0}; // Work already owed to each access class.
    double close = 0;         // Local active/recovery work tail, excluding bus waits.
  };
  struct Group {
    double projected[2] = {-1,-1}; // Latest transfer-time estimate per class.
    double column = 0;        // Fluid bank-group data-path work, in busy cycles.
    double activation = 0;    // Bank-group activation-power work.
    bool used = false;
    bool write = false;       // Last local direction; defines the paid spacing.
  };
  struct Rank {
    double activation = 0;    // Fluid four-activation-window power budget.
  };
  struct Channel {
    double projected[2] = {-1,-1}; // Latest transfer-time estimate per class.
    double column = 0;        // Fluid channel work; excludes bank preparation stalls.
    bool write = false;       // Direction of the last reserved transfer.
    bool used = false;
    double recent_writes = 0; // Exponentially aged write count over one row cycle.
    Cycle observed = 0;       // Last admission time for analytic age decay.
  };
  Hardware hw;
  std::vector<Bank> banks;
  std::vector<Group> groups;
  std::vector<Rank> ranks;
  std::vector<Channel> channels;
  int ci=-1, ri=-1, gi=-1, bi=-1, rowi=-1;
  int nc=1, nr=1, ng=1, nb=1;
  double bl=1, rcd=1, rp=1, ras=1, rtp=1, wr=1, cwl=1;
  double ccds=1, ccdl=1, rrds=1, rrdl=1, faw=1, wtrs=1, wtrl=1;
  double ccdsw=1, ccdlw=1;
  double latency=1, ingress=1, bus_scale=1, bank_scale=1, batch=1;
  int index(const std::string& name) const {
    auto it=std::find(hw.level_names.begin(),hw.level_names.end(),name);
    return it==hw.level_names.end() ? -1 : int(it-hw.level_names.begin());
  }
  int count(int i) const { return i<0 ? 1 : std::max(1,hw.level_sizes.at(i)); }
  int level(const Access& a,int i,int n) const {
    return i<0 || i>=int(a.levels.size()) ? 0 : std::clamp(a.levels[i],0,n-1);
  }
  double timing(const std::string& name,double fallback) const {
    auto it=hw.timings.find(name);
    return it==hw.timings.end() || it->second<=0 ? fallback : double(it->second);
  }
 public:
  void initialize(Hardware hardware,Parameters& p) override {
    hw=std::move(hardware);
    ci=index("channel"); ri=index("rank"); gi=index("bankgroup");
    bi=index("bank"); rowi=index("row");
    // Accept the capitalization used in configuration-level hierarchy names.
    if(ci<0) ci=index("Channel");
    if(ri<0) ri=index("Rank");
    if(gi<0) gi=index("BankGroup");
    if(bi<0) bi=index("Bank");
    if(rowi<0) rowi=index("Row");
    nc=count(ci); nr=count(ri); ng=count(gi); nb=count(bi);
    channels.assign(nc,{}); ranks.assign(nc*nr,{});
    groups.assign(nc*nr*ng,{}); banks.assign(nc*nr*ng*nb,{});
    latency=std::max<Cycle>(1,hw.read_latency);
    bl=timing("nBL",std::max(1,hw.internal_prefetch_size/2));
    rcd=timing("nRCD",timing("nRCDRD",latency-bl));
    rp=timing("nRP",rcd); ras=timing("nRAS",rcd+rp);
    rtp=timing("nRTP",bl); wr=timing("nWR",rcd);
    cwl=timing("nCWL",latency-bl);
    ccds=timing("nCCDS",bl); ccdl=timing("nCCDL",ccds);
    // Some standards expose a longer write/write data-path recovery.
    ccdsw=timing("nCCDS_WR",timing("nCCDSWR",ccds));
    ccdlw=timing("nCCDL_WR",timing("nCCDLWR",ccdl));
    rrds=timing("nRRDS",bl); rrdl=timing("nRRDL",rrds);
    faw=timing("nFAW",4*rrds);
    wtrs=timing("nWTRS",bl); wtrl=timing("nWTRL",wtrs);
    ingress=p.number("ingress_cycles",1,0,8);
    bus_scale=p.number("bus_service_scale",1,0.5,2);
    bank_scale=p.number("bank_recovery_scale",1,0.5,2);
    // A controller drains buffered writes in groups. The watermark span
    // supplies a physical batch-size estimate, not a traffic-specific fit.
    batch=std::max(1.0,hw.write_capacity*
        (hw.write_high_watermark-hw.write_low_watermark)*
        p.number("write_batch_scale",1,0.1,4));
  }

  Cycle predict(Access a,Cycle now) override {
    int c=level(a,ci,nc), r=c*nr+level(a,ri,nr);
    int g=r*ng+level(a,gi,ng), b=g*nb+level(a,bi,nb);
    int row=rowi<0 || rowi>=int(a.levels.size()) ? 0 : a.levels[rowi];
    bool write=a.type==AccessType::Write;
    auto& bank=banks[b]; auto& group=groups[g];
    auto& rank=ranks[r]; auto& channel=channels[c];
    double start=double(now)+ingress, ready=start, preparation=0;
    // A transfer delayed by shared work can leave bank recovery after the
    // admission-time local work has drained. Once the latest projection is
    // due, estimate that residual without appending its entire shared wait to
    // the bank's fluid workload. Its row and direction are the last admission
    // context below, not observations of actual service. Different-row and
    // opposite-direction future projections remain bypassable; no pending
    // transfer or completion event is stored. This also constrains due row
    // exchanges whose local wait was discounted for batching. Inferred order
    // can disagree with actual issue order. Due-only bounds extend at most
    // one configured spacing/recovery interval past start. The same-context
    // future bound can instead inherit the preceding estimated shared wait;
    // its direct increment over that projection is one column spacing.
    // Closed-loop effects can accumulate, and replacement loses older history.
    double bank_residual=start;
    // Consecutive admissions in the same bank, row and direction have
    // identical row readiness. The older access is a stronger predecessor
    // estimate than an arbitrary other-row or opposite-direction projection:
    // row-hit preference offers no reason to bypass it. Preserve its column
    // spacing even when it is still future. This remains a one-step bound;
    // no request record or event is retained and local fluid tails are unchanged.
    const bool same_context=bank.used && bank.write==write &&
        bank.row[write ? 1 : 0]==row;
    if(bank.used && bank.projected>=0 &&
        (bank.projected<=start || same_context)) {
      if(bank.row[bank.write ? 1 : 0]!=row) {
        const double recovery=bank.write ? cwl+bl+wr : rtp;
        bank_residual=std::max(start,bank.projected+recovery*bank_scale+rp+rcd);
      } else {
        const double spacing=bank.write==write ? (write ? ccdlw : ccdl) :
            (write ? std::max(bl,latency-cwl+2) : cwl+bl+wtrl);
        bank_residual=std::max(start,bank.projected+spacing*bus_scale);
      }
    }
    // The watermark span is only an upper batch estimate. Sparse arrivals
    // cannot supply that many writes in a bank's row-exchange interval.
    // A leaky count estimates writes available to share one exchange; it
    // is a traffic statistic, not a write queue or completion inventory.
    channel.recent_writes*=std::exp(-double(std::max<Cycle>(0,now-channel.observed))/(ras+rp));
    channel.observed=now;
    const double active_batch=std::min(batch,1.0+channel.recent_writes);
    if(write) channel.recent_writes+=1;
    // A bus batch may span many banks. Only writes to this bank can share
    // its PRE/ACT and recovery; other banks cannot amortize local work.
    bank.recent_writes*=std::exp(-double(std::max<Cycle>(0,now-bank.observed))/(ras+rp));
    bank.observed=now;
    // PRE/ACT sharing counts the current access only if it is a write.
    // A read contributes no extra write; the unit floor preserves a full
    // isolated exchange when the aged write count is below one.
    const double bank_batch=std::min(batch,std::max(1.0,bank.recent_writes+(write ? 1.0 : 0.0)));
    // Keep a separate, heuristic backlog-sharing factor for class/row
    // priority. Its unit term represents the arriving service context;
    // the aged write mass represents opportunities to bypass grouped work.
    // It is not a write count or an inferred service order.
    const double backlog_share=std::min(batch,1.0+bank.recent_writes);
    if(write) bank.recent_writes+=1;
    // Bus turnarounds use active_batch; local row work uses bank_batch.
    // Separate class-local row histories describe locality within controller
    // read/write batches, not two physically open rows. Cross-class exchange
    // consumes a bank-local amortized PRE+ACT cost; exact drains are unknowable.
    const int type=write ? 1 : 0;
    const bool exchange=bank.used && bank.write!=write &&
        bank.row[1-type]!=row;
    const double exchange_work=exchange ? (rp+rcd)/bank_batch : 0;
    // FR-FCFS can coalesce interleaved references to a row while bank work
    // is pending. A small recent-row directory approximates that opportunity.
    // History expires after the larger of a row cycle and current local
    // workload delay: a busy bank provides a longer coalescing opportunity.
    // Four entries retain a few competing rows without growing with traffic.
    // Entries contain only observed row tags and last-reference times. They
    // have no request counts, commands, service order, or departure events.
    bool coalesced=false;
    int replace=0;
    for(int i=0;i<4;++i) {
      const auto& h=bank.recent[type][i];
      if(h.row==row && h.seen>=0 && double(now-h.seen)<=std::max(ras+rp,bank.close-double(now)) &&
          bank.close>double(now)) coalesced=true;
      if(h.row==row) { replace=i; break; }
      if(h.seen<bank.recent[type][replace].seen) replace=i;
    }
    bank.recent[type][replace]={row,now};
    if(bank.row[type]!=row && !coalesced) {
      // A row change waits for active-time/data recovery, then restores the
      // bitlines and senses the new row. Different banks overlap this work.
      // Missing class history does not imply a precharged bank: the other
      // class may already have used a different row. A new class-local row
      // pays its full preparation here. exchange_work is only needed below
      // when reusing a context whose preparation was previously charged.
      const int prior_row=bank.row[type]>=0 ? bank.row[type] : bank.row[1-type];
      const double precharge=prior_row>=0 && prior_row!=row ? rp : 0;
      double activate=std::max(start,bank.close)+precharge;
      // Activation work is charged at admission. Waiting on one bank must
      // not reserve the rank's power budget far into the future.
      double power_wait=std::max({0.0,rank.activation-start,group.activation-start});
      activate=std::max(activate,start+power_wait);
      rank.activation=std::max(start,rank.activation)+std::max(rrds,faw/4);
      group.activation=std::max(start,group.activation)+rrdl;
      preparation=rcd+precharge;
      ready=activate+rcd;
      bank.close=activate+ras*bank_scale;
      bank.row[type]=row;
    } else {
      ready=std::max(ready,bank.column);
      if(exchange) {
        // The opposite class's admission-order work, including recovery,
        // is not a mandatory FIFO predecessor under class/row priority.
        // Discount that existing wait with backlog_share. The new PRE+ACT
        // and minimum-active-time costs instead use the write-group count;
        // a read can bypass queued writes without eliminating a row change.
        ready=std::max(bank.class_column[type],
            start+(std::max(ready,bank.close)-start)/backlog_share+exchange_work);
        bank.close=std::max(bank.close,ready+(ras-rcd)/bank_batch);
      }
      preparation=exchange_work;
      bank.row[type]=row;
    }
    const double channel_service=write ? ccdsw : ccds;
    const double group_service=write ? ccdlw : ccdl;
    // Each fluid tail includes the last transfer's same-direction spacing.
    // At a direction boundary replace that already paid spacing with the
    // corresponding cross-direction interval. In particular, subtract the
    // PREVIOUS class's service, not the arriving class's different CCD.
    // A negative adjustment releases recovery needed only by another write;
    // it does not discard a transfer. Divide the boundary adjustment by the
    // estimated batch size because admission switches overcount drain switches.
    double bus_adjust=0, group_adjust=0;
    if(channel.used && write!=channel.write) {
      const double previous=channel.write ? ccdsw : ccds;
      const double crossing=write ? std::max(bl,latency-cwl+2) : cwl+bl+wtrs;
      bus_adjust=(crossing-previous)*bus_scale/active_batch;
    }
    if(group.used && write!=group.write) {
      const double previous=group.write ? ccdlw : ccdl;
      const double crossing=write ? std::max(bl,latency-cwl+2) : cwl+bl+wtrl;
      group_adjust=(crossing-previous)*bus_scale/active_batch;
    }
    // Work-conserving fluid service excludes a bank's preparation stalls.
    // Each group's direction is local, but bus batches can span many groups,
    // so the statistical batch estimate remains channel-wide.
    double bus_start=std::max(start,channel.column+bus_adjust);
    double group_start=std::max(start,group.column+group_adjust);
    double queue_wait=std::max(bus_start-start,group_start-start);
    // Admission-time fluid work can drain during bank preparation. Retain a
    // bounded estimate of the residual bus/group recovery once the most recent
    // projected transfer of either class is due. A future transfer is not a
    // committed FIFO predecessor: other banks/classes can bypass it. This adds
    // only an elapsed-pipeline constraint, not bank stalls to shared demand.
    // Projections are estimates, never completion observations or pending events.
    double residual=start;
    for(int prior=0;prior<2;++prior) {
      const bool prior_write=prior==1;
      const double short_spacing=prior_write==write ? channel_service :
          (write ? std::max(bl,latency-cwl+2) : cwl+bl+wtrs);
      const double long_spacing=prior_write==write ? group_service :
          (write ? std::max(bl,latency-cwl+2) : cwl+bl+wtrl);
      if(channel.projected[prior]>=0 && channel.projected[prior]<=start)
        residual=std::max(residual,channel.projected[prior]+short_spacing*bus_scale);
      if(group.projected[prior]>=0 && group.projected[prior]<=start)
        residual=std::max(residual,group.projected[prior]+long_spacing*bus_scale);
    }
    double column=std::max({ready,start+preparation+queue_wait,residual,bank_residual});
    bank.projected=column;
    channel.projected[type]=column;
    group.projected[type]=column;
    channel.column=bus_start+channel_service*bus_scale;
    channel.used=true; channel.write=write;
    group.column=group_start+group_service*bus_scale;
    group.used=true; group.write=write;
    // Local work is charged from ready, excluding shared waits, to avoid
    // recursively treating shared waiting as occupied bank service. This is
    // fluid accounting, not a physical recovery guarantee: when column > ready,
    // bank spacing/recovery can expire before the projected column operation.
    // Bank residuals repair a part of this omission, including consecutive
    // same-context column spacing. Future row changes and uncertain service
    // order remain fluid; neither this nor the shared estimates are a schedule.
    bank.column=ready+group_service*bus_scale;
    bank.class_column[type]=bank.column;
    bank.used=true; bank.write=write;
    bank.close=std::max(bank.close,ready+
        (write ? cwl+bl+wr : rtp)*bank_scale);
    // Reads depart after the complete read pipeline. Posted-write acknowledgment
    // is estimated at column issue, before the write data pipeline retires.
    // Omitting CWL+BL from this return does not remove their resource-work
    // charges. Those approximate tails are not a guarantee against early reuse,
    // particularly when shared waiting postpones column beyond local ready.
    double departure=column+(write ? 0 : latency);
    return std::max(now+1,Cycle(std::ceil(departure)));
  }
};
}
extern "C" std::uint32_t ramulator_atomic_model_api_version() {
  return Ramulator::AtomicModel::api_version;
}
extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model() {
  return new ResourceModel;
}
