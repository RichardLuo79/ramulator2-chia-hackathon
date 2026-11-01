#include "ramulator/controller/atomic_model/api.h"

// Admission-order transaction service envelopes. No command objects, command
// selection, pending request queue, completion callback, or simulation tick.
namespace {
using namespace Ramulator::AtomicModel;
class ServiceEnvelope final : public Model {
  struct RecentRow { long long row=-1; double seen=-1e30; };
  struct Bank { RecentRow recent[8]; long long row = -1; long long class_row[2] = {-1,-1}; double ready = -1e30; double read_run_work = 0; double last_arrival=-1e30, last_row_cost=0; int last_kind = -1; };
  struct Channel { double work = 0, transfer_work = 0; bool mixed = false; };
  Hardware hw;
  std::vector<Bank> banks;
  std::vector<Channel> channels;
  std::vector<double> groups;
  std::vector<double> activation_work;
  std::vector<double> group_activation_work;
  int rank_level=0, ranks_per_channel=1;
  double activation_service=1, group_activation_service=1;
  int bank_level=0, row_level=0, channel_level=0, group_level=0;
  int banks_per_channel=1, groups_per_channel=1, banks_per_group=1;
  double base=1, rcd=1, rp=1, rc=1, ccd=1, burst=1, rtp=1, write_recovery=1, write_to_read=1;
  double bank_scale=1, bus_scale=1, overhead=1, write_tax=0;
  double timing(const char* key, double fallback) const {
    auto it=hw.timings.find(key);
    return it==hw.timings.end()?fallback:double(it->second);
  }
 public:
  void initialize(Hardware hardware, Parameters& p) override {
    hw=std::move(hardware);
    bank_level=int(hw.level_index("Bank"));
    row_level=int(hw.level_index("Row"));
    channel_level=int(hw.level_index("Channel"));
    group_level=int(hw.level_index("BankGroup"));
    rank_level=int(hw.level_index("Rank"));
    ranks_per_channel=hw.level_sizes[rank_level];
    activation_work.assign(hw.level_sizes[channel_level]*ranks_per_channel,0);
    banks_per_group=hw.level_sizes[bank_level];
    for(int i=channel_level+1;i<=bank_level;++i)
      banks_per_channel*=hw.level_sizes[i];
    groups_per_channel=banks_per_channel/banks_per_group;
    banks.resize(hw.level_sizes[channel_level]*banks_per_channel);
    groups.resize(hw.level_sizes[channel_level]*groups_per_channel,0);
    group_activation_work.assign(groups.size(),0);
    channels.resize(hw.level_sizes[channel_level]);
    base=hw.read_latency;
    rcd=timing("nRCD",base*0.8);
    rp=timing("nRP",rcd);
    rc=timing("nRC",rcd+rp+rcd);
    ccd=timing("nCCDL",timing("nCCD_L",timing("nBL",8)));
    burst=timing("nBL",8);
    activation_service=std::max(timing("nRRDS",timing("nRRD_S",burst)),timing("nFAW",4*burst)/4.0);
    group_activation_service=timing("nRRDL",timing("nRRD_L",activation_service));
    rtp=timing("nRTP",ccd);
    write_recovery=timing("nCWL",base-burst)+burst+timing("nWR",rcd*2)+rp+rcd;
    write_to_read=timing("nCWL",base-burst)+burst+timing("nWTRL",timing("nWTR_L",ccd));
    bank_scale=p.number("bank_scale",1,0.25,2);
    bus_scale=p.number("bus_scale",1,0.25,3);
    overhead=p.number("admission_overhead",1,0,8);
    // Mean direction-switch work per write, amortized across a small drain.
    double batch=p.number("write_batch",12,1,128);
    write_tax=(timing("nRTW",base)+timing("nWTRL",timing("nWTR_L",ccd)))/batch;
  }
  Cycle predict(Access a, Cycle now) override {
    int ch=a.levels[channel_level], bi=ch;
    for(int i=channel_level+1;i<=bank_level;++i)
      bi=bi*hw.level_sizes[i]+a.levels[i];
    int gi=bi/banks_per_group;
    Bank& b=banks[bi];
    Channel& c=channels[ch];
    // Provenance of the current fluid busy period, not a write queue.
    if(c.work<=now) c.mixed=false;
    bool cold=b.row<0, hit=b.row==a.levels[row_level];
    // For an established-row conflict the admission bookkeeping can overlap
    // the initial precharge step. Cold activation and column hits still pay
    // the full admission overhead. This is an observed transaction-path
    // convention, not an emulated command issue time.
    double row_cost=hit?0:(rcd+(cold?0:std::max(0.0,rp-overhead)));
    int kind=a.type==AccessType::Write?1:0;
    // Separate read/write locality approximates the controller's batching:
    // alternating admission types need not imply a physical row switch each
    // time. This is only a service-demand estimate; no requests are retained.
    bool class_hit=hit || b.class_row[kind]==a.levels[row_level];
    double backlog=std::max(0.0,b.ready-now);
    bool batch_hit=false;
    int slot=0;
    for(int i=0;i<8;++i) {
      if(b.recent[i].row==a.levels[row_level]) {
        batch_hit=backlog>ccd && now-b.recent[i].seen < std::max(rc,backlog);
        slot=i; break;
      }
      if(b.recent[i].seen<b.recent[slot].seen) slot=i;
    }
    b.recent[slot]={a.levels[row_level],double(now)};
    // The minimum active-row interval has partly elapsed during a read
    // column run. Its next conflict still needs read recovery, precharge,
    // and activation, whether the next access is a read or a write. The
    // preceding access type determines recovery. A preceding write retains
    // full row-cycle demand here and receives its recovery floor below.
    double row_service=rc;
    if(b.last_kind==0)
      row_service=std::max(rc-b.read_run_work,rtp+rp+rcd);
    double service=(class_hit||batch_hit)?(kind?2*ccd:ccd):row_service;
    // Sparse conflicting accesses encounter the preceding write's data and
    // cell-restoration tail. With queued work, admission order is a weaker
    // proxy for service order: retain a decaying recovery contribution as
    // backlog exceeds the intrinsic startup path. This is an aggregate service
    // estimate, not a scheduled write or a command-legality timestamp.
    if(b.last_kind==1 && !hit) {
      double in_service=rc/(rc+std::max(0.0,backlog-row_cost));
      // A newly admitted write may still be bypassed by a ready read.
      // Its age relative to the precharge portion of its intrinsic row path
      // estimates whether its bank work has begun. A row-hit write has no
      // row startup and can begin after one admission cycle; other writes
      // retain a burst-length uncertainty ramp. These are historical features,
      // not observed command issue times.
      double age=double(now)-b.last_arrival;
      double started=std::clamp((age-std::max(0.0,b.last_row_cost-rcd))
                               /std::max(1.0,b.last_row_cost==0?1.0:burst),0.0,1.0);
      service=std::max(service,in_service*started*write_recovery);
    }
    // A read following a same-row write still waits for write data and
    // write-to-read bus recovery; row reuse eliminates precharge, not this
    // direction turnaround. Under deep backlog admission order is uncertain.
    if(kind==0 && b.last_kind==1 && hit && now>b.last_arrival) {
      double confidence=std::clamp(1.0-std::max(0.0,backlog-row_cost)/rc,0.0,1.0);
      service=std::max(service,confidence*write_to_read);
    }
    b.read_run_work=(kind==0 && b.last_kind==0 && hit)
      ? std::min(rc,b.read_run_work+std::max(ccd,double(now)-b.ready)) : 0;
    b.last_arrival=double(now);
    b.last_row_cost=row_cost;
    b.last_kind=kind;
    b.class_row[kind]=a.levels[row_level];
    // Bank envelope is a request-level workload estimate. The newest admitted
    // row is a locality proxy, not a claim about an oracle's open row.
    // Rank work estimates activation throughput and the average four-
    // activation power constraint. It is charged only for estimated new
    // rows; no ACT commands or sliding command windows are represented.
    double activation_wait=0;
    if(!class_hit && !batch_hit) {
      double& aw=activation_work[ch*ranks_per_channel+a.levels[rank_level]];
      // A bank group shares a longer activation recovery path. Two fluid
      // workloads estimate rank power and local group demand independently;
      // neither records activation commands or a command window.
      double& gaw=group_activation_work[gi];
      activation_wait=std::max({0.0,aw-now,gaw-now});
      aw=std::max(double(now),aw)+activation_service;
      gaw=std::max(double(now),gaw)+group_activation_service;
    }
    double ready=std::max(double(now)+row_cost+activation_wait,
                          b.ready+service*bank_scale);
    // Independently draining work avoids serializing bank activation behind
    // the channel data path. These clocks summarize work, not future commands.
    // In a read-only fluid busy period, a ready latest-row read can use
    // transfer bandwidth while another bank prepares a row. If any write
    // contributed to this busy period, preserve the full shared envelope:
    // direction recovery and write retention make overlap less certain.
    // This records aggregate workload provenance, not future bus slots.
    double shared=(kind==0 && hit && ready<=now && !c.mixed)
      ? c.transfer_work : c.work;
    double channel_wait=std::max(0.0,shared-now);
    double group_wait=std::max(0.0,groups[gi]-now);
    double wait=std::max({ready-now,channel_wait,group_wait});
    // A transfer cannot consume channel bandwidth before its own row
    // access path has traversed activation/precharge. Preserve that startup
    // in the channel workload, but do not propagate a bank's queue backlog
    // to unrelated banks (which would impose artificial head-of-line blocking).
    double transfer_service=(burst+(kind?write_tax:0))*bus_scale;
    c.work=std::max(double(now)+row_cost,c.work)+transfer_service;
    c.transfer_work=std::max(double(now),c.transfer_work)+transfer_service;
    if(kind) c.mixed=true;
    groups[gi]=std::max(double(now),groups[gi])+ccd*(kind?2:1)*bus_scale;
    // Shared transfer pressure also occupies this bank's transaction path.
    // Feed back only this admission's aggregate service position; there is
    // no pending work list and previous predictions remain immutable.
    // While a read's column waits, its row has also aged. Credit this
    // elapsed time so the next conflict does not pay minimum active time twice.
    if(kind==0)
      b.read_run_work=std::min(rc,b.read_run_work+std::max(0.0,double(now)+wait-ready));
    b.ready=double(now)+wait;
    b.row=a.levels[row_level];
    return now+std::max<Cycle>(1,Cycle(std::ceil(base+overhead+wait)));
  }
};
}
extern "C" std::uint32_t ramulator_atomic_model_api_version() {
  return Ramulator::AtomicModel::api_version;
}
extern "C" Ramulator::AtomicModel::Model* ramulator_create_atomic_model() {
  return new ServiceEnvelope;
}
