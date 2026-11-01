#ifndef __MEM_RAMULATOR2_BASE_HH__
#define __MEM_RAMULATOR2_BASE_HH__

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mem/abstract_mem.hh"
#include "params/AbstractMemory.hh"

namespace Ramulator
{
  class IFrontEnd;
  class IMemorySystem;
}

namespace gem5
{

namespace memory
{

class Ramulator2Base : public AbstractMemory
{
  protected:
    class MemorySystemPort : public ResponsePort
    {
      private:
        Ramulator2Base& ramulator2;
        PortID portId;

      public:
        MemorySystemPort(const std::string& _name,
                         Ramulator2Base& _ramulator2,
                         PortID _port_id);

      protected:
        Tick recvAtomic(PacketPtr pkt) override
        {
            return ramulator2.recvAtomic(pkt);
        };
        void recvFunctional(PacketPtr pkt) override
        {
            ramulator2.recvFunctional(pkt);
        };
        bool recvTimingReq(PacketPtr pkt) override
        {
            return ramulator2.recvTimingReq(pkt, portId);
        };
        void recvRespRetry() override
        {
            ramulator2.recvRespRetry(portId);
        };

        AddrRangeList getAddrRanges() const override;
    };

    struct PortState
    {
        bool retryReq = false;
        bool retryResp = false;
        std::deque<PacketPtr> responseQueue;
        EventFunctionWrapper sendResponseEvent;
        std::unique_ptr<Packet> pendingDelete;

        PortState(Ramulator2Base& ramulator2, PortID port_id);
    };

    std::vector<std::unique_ptr<PortState>> portStates;

    std::string ramulator_config;
    Ramulator::IFrontEnd* ramulator2_frontend;
    Ramulator::IMemorySystem* ramulator2_memorysystem;
    bool ramulator2_finalized;

    Tick startTick;
    const bool eventDriven;
    const bool clockEdgeFirst;
    bool serviceStarted = false;
    bool inService = false;
    Tick dramPeriod = 0;
    // Timestamp of the next controller cycle not yet ticked or skipped.
    Tick nextTick = 0;

    struct ServiceStats : public statistics::Group
    {
        statistics::Scalar serviceEvents;
        ServiceStats(statistics::Group* parent);
    } serviceStats;
    struct OutstandingRead
    {
        PacketPtr packet;
        PortID portId;
    };
    using AdmissionToken = std::uint64_t;

    // Exact, bounded completion state: only accepted requests awaiting their
    // Ramulator callback have entries. The monotonic scalar is not a history.
    std::unordered_map<AdmissionToken, OutstandingRead> outstandingReads;
    std::unordered_set<AdmissionToken> outstandingWrites;
    AdmissionToken nextAdmissionToken;
    bool admissionInProgress;
    bool drainSignalDeferred;

    Ramulator2Base(const AbstractMemoryParams& p,
                   const std::string& ramulator_config,
                   size_t num_ports, bool event_driven, bool clock_edge_first);
    ~Ramulator2Base();

    void initRamulator();
    unsigned int nbrOutstanding() const;
    AdmissionToken allocateAdmissionToken();
    void beginAdmission();
    void endAdmission();
    void maybeSignalDrainDone();

    virtual MemorySystemPort& getMemoryPort(PortID port_id) = 0;
    virtual AddrRange getPortRange(PortID port_id) const = 0;
    virtual int getIngressId(PortID port_id) const = 0;

    void accessAndRespond(PacketPtr pkt, PortID port_id);
    void sendResponse(PortID port_id);

    enum class StatsWriteMode
    {
        Snapshot,
        Final
    };
    void writeRamulatorStats(const std::string& path, StatsWriteMode mode);

    void tick();
    void skipIdleCycles();
    void synchronizeClock();
    void scheduleService();
    EventFunctionWrapper tickEvent;

  public:
    DrainState drain() override;

    void startup() override;
    void resetStats() override;
    void preDumpStats() override;

  protected:
    Tick recvAtomic(PacketPtr pkt);
    void recvFunctional(PacketPtr pkt);
    bool recvTimingReq(PacketPtr pkt, PortID port_id);
    void recvRespRetry(PortID port_id);
};

} // namespace memory
} // namespace gem5

#endif // __MEM_RAMULATOR2_BASE_HH__
