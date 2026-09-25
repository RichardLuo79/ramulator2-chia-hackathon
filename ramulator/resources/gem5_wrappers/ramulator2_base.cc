#include "mem/ramulator2/ramulator2_base.hh"

#include <fstream>
#include <limits>

#include "base/callback.hh"
#include "base/output.hh"
#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/Ramulator2.hh"
#include "sim/full_system.hh"
#include "sim/system.hh"

// gem5 defines warn as a macro — protect Ramulator headers
#pragma push_macro("warn")
#undef warn

#include "ramulator/base/base.h"
#include "ramulator/base/config.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"
#include "mem/ramulator2/ramulator2_identity.hh"

#pragma pop_macro("warn")

namespace gem5
{

namespace memory
{

Ramulator2Base::Ramulator2Base(const AbstractMemoryParams& p,
                               const std::string& _ramulator_config,
                               size_t num_ports, bool event_driven,
                               bool clock_edge_first) :
    AbstractMemory(p),
    ramulator_config(_ramulator_config),
    ramulator2_frontend(nullptr), ramulator2_memorysystem(nullptr),
    ramulator2_finalized(false),
    startTick(0),
    eventDriven(event_driven),
    clockEdgeFirst(event_driven || clock_edge_first), serviceStats(this),
    nextAdmissionToken(1),
    admissionInProgress(false), drainSignalDeferred(false),
    // Run before ordinary memory traffic, but after CPU switching. Admission
    // also enforces the boundary rule for arrivals at an earlier priority.
    tickEvent([this]{ tick(); }, name(), false,
        clockEdgeFirst ? Event::CPU_Switch_Pri + 1 : Event::Default_Pri)
{
    for (size_t i = 0; i < num_ports; ++i) {
        portStates.push_back(std::make_unique<PortState>(
            *this, static_cast<PortID>(i)));
    }

    registerExitCallback([this]() {
        writeRamulatorStats(
            simout.resolve("ramulator_stats.yaml"),
            StatsWriteMode::Final);
    });
}

Ramulator2Base::ServiceStats::ServiceStats(statistics::Group* parent)
    : statistics::Group(parent),
      ADD_STAT(serviceEvents, statistics::units::Count::get(),
               "Ramulator controller service invocations, including boundary admission")
{ }

Ramulator2Base::~Ramulator2Base()
{
    delete ramulator2_frontend;
    delete ramulator2_memorysystem;
}

void
Ramulator2Base::initRamulator()
{
    Ramulator::ConfigNode config = Ramulator::Config::parse_config_string(ramulator_config);
    ramulator2_frontend = Ramulator::Factory::create_frontend(config);
    ramulator2_memorysystem = Ramulator::Factory::create_memory_system(config);

    ramulator2_frontend->connect_memory_system(ramulator2_memorysystem);
    ramulator2_memorysystem->connect_frontend(ramulator2_frontend);

    // gem5 packet size is passed through to Ramulator; the memory system
    // validates that each request fits in one DRAM transaction.
}

void
Ramulator2Base::startup()
{
    startTick = curTick();
    // Match the legacy wrapper's conversion, including its tick rounding.
    dramPeriod = ramulator2_memorysystem->get_tCK() * sim_clock::as_float::ns;
    fatal_if(dramPeriod == 0, "Ramulator2 DRAM clock period rounds to zero\n");
    nextTick = clockEdge();

    if (FullSystem) {
        // This delayed first tick avoids early FS boot issues observed by the
        // original wrapper, while still supporting checkpoint restores after
        // that point. credits: @sangjae4309
        // please check https://github.com/CMU-SAFARI/ramulator2/pull/79 and
        // https://github.com/sangjae4309/gem5-ramulator2/issues/5 for details
        constexpr Tick fsBootWorkaroundTick = 13121004000177;
        nextTick = curTick() < fsBootWorkaroundTick ?
            fsBootWorkaroundTick : clockEdge();
    }
    serviceStarted = true;
    if (eventDriven)
        scheduleService();
    else
        schedule(tickEvent, nextTick);
}

void
Ramulator2Base::resetStats() {
    synchronizeClock();
    ClockedObject::resetStats();

    if (ramulator2_frontend)
        ramulator2_frontend->reset_stats_recursive();
    if (ramulator2_memorysystem)
        ramulator2_memorysystem->reset_stats_recursive();
}

void
Ramulator2Base::preDumpStats()
{
    ClockedObject::preDumpStats();

    writeRamulatorStats(simout.resolve(
        "ramulator_stats." + std::to_string(curTick()) + ".yaml"),
        StatsWriteMode::Snapshot);
}

void
Ramulator2Base::writeRamulatorStats(const std::string& path,
                                    StatsWriteMode mode)
{
    if (!ramulator2_frontend || !ramulator2_memorysystem)
        return;

    synchronizeClock();

    if (mode == StatsWriteMode::Final) {
        if (!ramulator2_finalized) {
            ramulator2_frontend->finalize();
            ramulator2_memorysystem->finalize();
            ramulator2_finalized = true;
        }
    } else {
        ramulator2_frontend->update_stats_recursive();
        ramulator2_memorysystem->update_stats_recursive();
    }

    std::ofstream ofs(path);
    if (!ofs) {
        fatal("Ramulator2 failed to open stats file %s\n", path.c_str());
    }

    ramulator2_frontend->print_stats(ofs);
    ramulator2_memorysystem->print_stats(ofs);
    ofs.flush();

    if (!ofs) {
        fatal("Ramulator2 failed to write stats file %s\n", path.c_str());
    }
}

void
Ramulator2Base::sendResponse(PortID port_id)
{
    auto& state = *portStates.at(port_id);
    assert(!state.retryResp);
    assert(!state.responseQueue.empty());

    DPRINTF(Ramulator2, "Attempting to send response\n");

    bool success = getMemoryPort(port_id).sendTimingResp(
        state.responseQueue.front());
    if (success) {
        state.responseQueue.pop_front();

        DPRINTF(Ramulator2, "Have %zu read, %zu write, %zu responses outstanding\n",
                outstandingReads.size(), outstandingWrites.size(),
                state.responseQueue.size());

        if (!state.responseQueue.empty() &&
            !state.sendResponseEvent.scheduled()) {
            schedule(state.sendResponseEvent, curTick());
        }

        maybeSignalDrainDone();
    } else {
        state.retryResp = true;

        DPRINTF(Ramulator2, "Waiting for response retry\n");

        assert(!state.sendResponseEvent.scheduled());
    }
}

unsigned int
Ramulator2Base::nbrOutstanding() const
{
    size_t outstanding = outstandingReads.size() + outstandingWrites.size();
    for (const auto& state : portStates) {
        outstanding += state->responseQueue.size();
    }
    panic_if(outstanding > std::numeric_limits<unsigned int>::max(),
             "Ramulator2 outstanding-request count overflow\n");
    return static_cast<unsigned int>(outstanding);
}

Ramulator2Base::AdmissionToken
Ramulator2Base::allocateAdmissionToken()
{
    panic_if(nextAdmissionToken ==
                 std::numeric_limits<AdmissionToken>::max(),
             "Ramulator2 admission-token space exhausted\n");
    return nextAdmissionToken++;
}

void
Ramulator2Base::beginAdmission()
{
    panic_if(admissionInProgress,
             "Ramulator2 does not support nested request admission\n");
    admissionInProgress = true;
}

void
Ramulator2Base::endAdmission()
{
    assert(admissionInProgress);
    admissionInProgress = false;

    if (drainSignalDeferred) {
        drainSignalDeferred = false;
        if (nbrOutstanding() == 0)
            signalDrainDone();
    }
}

void
Ramulator2Base::maybeSignalDrainDone()
{
    if (nbrOutstanding() != 0)
        return;

    // Atomic/coalesced writes may invoke their completion callback inside
    // receive_external_requests(). Defer the signal until accessAndRespond()
    // has either queued a response or installed pendingDelete.
    if (admissionInProgress) {
        drainSignalDeferred = true;
        return;
    }
    signalDrainDone();
}

void
Ramulator2Base::skipIdleCycles()
{
    if (!eventDriven || !serviceStarted || inService ||
        !system()->isTimingMode() || curTick() < nextTick)
        return;

    // Never skip a cycle with work. In particular, if a completion event at
    // this timestamp has not run yet, leave that cycle to the event queue.
    const auto idle = ramulator2_memorysystem->idle_ticks();
    panic_if(idle < 0, "Ramulator2 returned a negative idle interval\n");
    const Tick elapsed = (curTick() - nextTick) / dramPeriod + 1;
    const Tick skipped = std::min(elapsed, static_cast<Tick>(idle));
    if (skipped) {
        ramulator2_memorysystem->fast_forward(skipped);
        nextTick += skipped * dramPeriod;
    }
}

void
Ramulator2Base::synchronizeClock()
{
    if (!clockEdgeFirst || !serviceStarted || inService ||
        !system()->isTimingMode() || curTick() < nextTick)
        return;

    skipIdleCycles();
    if (nextTick <= curTick()) {
        // A request can arrive at an earlier event priority than tickEvent.
        // Finish this edge once, before admission, even in that case. Do not
        // fast-forward across callbacks, or leave a duplicate tick queued.
        panic_if(nextTick != curTick(), "Ramulator2 missed a service edge\n");
        if (tickEvent.scheduled()) {
            panic_if(tickEvent.when() != curTick(),
                     "Ramulator2 boundary service timestamp differs\n");
            deschedule(tickEvent);
        }
        tick();
    }
}

void
Ramulator2Base::scheduleService()
{
    if (!serviceStarted || inService)
        return;

    Tick when = nextTick;
    if (system()->isTimingMode()) {
        const auto idle = ramulator2_memorysystem->idle_ticks();
        panic_if(idle < 0, "Ramulator2 returned a negative idle interval\n");
        if (idle == std::numeric_limits<Ramulator::Clk_t>::max()) {
            if (tickEvent.scheduled())
                deschedule(tickEvent);
            return;
        }
        panic_if(static_cast<Tick>(idle) > (MaxTick - when) / dramPeriod,
                 "Ramulator2 next service time overflow\n");
        when += static_cast<Tick>(idle) * dramPeriod;
    }
    // A zero-idle controller keeps the original per-cycle schedule. Generic
    // DRAM takes the minimum across channels, so mixed controllers are safe.
    panic_if(when < curTick(), "Ramulator2 service would be in the past\n");
    if (!tickEvent.scheduled())
        schedule(tickEvent, when);
    else if (tickEvent.when() != when)
        reschedule(tickEvent, when);
}

void
Ramulator2Base::tick()
{
    ++serviceStats.serviceEvents;
    if (clockEdgeFirst) {
        skipIdleCycles();
        // Advance before callbacks: sendRetryReq can synchronously admit a
        // request. Its clock is already current, and scheduling waits until
        // this service invocation has finished.
        nextTick = curTick() + dramPeriod;
        inService = true;
    }
    // Only tick when it's timing mode
    if (system()->isTimingMode()) {
        ramulator2_memorysystem->tick();

        // is the connected port waiting for a retry, if so check the
        // state and send a retry if conditions have changed
        for (size_t i = 0; i < portStates.size(); ++i) {
            PortID port_id = static_cast<PortID>(i);
            auto& state = *portStates[i];
            if (state.retryReq) {
                state.retryReq = false;
                getMemoryPort(port_id).sendRetryReq();
            }
        }
    }

    if (clockEdgeFirst)
        inService = false;
    if (eventDriven) {
        scheduleService();
    } else {
        schedule(tickEvent, curTick() + dramPeriod);
    }
}

Tick
Ramulator2Base::recvAtomic(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");

    access(pkt);
    return 50000;   // Arbitrary latency of 50ns
}

void
Ramulator2Base::recvFunctional(PacketPtr pkt)
{
    pkt->pushLabel(name());
    functionalAccess(pkt);

    for (auto& state : portStates) {
        for (auto i = state->responseQueue.begin();
             i != state->responseQueue.end(); ++i) {
            pkt->trySatisfyFunctional(*i);
        }
    }

    pkt->popLabel();
}

bool
Ramulator2Base::recvTimingReq(PacketPtr pkt, PortID port_id)
{
    DPRINTF(Ramulator2, "recvTimingReq: request %s addr %#x size %d\n",
            pkt->cmdString(), pkt->getAddr(), pkt->getSize());

    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");

    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "Should only see read and writes at memory controller, "
             "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());

    // we should not get a new request after committing to retry the
    // current one, but unfortunately the CPU violates this rule, so
    // simply ignore it for now
    synchronizeClock();
    auto& state = *portStates.at(port_id);
    if (state.retryReq)
        return false;

    bool enqueue_success = false;
    const int ingress_id = getIngressId(port_id);
    const bool has_context_id = pkt->req->hasContextId();
    const auto context_id = has_context_id ? pkt->req->contextId() : 0;
    const int source_id = ramulator2SourceId(has_context_id, context_id);
    // Hardware prefetches are eligible only when gem5 propagated the
    // originating demand instruction sequence through their lineage.
    const bool has_stable_seq = pkt->req->hasInstSeqNum() &&
                                !pkt->req->isInstFetch();
    const auto inst_seq = has_stable_seq ? pkt->req->getReqInstSeqNum() : 0;
    const bool ramulator_write = !pkt->isRead();
    const auto identity = ramulator2RequestIdentity(
        source_id, has_stable_seq, inst_seq, pkt->getAddr(),
        ramulator_write, pkt->requestorId(),
        ramulator2_memorysystem->get_tx_bytes());

    beginAdmission();
    if (pkt->isRead())
    {
        const AdmissionToken token = allocateAdmissionToken();
        const bool inserted = outstandingReads.emplace(
            token, OutstandingRead{pkt, port_id}).second;
        panic_if(!inserted,
                 "Ramulator2 reused read admission token %llu\n",
                 static_cast<unsigned long long>(token));

        // Generate a Ramulator READ and bind completion to this exact Packet.
        enqueue_success = ramulator2_frontend->
            receive_external_requests(0, pkt->getAddr(), identity.sourceId,
            ingress_id, identity.frontendId, identity.frontendSubId,
            [this, token](Ramulator::Request& req) {
                DPRINTF(Ramulator2, "Read to %ld completed.\n", req.addr);
                auto completion = outstandingReads.find(token);
                panic_if(completion == outstandingReads.end(),
                         "Ramulator2 completed unknown read token %llu\n",
                         static_cast<unsigned long long>(token));
                const PacketPtr completed_packet = completion->second.packet;
                const PortID completed_port = completion->second.portId;
                outstandingReads.erase(completion);
                accessAndRespond(completed_packet, completed_port);
                maybeSignalDrainDone();
            },
            pkt->getSize());

        if (!enqueue_success)
        {
            const size_t erased = outstandingReads.erase(token);
            panic_if(erased != 1,
                     "Ramulator2 rejected read token %llu after completion\n",
                     static_cast<unsigned long long>(token));
            state.retryReq = true;
        }
    } else if (pkt->isWrite()) {
        // Account before admission because Atomic and coalesced writes may
        // complete synchronously inside receive_external_requests().
        const AdmissionToken token = allocateAdmissionToken();
        const bool inserted = outstandingWrites.insert(token).second;
        panic_if(!inserted,
                 "Ramulator2 reused write admission token %llu\n",
                 static_cast<unsigned long long>(token));
        enqueue_success = ramulator2_frontend->
            receive_external_requests(1, pkt->getAddr(), identity.sourceId,
            ingress_id, identity.frontendId, identity.frontendSubId,
            [this, token](Ramulator::Request& req) {
                DPRINTF(Ramulator2, "Write to %ld completed.\n", req.addr);
                const size_t erased = outstandingWrites.erase(token);
                panic_if(erased != 1,
                         "Ramulator2 completed unknown write token %llu\n",
                         static_cast<unsigned long long>(token));
                maybeSignalDrainDone();
            },
            pkt->getSize());

        if (enqueue_success)
        {
            accessAndRespond(pkt, port_id);
        }
        else
        {
            const size_t erased = outstandingWrites.erase(token);
            panic_if(erased != 1,
                     "Ramulator2 rejected write token %llu after completion\n",
                     static_cast<unsigned long long>(token));
            state.retryReq = true;
        }
    } else {
        // keep it simple and just respond if necessary
        accessAndRespond(pkt, port_id);
        endAdmission();
        return true;
    }

    endAdmission();
    if (eventDriven)
        scheduleService();
    return enqueue_success;
}

void
Ramulator2Base::recvRespRetry(PortID port_id)
{
    DPRINTF(Ramulator2, "Retrying\n");

    auto& state = *portStates.at(port_id);
    assert(state.retryResp);
    state.retryResp = false;
    sendResponse(port_id);
}

void
Ramulator2Base::accessAndRespond(PacketPtr pkt, PortID port_id)
{
    DPRINTF(Ramulator2, "Access for address %lld\n", pkt->getAddr());

    bool needsResponse = pkt->needsResponse();

    access(pkt);

    // turn packet around to go back to requestor if response expected
    if (needsResponse) {
        // access already turned the packet into a response
        assert(pkt->isResponse());

        // Assume frontend latency = 0
        Tick time = curTick() + pkt->headerDelay + pkt->payloadDelay;
        // Here we reset the timing of the packet before sending it out.
        pkt->headerDelay = pkt->payloadDelay = 0;

        DPRINTF(Ramulator2, "Queuing response for address %lld\n",
                pkt->getAddr());

        // queue it to be sent back
        auto& state = *portStates.at(port_id);
        state.responseQueue.push_back(pkt);

        // if we are not already waiting for a retry, or are scheduled
        // to send a response, schedule an event
        if (!state.retryResp && !state.sendResponseEvent.scheduled())
            schedule(state.sendResponseEvent, time);
    } else {
        // queue the packet for deletion
        portStates.at(port_id)->pendingDelete.reset(pkt);
    }
}

DrainState
Ramulator2Base::drain()
{
    // check our outstanding reads and writes and if any they need to drain
    return nbrOutstanding() != 0 ? DrainState::Draining : DrainState::Drained;
}

Ramulator2Base::MemorySystemPort::MemorySystemPort(
        const std::string& _name, Ramulator2Base& _ramulator2,
        PortID _port_id)
    : ResponsePort(_name), ramulator2(_ramulator2), portId(_port_id)
{ }

AddrRangeList
Ramulator2Base::MemorySystemPort::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(ramulator2.getPortRange(portId));
    return ranges;
}

Ramulator2Base::PortState::PortState(Ramulator2Base& ramulator2,
                                     PortID port_id)
    : sendResponseEvent(
          [&ramulator2, port_id]{ ramulator2.sendResponse(port_id); },
          ramulator2.name() + ".sendResponse" + std::to_string(port_id))
{ }

} // namespace memory
} // namespace gem5
