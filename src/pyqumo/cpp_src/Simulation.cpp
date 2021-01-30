#include "Simulation.h"
#include <chrono>
#include <random>
#include <iostream>


// ==========================================================================
// Class NodeData
// ==========================================================================
NodeData::NodeData() = default;

NodeData::NodeData(const NodeData &other)
: systemSize(other.systemSize),
queueSize(other.queueSize),
serverSize(other.serverSize),
delays(other.delays),
departures(other.departures),
waitTime(other.waitTime),
responseTime(other.responseTime),
lossProb(other.lossProb),
dropProb(other.dropProb),
deliveryProb(other.deliveryProb),
numPacketsGenerated(other.numPacketsGenerated),
numPacketsDelivered(other.numPacketsDelivered),
numPacketsLost(other.numPacketsLost),
numPacketsArrived(other.numPacketsArrived),
numPacketsServed(other.numPacketsServed),
numPacketsDropped(other.numPacketsServed)
{
}

NodeData::NodeData(const NodeRecords& records)
: systemSize(records.systemSize()->getSizeDist()),
queueSize(records.queueSize()->getSizeDist()),
serverSize(records.serverSize()->getSizeDist()),
delays(*records.delays()),
departures(*records.departures()),
waitTime(*records.waitTimes()),
responseTime(*records.responseTimes()),
numPacketsGenerated(records.numPacketsGenerated()->get()),
numPacketsDelivered(records.numPacketsDelivered()->get()),
numPacketsLost(records.numPacketsLost()->get()),
numPacketsArrived(records.numPacketsArrived()->get()),
numPacketsServed(records.numPacketsServed()->get()),
numPacketsDropped(records.numPacketsDropped()->get())
{
    unsigned numPacketsProcessed = numPacketsDelivered + numPacketsLost;
    lossProb = numPacketsProcessed != 0 ? static_cast<double>(numPacketsLost) / numPacketsProcessed : 0.0;
    dropProb = numPacketsArrived != 0 ? static_cast<double>(numPacketsDropped) / numPacketsArrived : 0.0;
    deliveryProb = 1.0 - lossProb;
}

NodeData& NodeData::operator=(const NodeData &other) = default;

std::string NodeData::text(const std::string& prefix) const {
    std::stringstream ss;
    ss << prefix << "- systemSize: " << systemSize.toString() << std::endl;
    ss << prefix << "- queueSize: " << queueSize.toString() << std::endl;
    ss << prefix << "- serverSize: " << serverSize.toString() << std::endl;
    ss << prefix << "- delays: " << delays.toString() << std::endl;
    ss << prefix << "- departures: " << departures.toString() << std::endl;
    ss << prefix << "- waitTime: " << waitTime.toString() << std::endl;
    ss << prefix << "- responseTime: " << responseTime.toString() << std::endl;
    ss << prefix << "- numPacketsGenerated: " << numPacketsGenerated << std::endl;
    ss << prefix << "- numPacketsDelivered: " << numPacketsDelivered << std::endl;
    ss << prefix << "- numPacketsLost: " << numPacketsLost << std::endl;
    ss << prefix << "- numPacketsArrived: " << numPacketsArrived << std::endl;
    ss << prefix << "- numPacketsServed: " << numPacketsServed << std::endl;
    ss << prefix << "- numPacketsDropped: " << numPacketsDropped << std::endl;
    ss << prefix << "- lossProb: " << lossProb << std::endl;
    ss << prefix << "- dropProb: " << dropProb << std::endl;
    ss << prefix << "- deliveryProb: " << deliveryProb << std::endl;
    return ss.str();
}


// ==========================================================================
// Class SimData
// ==========================================================================
SimData::SimData() = default;

SimData::SimData(const SimData& other) {
    operator=(other);
}

SimData::SimData(const Journal &journal, double simTime, double realTime) {
    for (auto& addrNodeRec: journal.nodeRecords()) {
        auto address = addrNodeRec.first;
        auto records = addrNodeRec.second;
        nodeData[address] = NodeData(*records);
    }
    numPacketsGenerated = journal.numPacketsGenerated()->get();
    this->simTime = simTime;
    this->realTimeMs = realTime;
}

SimData& SimData::operator=(const SimData &other) {
    nodeData.clear();
    for (auto& kv: other.nodeData) {
        nodeData[kv.first] = kv.second;
    }
    numPacketsGenerated = other.numPacketsGenerated;
    simTime = other.simTime;
    realTimeMs = other.realTimeMs;
    return *this;
}

std::string SimData::text(const std::string &prefix) const {
    std::stringstream ss;
    ss << prefix << "SimData" << std::endl;
    ss << prefix << "- numPacketsGenerated: " << numPacketsGenerated << std::endl;
    ss << prefix << "- simTime: " << simTime << std::endl;
    ss << prefix << "- realTimeMs: " << realTimeMs << std::endl;
    ss << prefix << "- nodeData:" << std::endl;
    for (auto& kv: nodeData) {
        ss << prefix << "\t" << kv.first << ":" << std::endl;
        ss << kv.second.text(prefix + "\t");
    }
    return ss.str();
}


// ==========================================================================
// Simulation functions
// ==========================================================================
SimData simulate_mm1(double arrivalRate, double serviceRate, int queueCapacity, int maxPackets) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto gen = std::default_random_engine(seed);
    struct Context { std::default_random_engine *gen = nullptr; std::exponential_distribution<double> fn; };
    Context arrivalContext = { &gen, std::exponential_distribution<double>(arrivalRate) };
    Context serviceContext = { &gen, std::exponential_distribution<double>(serviceRate) };
    auto intervalBuilder = [](Context *ctx) {
        return ContextFunctor([](void *ctx) {
            auto ctx_ = static_cast<Context*>(ctx);
            return ctx_->fn(*(ctx_->gen));
        }, ctx);
    };
    auto arrival = intervalBuilder(&arrivalContext);
    auto service = intervalBuilder(&serviceContext);
    return simulate_gg1(arrival, service, queueCapacity, maxPackets);
}


SimData simulate_gg1(const DblFn& arrival, const DblFn& service, int queueCapacity, int maxPackets) {
    auto startedAt = std::chrono::system_clock::now();
    auto network = buildOneHopeNetwork(arrival, service, queueCapacity);
    auto journal = new Journal;
    for (auto& addrNodePair: network->getNodes()) {
        journal->addNode(addrNodePair.second);
    }
    auto system = new System;

    // Execute main loop
    runMainLoop(network, system, journal, maxPackets);
    journal->commit();

    // Build node data
    double realTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - startedAt).count();
    auto simData = SimData(*journal, system->getTime(), realTimeMs);

    // Clear
    delete network;
    delete journal;
    delete system;

    return simData;
}
