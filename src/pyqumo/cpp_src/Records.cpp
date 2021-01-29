#include "Records.h"
#include "Components.h"
#include <sstream>


// ==========================================================================
// Class Counter
// ==========================================================================
Counter::Counter(int initValue) : value_(initValue) {}
Counter::Counter(const Counter &counter) : value_(counter.value_) {}
Counter::~Counter() = default;

Counter& Counter::operator=(const Counter& rside) {
    value_ = rside.get();
    return *this;
}

std::string Counter::toString() const {
    std::stringstream ss;
    ss << "(Counter: value=" << value_ << ")";
    return ss.str();
}


// ==========================================================================
// Class NodeRecords
// ==========================================================================
NodeRecords::NodeRecords(Journal *journal, Node *node, double time)
: node_(node), journal_(journal) {
    build(time);
}

NodeRecords::~NodeRecords() {
    clean();
}

void NodeRecords::reset(double time) {
    clean();
    build(time);
}

void NodeRecords::commit() {
    if (delays_) delays_->commit();
    if (departures_) departures_->commit();
    if (waitTimes_) waitTimes_->commit();
    if (responseTimes_) responseTimes_->commit();
}

void NodeRecords::clean() {
    delete systemSize_;
    systemSize_ = nullptr;
    delete queueSize_;
    queueSize_ = nullptr;
    delete serverSize_;
    serverSize_ = nullptr;
    delete delays_;
    delays_ = nullptr;
    delete departures_;
    departures_ = nullptr;
    delete waitTimes_;
    waitTimes_ = nullptr;
    delete responseTimes_;
    responseTimes_ = nullptr;
    delete numPacketsGenerated_;
    numPacketsGenerated_ = nullptr;
    delete numPacketsDelivered_;
    numPacketsDelivered_ = nullptr;
    delete numPacketsLost_;
    numPacketsLost_ = nullptr;
    delete numPacketsArrived_;
    numPacketsArrived_ = nullptr;
    delete numPacketsServed_;
    numPacketsServed_ = nullptr;
    delete numPacketsDropped_;
    numPacketsDropped_ = nullptr;
}

void NodeRecords::build(double time) {
    auto numMoments = journal_->getNumMoments();
    auto windowSize = journal_->getWindowSize();

    systemSize_ = new TimeSizeSeries(time, node_->size());
    queueSize_ = new TimeSizeSeries(time, node_->getQueue()->size());
    serverSize_ = new TimeSizeSeries(time, node_->getServer()->size());
    delays_ = new Series(numMoments, windowSize);
    departures_ = new Series(numMoments, windowSize);
    waitTimes_ = new Series(numMoments, windowSize);
    responseTimes_ = new Series(numMoments, windowSize);
    numPacketsGenerated_ = new Counter(0);
    numPacketsDelivered_ = new Counter(0);
    numPacketsLost_ = new Counter(0);
    numPacketsArrived_ = new Counter(0);
    numPacketsServed_ = new Counter(0);
    numPacketsDropped_ = new Counter(0);
}

std::string NodeRecords::toString() const {
    std::stringstream ss;
    ss << "(NodeRecords: address=" << node_->getAddress()
        << ", systemSize=" << systemSize_->toString()
        << ", queueSize=" << queueSize_->toString()
        << ", serverSize=" << serverSize_->toString()
        << ", delays=" << delays_->toString()
        << ", departures=" << departures_->toString()
        << ", waitTimes=" << waitTimes_->toString()
        << ", responseTimes=" << responseTimes_->toString()
        << ", numPacketsGenerated=" << numPacketsGenerated_->toString()
        << ", numPacketsDelivered=" << numPacketsDelivered_->toString()
        << ", numPacketsLost=" << numPacketsLost_->toString()
        << ", numPacketsArrived=" << numPacketsArrived_->toString()
        << ", numPacketsServed=" << numPacketsServed_->toString()
        << ", numPacketsDropped=" << numPacketsDropped_->toString()
        << ")";
    return ss.str();
}


// ==========================================================================
// Class Journal
// ==========================================================================
Journal::Journal(unsigned int windowSize, unsigned int numMoments, double time)
: windowSize_(windowSize), numMoments_(numMoments), initTime_(time), numPacketsGenerated_(new Counter(0)) {}

Journal::~Journal() {
    for (auto& kv: nodeRecordsMap_) {
        delete kv.second;
    }
    delete numPacketsGenerated_;
}

void Journal::addNode(Node *node) {
    auto address = node->getAddress();
    auto it = nodeRecordsMap_.find(address);
    if (it != nodeRecordsMap_.end()) {
        delete it->second;
    }
    nodeRecordsMap_[address] = new NodeRecords(this, node, initTime_);
}

void Journal::reset(double time) {
    for (auto& addrRecordsPair: nodeRecordsMap_) {
        addrRecordsPair.second->reset(time);
    }
    delete numPacketsGenerated_;
    numPacketsGenerated_ = new Counter(0);
}

void Journal::commit() {
    for (auto& addrRecordsPair: nodeRecordsMap_) {
        addrRecordsPair.second->commit();
    }
}

std::string Journal::toString() const {
    std::stringstream ss;
    ss << "(Journal: windowSize=" << windowSize_
        << ", numMoments=" << numMoments_
        << ", initTime=" << initTime_
        << ", records={";
    bool first = true;
    for (auto& addrRecordsPair: nodeRecordsMap_) {
        if (!first) ss << ", "; else first = false;
        ss << addrRecordsPair.first << ": " << addrRecordsPair.second->toString();
    }
    ss << "}";
    return ss.str();
}
