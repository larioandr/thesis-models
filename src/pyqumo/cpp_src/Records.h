#ifndef RECORDS_H_
#define RECORDS_H_

#include "Base.h"
#include "Statistics.h"
#include <map>


class Node;
class Journal;


class Counter : public Object {
public:
    Counter(int initValue = 0);
    Counter(const Counter& counter);
    ~Counter() override;

    Counter& operator=(const Counter& rside);

    inline int get() const { return value_; }
    inline void inc() { value_++; }
    inline void reset(int initValue = 0) { value_ = initValue; }

    std::string toString() const override;
private:
    int value_ = 0;
};


class NodeRecords : public Object {
public:
    explicit NodeRecords(Journal *journal, Node *node, double time = 0.0);
    ~NodeRecords() override;

    inline Journal* getJournal() const { return journal_; }
    inline Node *getNode() const { return node_; }

    inline TimeSizeSeries *systemSize() const { return systemSize_; }
    inline TimeSizeSeries *queueSize() const { return queueSize_; }
    inline TimeSizeSeries *serverSize() const { return serverSize_; }
    inline Series *delays() const { return delays_; }
    inline Series *departures() const { return departures_; }
    inline Series *waitTimes() const { return waitTimes_; }
    inline Series *responseTimes() const { return responseTimes_; }
    inline Counter *numPacketsGenerated() const { return numPacketsGenerated_; }
    inline Counter *numPacketsDelivered() const { return numPacketsDelivered_; }
    inline Counter *numPacketsLost() const { return numPacketsLost_; }
    inline Counter *numPacketsArrived() const { return numPacketsArrived_; }
    inline Counter *numPacketsServed() const { return numPacketsServed_; }
    inline Counter *numPacketsDropped() const { return numPacketsDropped_; }

    void reset(double time = 0.0);
    void commit();
    std::string toString() const override;
private:
    Node *node_;
    Journal *journal_;

    TimeSizeSeries *systemSize_ = nullptr;
    TimeSizeSeries *queueSize_ = nullptr;
    TimeSizeSeries *serverSize_ = nullptr;
    Series *delays_ = nullptr;
    Series *departures_ = nullptr;
    Series* waitTimes_ = nullptr;
    Series* responseTimes_ = nullptr;
    Counter *numPacketsGenerated_ = nullptr;
    Counter *numPacketsDelivered_ = nullptr;
    Counter *numPacketsLost_ = nullptr;
    Counter *numPacketsArrived_ = nullptr;
    Counter *numPacketsServed_ = nullptr;
    Counter *numPacketsDropped_ = nullptr;

    void clean();
    void build(double time);
};


class Journal : public Object {
public:
    explicit Journal(unsigned windowSize = 100, unsigned numMoments = 4, double time = 0.0);
    virtual ~Journal();

    void addNode(Node *node);

    inline unsigned getNumMoments() const { return numMoments_; }
    inline unsigned getWindowSize() const { return windowSize_; }
    inline NodeRecords *getRecords(int address) const { return nodeRecordsMap_.at(address); }
    inline Counter *numPacketsGenerated() const { return numPacketsGenerated_; }
    inline const std::map<int, NodeRecords*> nodeRecords() const { return nodeRecordsMap_; }

    void reset(double time = 0.0);
    void commit();

    std::string toString() const override;

private:
    unsigned windowSize_ = 100;
    unsigned numMoments_ = 4;
    double initTime_ = 0.0;
    std::map<int, NodeRecords*> nodeRecordsMap_;
    Counter *numPacketsGenerated_ = nullptr;
};


#endif //RECORDS_H_
