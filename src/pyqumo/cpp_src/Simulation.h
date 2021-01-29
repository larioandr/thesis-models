#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>

#include "Statistics.h"
#include "Components.h"
#include "System.h"


struct NodeData {
    SizeDist systemSize;
    SizeDist queueSize;
    SizeDist serverSize;
    VarData delays;
    VarData departures;
    VarData waitTime;
    VarData responseTime;
    unsigned numPacketsGenerated = 0;
    unsigned numPacketsDelivered = 0;
    unsigned numPacketsLost = 0;
    unsigned numPacketsArrived = 0;
    unsigned numPacketsServed = 0;
    unsigned numPacketsDropped = 0;
    double lossProb = 0.0;
    double dropProb = 0.0;
    double deliveryProb = 0.0;

    NodeData();
    NodeData(const NodeData& other);
    explicit NodeData(const NodeRecords& records);
    NodeData& operator=(const NodeData& other);

    std::string text(const std::string& prefix = "") const;
};

struct SimData {
    std::map<int, NodeData> nodeData;
    unsigned numPacketsGenerated = 0;
    double simTime = 0.0;
    double realTimeMs = 0.0;

    SimData();
    SimData(const SimData& other);
    explicit SimData(const Journal& journal, double simTime, double realTime);
    SimData& operator=(const SimData& other);

    std::string text(const std::string& prefix = "") const;
};

SimData simulate_mm1(double arrivalRate, double serviceRate, int queueCapacity = -1, int maxPackets = 10000);

SimData simulate_gg1(const DblFn& arrival, const DblFn& service, int queueCapacity = -1, int maxPackets = 10000);


#endif //SIMULATION_H
