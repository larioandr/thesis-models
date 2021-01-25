#ifndef MM1N_H_
#define MM1N_H_

#include <vector>
#include <string>

struct cStatistics {
    double avg = 0.0;
    double std = 0.0;
    double var = 0.0;
    size_t count = 0;

    virtual ~cStatistics();
};

struct cDiscreteStatistics : public cStatistics {
    std::vector<double> pmf;
    virtual ~cDiscreteStatistics();
};


struct cResults {
    cDiscreteStatistics systemSize;
    cDiscreteStatistics queueSize;
    cDiscreteStatistics busy;
    double lossProb;
    cStatistics departures;
    cStatistics responseTime;
    cStatistics waitTime;

    std::string tabulate() const;
};

cResults *cSimulateMm1n(double arrivalRate, double serviceRate, int queueCapacity, int maxPackets);

#endif
