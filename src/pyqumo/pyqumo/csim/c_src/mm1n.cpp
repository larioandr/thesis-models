#include <iostream>
#include <vector>
#include <utility>
#include <queue>
#include <string>
#include <sstream>
#include <limits>
#include <random>
#include <chrono>
#include <cstdio>
#include <algorithm>

#include "mm1n.h"


struct cPacket {
    double createdAt = 0.0;
    double serviceStartedAt = 0.0;
    double departedAt = 0.0;
    bool dropped = false;
    bool served = false;
};


typedef std::pair<double, int> TimeValue;


struct Records {
    std::vector<cPacket*> packets;
    std::vector<TimeValue> systemSize;

    double __busyDuration = 0.0;
    double __lastServerUpdate = 0.0;

    virtual ~Records() {
        for (std::vector<cPacket*>::iterator it = packets.begin(); it != packets.end(); ++it) {
            delete *it;
        }
        packets.clear();
    }

    void addSystemSize(double time, int value) {
        systemSize.push_back(TimeValue(time, value));
    }
};


cStatistics::~cStatistics() {}
cDiscreteStatistics::~cDiscreteStatistics() {}

std::string cResults::tabulate() const {
    std::stringstream ss;
    std::vector<std::string> titles = {
        "System size mean",
        "System size std.dev.",
        "System size PMF",
        "Queue size mean",
        "Queue size std.dev.",
        "Queue size PMF",
        "Utilization mean",
        "Utilization std.dev.",
        "Utilization PMF",
        "Loss probability",
        "Departure interval mean",
        "Response time mean",
        "Waiting time mean"
    };

    auto vectorToString = [](const std::vector<double>& array) -> std::string {
        std::stringstream ss_;
        ss_ << "[";
        for (int i = 0; i < array.size(); i++) {
            if (i > 0) {
                ss_ << ", ";
            }
            ss_ << array[i];
        }
        ss_ << "]";
        return ss_.str();
    };

    std::vector<std::string> values = {
        std::to_string(systemSize.avg),
        std::to_string(systemSize.std),
        vectorToString(systemSize.pmf),
        std::to_string(queueSize.avg),
        std::to_string(queueSize.std),
        vectorToString(queueSize.pmf),
        std::to_string(busy.avg),
        std::to_string(busy.std),
        vectorToString(busy.pmf),
        std::to_string(lossProb),
        std::to_string(departures.avg),
        std::to_string(responseTime.avg),
        std::to_string(waitTime.avg)
    };

    auto numRows = titles.size();
    for (int rowIndex = 0; rowIndex < numRows; ++rowIndex) {
        ss << titles[rowIndex] << ": " << values[rowIndex] << "\n";
    }
    return ss.str();
}


class ExponentialDistribution {
public:
    ExponentialDistribution(double rate, std::default_random_engine *gen) : gen(gen) {
        dist = new std::exponential_distribution<double>(rate);
    }

    ~ExponentialDistribution() {
        delete dist;
    }

    double next() const {
        return (*dist)(*gen);
    }

    inline double rate() const {
        return dist->lambda();
    }
private:
    std::exponential_distribution<double> *dist = nullptr;
    std::default_random_engine *gen;
};


class cParams {
public:
    cParams(ExponentialDistribution *arrivals,
            ExponentialDistribution *services,
            size_t queueCapacity,
            size_t maxPackets = 100000)
            : arrivals(arrivals), services(services),
            queueCapacity(queueCapacity), maxPackets(maxPackets)
    {
        // nop
    }

    inline const ExponentialDistribution& getArrivals() const { return *arrivals; }
    inline const ExponentialDistribution& getServices() const { return *services; }
    inline double getArrivalRate() const { return arrivals->rate(); }
    inline double getServiceRate() const { return services->rate(); }
    inline size_t getQueueCapacity() const { return queueCapacity; }
    inline size_t getMaxPackets() const { return maxPackets; }

  private:
    ExponentialDistribution *arrivals;
    ExponentialDistribution *services;
    size_t queueCapacity;
    size_t maxPackets;
};


class Queue {
private:
    size_t capacity;
    std::queue<cPacket*> queue;
public:
    Queue(size_t capacity) : capacity(capacity) {}

    int push(cPacket *packet) {
        if (queue.size() < capacity) {
            queue.push(packet);
            return 1;
        }
        return 0;
    }

    cPacket *pop() {
        if (queue.empty()) {
            return nullptr;
        }
        cPacket *packet = queue.front();
        queue.pop();
        return packet;
    }

    size_t getSize() const {
        return queue.size();
    }

    size_t getCapacity() const {
        return capacity;
    }
};


class Server {
    private:
        cPacket *packet;
    public:
        Server() : packet(nullptr) {}

        bool busy() const { return packet != nullptr; }
        bool ready() const { return packet == nullptr; }
        int getSize() const { return packet ? 1 : 0; }

        void put(cPacket *packet) {
            this->packet = packet;
        }

        cPacket *pop() {
            auto result = this->packet;
            this->packet = nullptr;
            return result;
        }
};


enum Event {STOP, ARRIVAL, SERVICE_END};


class System {
public:
    System(const cParams& params) {
        queue = new Queue(params.getQueueCapacity());
        server = new Server;

        time = 0.0;
        for (size_t i = 0; i < 2; i++) {
            eventTimes[i] = 0.0;
        }
        nextServiceEnd = nullptr;
        nextArrival = nullptr;
        stop_ = false;
    }

    ~System() {
        delete queue;
        delete server;
    }

    size_t getSize() const {
        return queue->getSize() + server->getSize();
    }

    Event nextEvent() {
        if (stop_) {
            return STOP;
        }

        if (nextArrival != nullptr) {
            if (nextServiceEnd == nullptr || *nextArrival < *nextServiceEnd) {
                this->time = *nextArrival;
                this->nextArrival = nullptr;
                return ARRIVAL;
            }
        }

        if (nextServiceEnd != nullptr) {
            this->time = *nextServiceEnd;
            this->nextServiceEnd = nullptr;
            return SERVICE_END;
        }

        return STOP;
    }

    void schedule(Event event, double interval) {
        if (interval < 0) {
            throw "interval < 0 in schedule()";
        }
        int offset = -1;
        if (event == ARRIVAL) {
            offset = 0;
            nextArrival = &eventTimes[offset];
        } else if (event == SERVICE_END) {
            offset = 1;
            nextServiceEnd = &eventTimes[offset];
        } else {
            throw "unrecognized event in schedule()";
        }
        eventTimes[offset] = time + interval;
    }

    void stop() { stop_ = true; }

    bool stopped() const { return stop_; }
    double getTime() const { return time; }
    Queue *getQueue() const { return queue; }
    Server *getServer() const { return server; }

private:
    double eventTimes[2];
    double *nextServiceEnd;
    double *nextArrival;
    bool stop_;
    double time;
    Server *server;
    Queue *queue;
};

const char *getEventName(Event event) {
    if (event == ARRIVAL)
        return "ARRIVAL";
    if (event == SERVICE_END)
        return "SERVICE_END";
    if (event == STOP)
        return "STOP";
    throw "unrecognized event";
}


//////////////////////////////////////
// SIMULATION ROUTINES
//////////////////////////////////////
void handleArrival(const cParams& params, System *system, Records *records) {
    // If generated enough packets, tell system to stop:
    size_t numcPacketsBuilt = records->packets.size();
    if (numcPacketsBuilt >= params.getMaxPackets()) {
        system->stop();
    }

    // Create the packet:
    auto now = system->getTime();
    auto packet = new cPacket;
    packet->createdAt = now;
    records->packets.push_back(packet);

    // Process the newly created packet:
    auto server = system->getServer();
    if (server->ready()) {
        // If server is ready, serve the packet immediately!
        server->put(packet);
        packet->serviceStartedAt = now;
        system->schedule(SERVICE_END, params.getServices().next());
        records->addSystemSize(now, system->getSize());

        records->__lastServerUpdate = now;
    } else {
        // If server is busy, queue it!
        auto queue = system->getQueue();
        if (queue->push(packet)) {
            records->addSystemSize(now, system->getSize());
        } else {
            packet->dropped = true;
        }
    }

    // Schedule next arrival
    system->schedule(ARRIVAL, params.getArrivals().next());
}


void handleServiceEnd(const cParams& params, System *system, Records *records) {
    // Current packet: "finish him"! Mark as departed and served.
    auto now = system->getTime();
    auto packet = system->getServer()->pop();
    if (packet == nullptr) {
        throw "unexpected empty server in handleServiceEnd()";
    }
    packet->served = true;
    packet->departedAt = now;

    // Start serving the next packet from queue.
    packet = system->getQueue()->pop();
    if (packet) {
        system->getServer()->put(packet);
        packet->serviceStartedAt = now;
        system->schedule(SERVICE_END, params.getServices().next());
    } else {
        records->__busyDuration += now - records->__lastServerUpdate;
    }

    // Record new system size.
    records->addSystemSize(now, system->getSize());
}


cDiscreteStatistics buildTimeValueStatistics(const std::vector<TimeValue>& records) {
    // 1) Find maximum value:
    size_t maxValue = 0;
    for (auto& tv: records) {
        size_t value = static_cast<size_t>(tv.second);
        if (value > maxValue) {
            maxValue = value;
        }
    }

    // 2) Estimate rates:
    std::vector<double> rates(maxValue + 1, 0);
    for (size_t i = 1; i < records.size(); ++i) {
        double interval = records[i].first - records[i - 1].first;
        rates[records[i - 1].second] += interval;
    }
    double duration = (*records.rbegin()).first - records[0].first;
    for (int i = 0; i <= maxValue; ++i) {
        rates[i] /= duration;
    }

    // 3) Find M1, M2, variance and standard deviation:
    double m1 = 0.0, m2 = 0.0;
    for (int i = 1; i <= maxValue; ++i) {
        m1 += i * rates[i];
        m2 += i * i * rates[i];
    }

    cDiscreteStatistics stats;
    stats.avg = m1;
    stats.var = m2 - m1*m1;
    stats.std = std::pow(stats.var, 0.5);
    stats.count = records.size();
    stats.pmf = rates;
    return stats;
}

void transformTimeValueArray(
        const std::vector<TimeValue>& source,
        std::vector<TimeValue>& target,
        const std::function<int(int)>& fn) {
    int numRecords = source.size();
    if (numRecords == 0) {
        return;
    }

    // Add first item manually:
    double time = source[0].first;
    int pValue = fn(source[0].second);
    target.push_back(TimeValue(time, pValue));
    if (numRecords == 1) {
        return;
    }

    // If we are here, then there are more records.
    for (int i = 1; i < numRecords - 1; i++) {
        time = source[i].first;
        int nValue = fn(source[i].second);
        if (nValue != pValue) {
            target.push_back(TimeValue(time, nValue));
            pValue = nValue;
        }
    }

    // Add last item manually, no matter of value - we need time.
    auto& lastRecord = source[numRecords - 1];
    target.push_back(TimeValue(lastRecord.first, fn(lastRecord.second)));
}

cStatistics buildScalarStatistics(const std::vector<double>& records) {
    size_t numRecords = records.size();
    double avg = std::accumulate(records.begin(), records.end(), 0.0) / numRecords;
    std::vector<double> squareDiffs;
    squareDiffs.reserve(numRecords);
    std::transform(records.begin(), records.end(), std::back_inserter(squareDiffs),
                   [&avg](double x) -> double { return std::pow(x - avg, 2); });
    double var = std::accumulate(squareDiffs.begin(), squareDiffs.end(), 0) / (numRecords - 1);

    cStatistics stats;
    stats.avg = avg;
    stats.var = var;
    stats.std = std::pow(var, 0.5);
    stats.count = numRecords;
    return stats;
}

cResults *buildResults(System *system, Records *records) {
    auto results = new cResults;
    results->systemSize = buildTimeValueStatistics(records->systemSize);

    // Build queue size statistics:
    std::vector<TimeValue> queueSizeArray;
    queueSizeArray.reserve(records->systemSize.size());
    transformTimeValueArray(records->systemSize, queueSizeArray,
                            [](int x) -> int { return x > 1 ? x - 1 : 0; });
    results->queueSize = buildTimeValueStatistics(queueSizeArray);

    // Build busy ratio:
    std::vector<TimeValue> serverStateArray;
    serverStateArray.reserve(records->systemSize.size());
    transformTimeValueArray(records->systemSize, serverStateArray,
                            [](int x) -> int { return x > 0 ? 1 : 0; });
    results->busy = buildTimeValueStatistics(serverStateArray);

    // Estimate drop ratio
    int numDropped = 0;
    for (auto& pkt: records->packets) {
        if (pkt->dropped) {
            numDropped++;
        }
    }
    results->lossProb = numDropped / records->packets.size();

    // Estimate departure intervals.
    std::vector<double> departureIntervals;
    double prevDeparture = 0.0;
    for (auto& pkt: records->packets) {
        if (pkt->served) {
            departureIntervals.push_back(pkt->departedAt - prevDeparture);
            prevDeparture = pkt->departedAt;
        }
    }
    results->departures = buildScalarStatistics(departureIntervals);

    // Estimate waiting and response time.
    std::vector<double> waitTimes;
    std::vector<double> responseTimes;
    for (auto& pkt: records->packets) {
        if (pkt->served) {
            waitTimes.push_back(pkt->serviceStartedAt - pkt->createdAt);
            responseTimes.push_back(pkt->departedAt - pkt->createdAt);
        }
    }
    results->waitTime = buildScalarStatistics(waitTimes);
    results->responseTime = buildScalarStatistics(responseTimes);

    return results;
}


cResults *_simulate(const cParams& params) {
    System *system = new System(params);
    Records *records = new Records;

    // Initialize model
    records->addSystemSize(0.0, 0);
    system->schedule(ARRIVAL, params.getArrivals().next());

    // Run main loop
    while (!system->stopped()) {
        auto event = system->nextEvent();
//        printf("[%.3f] handling event %s: queue size = %lu, server busy = %d\n",
//               system->getTime(), getEventName(event), system->getQueue()->getSize(),
//               system->getServer()->busy() ? 1 : 0);
        if (event == ARRIVAL) {
            handleArrival(params, system, records);
        } else if (event == SERVICE_END) {
            handleServiceEnd(params, system, records);
        }
    }

    // Record final system size state
    records->addSystemSize(system->getTime(), system->getSize());

    if (system->getServer()->busy()) {
        records->__busyDuration += system->getTime() - records->__lastServerUpdate;
    }
    auto results = buildResults(system, records);
    delete system;
    delete records;
    return results;
}


cResults *cSimulateMm1n(double arrivalRate, double serviceRate, int queueCapacity, int maxPackets) {
    // Create random generators and distributions:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    ExponentialDistribution *arrivals = new ExponentialDistribution(arrivalRate, &gen);
    ExponentialDistribution *services = new ExponentialDistribution(serviceRate, &gen);

    // Create parameters and launch simulation:
    cParams *params = new cParams(arrivals, services, queueCapacity, maxPackets);

    // Run simulation:
    auto *results = _simulate(*params);

    // Clean:
    delete params;
    delete arrivals;
    delete services;

    return results;
}
