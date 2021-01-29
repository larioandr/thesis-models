#include <iostream>
#include <string>
#include <chrono>
#include <random>
#include "Components.h"
#include "System.h"
#include "Simulation.h"


class Exp {
public:
    explicit Exp(double rate) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        gen = std::default_random_engine(seed);
        exp = std::exponential_distribution<double>(rate);
    }
    Exp(const Exp& rside) : exp(rside.exp), gen(rside.gen) {
    }

    virtual ~Exp() {}

    double operator()() {
        return exp(gen);
    }

private:
    std::exponential_distribution<double> exp;
    std::default_random_engine gen;
};


int main(int argc, char **argv) {
    // Parse mandatory parameters
    if (argc < 4 || argc > 5) {
        std::cout << "Format: gg1.x <ARRIVAL_RATE> <SERVICE_RATE> <QUEUE_CAPACITY> [NUM_PACKETS]\n";
        return 1;
    }
    double arrivalRate = std::stod(argv[1]);
    double serviceRate = std::stod(argv[2]);
    int queueCapacityInt = std::stoi(argv[3]);

    if (queueCapacityInt < 0) {
        std::cout << "ERROR: queue capacity must be non-negative\n";
        return 1;
    }
    if (arrivalRate <= 0 || serviceRate <= 0) {
        std::cout << "ERROR: arrival and service rates must be positive\n";
        return 1;
    }
    ssize_t queueCapacity = static_cast<size_t>(queueCapacityInt);

    // Check whether number of packets were provided:
    size_t maxPackets = 10000;
    if (argc == 5) {
        int maxPackets_ = std::stoi(argv[4]);
        if (maxPackets_ <= 0) {
            std::cerr << "ERROR: number of packets must be positive\n";
            return 1;
        }
        maxPackets = static_cast<size_t>(maxPackets_);
    }

    auto ret = simulate_mm1(arrivalRate, serviceRate, queueCapacity, maxPackets);
    std::cout << ret.text() << std::endl;
    return 0;

    // Build network, system and journal
//    auto arrival = new Exp(arrivalRate);
//    auto service = new Exp(serviceRate);
//
//    auto network = buildOneHopeNetwork(*arrival, *service, queueCapacity);
//
//    auto journal = new Journal;
//    for (auto& addrNodePair: network->getNodes()) {
//        journal->addNode(addrNodePair.second);
//    }
//
//    auto system = new System;
//
//    // Execute main loop
//    runMainLoop(network, system, journal, maxPackets);
//    journal->commit();
//
//    std::cout << journal->toString() << std::endl;
//
//    // Clear
//    delete network;
//    delete journal;
//    delete system;
//    delete arrival;
//    delete service;
}
