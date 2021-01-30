from libcpp.vector cimport vector
from libcpp.map cimport map


cdef extern from "Statistics.h":
    cdef cppclass SizeDist:
        double getMean()
        double getVariance()
        double getStdDev()
        double getMoment(int order)
        vector[double] getPmf()

    cdef cppclass VarData:
        double avg
        double std
        double var
        unsigned count
        vector[double] moments


cdef extern from "Simulation.h":
    cdef cppclass NodeData:
        SizeDist systemSize
        SizeDist queueSize
        SizeDist serverSize
        VarData delays
        VarData departures
        VarData waitTime
        VarData responseTime
        unsigned numPacketsGenerated
        unsigned numPacketsDelivered
        unsigned numPacketsLost
        unsigned numPacketsArrived
        unsigned numPacketsServed
        unsigned numPacketsDropped
        double lossProb
        double dropProb
        double deliveryProb

    cdef cppclass SimData:
        map[int, NodeData] nodeData
        unsigned numPacketsGenerated
        double simTime
        double realTimeMs

    # noinspection PyPep8Naming
    SimData simulate_mm1(
            double arrivalRate,
            double serviceRate,
            int queueCapacity,
            int maxPackets)
