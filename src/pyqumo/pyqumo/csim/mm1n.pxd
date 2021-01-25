from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "mm1n.h":
    # noinspection PyPep8Naming
    cdef cppclass cStatistics:
        double avg
        double std
        double var
        size_t count

    # noinspection PyPep8Naming
    cdef cppclass cDiscreteStatistics:
        vector[double] pmf

    # noinspection PyPep8Naming
    cdef cppclass cResults:
        cDiscreteStatistics systemSize
        cDiscreteStatistics queueSize
        cDiscreteStatistics busy
        double lossProb
        cStatistics departures
        cStatistics responseTime
        cStatistics waitTime

        string tabulate() const

    # noinspection PyPep8Naming
    cResults* cSimulateMm1n(
            double arrivalRate,
            double serviceRate,
            int queueCapacity,
            int maxPackets);
