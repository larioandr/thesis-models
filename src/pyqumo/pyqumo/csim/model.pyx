import numpy as np
from libcpp.vector cimport vector
from pyqumo.csim.model cimport SimData, NodeData, simMM1, VarData
from pyqumo.sim.helpers import Statistics
from pyqumo.sim.gg1 import Results
from pyqumo.random import CountableDistribution


# noinspection PyUnresolvedReferences
cdef vector_asarray(vector[double] vect):
    cdef int n = vect.size()
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = vect[i]
    return ret


cdef _build_statistics(VarData* cs):
    return Statistics(avg=cs.avg, std=cs.std, var=cs.var, count=cs.count)


# noinspection PyUnresolvedReferences
cdef _build_results(const SimData& sim_data):
    cdef int addr = 0
    cdef NodeData data = sim_data.nodeData.at(0)
    results = Results()
    results.system_size = CountableDistribution(
        vector_asarray(data.systemSize.getPmf())
    )
    results.queue_size = CountableDistribution(
        vector_asarray(data.queueSize.getPmf())
    )
    results.busy = CountableDistribution(
        vector_asarray(data.serverSize.getPmf())
    )
    results.loss_prob = data.lossProb
    results.departures = _build_statistics(&data.departures)
    results.response_time = _build_statistics(&data.responseTime)
    results.wait_time = _build_statistics(&data.waitTime)
    results.real_time = sim_data.realTimeMs
    return results


cdef call_simMM1(double arrival_rate, double service_rate,
                        int queue_capacity, int max_packets):
    cdef SimData c_ret = simMM1(
        arrival_rate, service_rate, queue_capacity, max_packets)
    result: Results = _build_results(c_ret)
    return result


def simulate_mm1n(
        arrival_rate: float,
        service_rate: float,
        queue_capacity: int,
        max_packets: int = 100000
) -> Results:
    """
    Wrapper for C++ implementation of M/M/1/N model.

    Returns results in the same dataclass as defined for G/G/1 model
    in `pyqumo.sim.gg1.Results`.

    Parameters
    ----------
    arrival_rate : float
    service_rate : float
    queue_capacity : int
    max_packets : int, optional
        By default 100'000

    Returns
    -------
    results : Results
    """
    return call_simMM1(
        arrival_rate,
        service_rate,
        queue_capacity,
        max_packets
    )
