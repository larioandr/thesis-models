from collections import namedtuple

import numpy as np
from scipy.optimize import fsolve

from pyqumo.distributions import SemiMarkovAbsorb, LinComb, Constant, VarChoice
from pycsmaca.utilities import SPEED_OF_LIGHT


# noinspection PyTypeChecker
def get_bianchi_model_parameters(num_clients, cwmin, cwmax):
    # - m: number of backoff stages (number of times CW increases)
    # - p: collision probability during station transmission
    # - tau: probability that a station transmits
    num_stages = np.log2(cwmax / cwmin)
    assert not (abs(num_stages - round(num_stages)) > 0)
    num_stages = int(num_stages)

    def equations(variables):
        p, t = variables
        m, n = num_stages, num_clients
        w = cwmin
        return [
            p - (1 - (1 - t) ** (n - 1)),
            t - 2 / (1 + w + p * w * sum(pow(2 * p, i) for i in range(m)))
        ]

    solution = fsolve(equations, np.asarray((0.5, 0.5)))
    bianchi_p = round(float(solution[0]), 10)
    bianchi_tau = round(float(solution[1]), 10)

    return namedtuple('BianchiModelParams', ['m', 'n', 'W', 'p', 'tau'])(
        num_stages, num_clients, cwmin, bianchi_p, bianchi_tau
    )


def get_bianchi_chain_state_index(stage, backoff, cwmin):
    # On each stage:
    # - states 1, ..., Wi-1: backoff slots
    # - state 0: state for transmission
    return cwmin * (2 ** stage - 1) + backoff


def get_bianchi_slot_times(payload, ack, machdr, phyhdr, preamble, bitrate,
                           difs, sifs, slot, distance=100, c=SPEED_OF_LIGHT):
    propagation = distance / c
    t_data_ctrl = preamble + (machdr + phyhdr) / bitrate
    t_ack = preamble + (phyhdr + ack) / bitrate

    t_empty = Constant(slot)
    t_data = LinComb([
        difs + sifs + 2 * propagation + t_data_ctrl + t_ack,
        payload,
    ], [1, 1 / bitrate])
    t_collided = LinComb([
        difs + sifs + 6 * propagation + t_data_ctrl + t_ack,
        payload,
    ], [1, 1 / bitrate])

    return namedtuple('BianchiSlotTimes', ['empty', 'data', 'collided'])(
        t_empty, t_data, t_collided
    )


def get_bianchi_slot_probs(params):
    n, tau = params.n, params.tau
    result = namedtuple('BianchiSlotProbs', [
        'wait_slot_empty', 'wait_slot_success', 'wait_slot_collided',
        'trans_slot_success', 'trans_slot_collided',
    ])
    if params.n > 1:
        #
        # P_tr and P_s are probabilities that slot is busy with neighbours
        # transmission and (conditional) with successful transmission.
        # We use (n - 1) in these formulas since in these slots observed
        # station is waiting in backoff.
        #
        p_tr = 1 - (1 - tau) ** (n - 1)
        p_s = (n - 1) * tau * (1 - tau) ** (n - 2) / p_tr
        return result(
            1 - p_tr, p_tr * p_s, p_tr * (1 - p_s), 1 - params.p, params.p
        )
    return result(1, 0, 0, 1 - params.p, params.p)


def get_bianchi_time_matrix(params):
    order = params.W * (2 ** (params.m + 1) - 1)
    get_index = get_bianchi_chain_state_index

    mat = np.zeros((order, order))
    cw = params.W
    for i in range(params.m + 1):
        for b in range(1, cw):
            mat[get_index(i, b, params.W), get_index(i, b - 1, params.W)] = 1
        if i < params.m:
            cw *= 2
            next_i = i + 1
        else:
            next_i = i
        _p = params.p / cw
        for b in range(cw):
            mat[get_index(i, 0, params.W), get_index(next_i, b, params.W)] = _p
    return mat


def get_bianchi_collision_probability(params):
    n, tau = params.n, params.tau
    #
    # P_tr and P_s are probabilities that slot is busy with neighbours
    # transmission and (conditional) with successful transmission.
    #
    p_tr = 1 - (1 - tau) ** n
    p_s = n * tau * (1 - tau) ** (n - 1) / p_tr
    return p_tr * (1 - p_s)


def get_bianchi_throughput(params, payload_mean, t_empty, t_data, t_coll):
    n, tau = params.n, params.tau
    p_tr = 1 - (1 - tau) ** n
    p_s = n * tau * (1 - tau) ** (n - 1) / p_tr
    p_c = p_tr * (1 - p_s)
    return p_s * p_tr * payload_mean / (
            (1 - p_tr) * t_empty + p_tr * p_s * t_data + p_c * t_coll)


def bianchi_time(
        num_clients, payload_size, ack_size, mac_header_size, phy_header_size,
        preamble, bitrate, difs, sifs, slot, cwmin, cwmax, distance=100,
        c=SPEED_OF_LIGHT):
    #
    # 1) Estimate Bianchi model parameters:
    #
    bianchi = get_bianchi_model_parameters(num_clients, cwmin, cwmax)

    #
    # 2) Build the transitional stochastic matrix for absorbing process:
    #
    order = bianchi.W * (2 ** (bianchi.m + 1) - 1)
    get_index = get_bianchi_chain_state_index
    mat = get_bianchi_time_matrix(bianchi)

    #
    # 3) Build waiting time distributions and slot type probabilities:
    #
    slot_times = get_bianchi_slot_times(
        payload_size, ack_size, mac_header_size, phy_header_size, preamble,
        bitrate, difs, sifs, slot, distance, c
    )
    slot_probs = get_bianchi_slot_probs(bianchi)

    time_dists = [Constant(0)] * order
    for i in range(bianchi.m):
        cw = cwmin * (2 ** i)
        for b in range(1, cw):
            time_dists[get_index(i, b, cwmin)] = VarChoice(
                [slot_times.empty, slot_times.data, slot_times.collided],
                [slot_probs.wait_slot_empty, slot_probs.wait_slot_success,
                 slot_probs.wait_slot_collided]
            )
        time_dists[get_index(i, 0, cwmin)] = VarChoice(
            [slot_times.collided, slot_times.data],
            [slot_probs.trans_slot_collided, slot_probs.trans_slot_success],
        )

    p0 = [1 / cwmin] * cwmin + [0] * (order - cwmin)

    process = SemiMarkovAbsorb(mat, time_dists, p0)
    # print('Params: ', bianchi)
    # print(mat)
    # for it, d in enumerate(time_dists):
    #     print(f'{it}: ', d)
    samples = process.generate(1000)

    simret = namedtuple('SimRet', ['mean', 'std', 'process', 'p_collision',
                                   'throughput'])
    p_collision = get_bianchi_collision_probability(bianchi)

    try:
        _payload_mean = payload_size.mean()
    except AttributeError:
        _payload_mean = payload_size

    throughput = get_bianchi_throughput(
        bianchi, _payload_mean, slot_times.empty.mean(), slot_times.data.mean(),
        slot_times.collided.mean()
    )
    return simret(samples.mean(), samples.std(), process, p_collision,
                  throughput)


if __name__ == '__main__':
    ret = bianchi_time(
        num_clients=1,
        payload_size=Constant(1000),
        ack_size=100,
        mac_header_size=50,
        phy_header_size=25,
        preamble=10e-3,
        bitrate=1000,
        difs=200e-3,
        sifs=100e-3,
        slot=50e-3,
        cwmin=4,
        cwmax=64,
        distance=100,
        c=1e5
    )
    print('collision probability: ', ret.p_collision)
    print('service duration: ', ret.mean)
    print('throughput: ', ret.throughput)
