from collections import namedtuple

import pytest
from numpy import asarray
from numpy.testing import assert_almost_equal, assert_allclose

from pycsmaca.analytic.bianchi import get_bianchi_model_parameters, \
    get_bianchi_chain_state_index, get_bianchi_slot_times, \
    get_bianchi_time_matrix


@pytest.mark.parametrize('num_clients, cwmin, cwmax, n, m, w, p, tau', [
    (5, 4, 4, 5, 0, 4, 0.870, 0.400),
    (5, 2, 8, 5, 2, 2, 0.753, 0.295),
    (3, 8, 16, 3, 1, 8, 0.317, 0.173),
])
def test_bianchi_model_parameters(num_clients, cwmin, cwmax, n, m, w, p, tau):
    ret = get_bianchi_model_parameters(num_clients, cwmin, cwmax)
    assert ret.m == m
    assert ret.n == n
    assert ret.W == w
    assert_almost_equal(ret.p, p, decimal=2)
    assert_almost_equal(ret.tau, tau, decimal=2)


@pytest.mark.parametrize('stage, backoff, cwmin, expected', [
    (0, 0, 2, 0),
    (0, 3, 4, 3),
    (1, 2, 8, 10),
    (2, 10, 4, 22),
])
def test_bianchi_chain_state_index(stage, backoff, cwmin, expected):
    actual = get_bianchi_chain_state_index(stage, backoff, cwmin)
    assert actual == expected


def test_bianchi_slot_times():
    payload = 1000
    ack = 250
    machdr = 100
    phyhdr = 50
    preamble = 0.05
    bitrate = 500
    difs = 0.5
    sifs = 0.25
    slot = 0.1
    distance = 10
    c = 200

    ret = get_bianchi_slot_times(
        payload=payload, ack=ack, machdr=machdr, phyhdr=phyhdr,
        preamble=preamble, bitrate=bitrate, difs=difs, sifs=sifs, slot=slot,
        distance=distance, c=c
    )
    assert ret.empty() == 0.1

    # Data slot is: DIFS + DATA + SIFS + ACK + 2 * propagation
    prop = distance / c
    ddsa = difs + preamble + (phyhdr + machdr) / bitrate + sifs + preamble + \
           (phyhdr + ack) / bitrate
    expected_data = ddsa + 2 * prop + payload / bitrate

    assert_almost_equal(ret.data.mean(), expected_data)

    assert expected_data <= ret.collided.mean() <= expected_data + 8 * prop


def test_bianchi_matrix_m0():
    p = 0.5
    order = 4
    params = namedtuple('_P', ['m', 'W', 'p'])(0, 4, p)
    mat = get_bianchi_time_matrix(params)

    assert mat.shape == (order, order)

    expected = asarray([
        [p/4, p/4, p/4, p/4],
        [1,   0,   0,   0],
        [0,   1,   0,   0],
        [0,   0,   1,   0],
    ])

    assert_allclose(mat, expected)


def test_bianchi_matrix_m1():
    p = 0.2
    order = 14
    params = namedtuple('_P', ['m', 'W', 'p'])(2, 2, p)
    mat = get_bianchi_time_matrix(params)

    assert mat.shape == (order, order)

    expected = asarray([
        [0, 0, p/4, p/4, p/4, p/4, 0,   0,   0,   0,   0,   0,   0,   0],
        [1, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   p/8, p/8, p/8, p/8, p/8, p/8, p/8, p/8],
        [0, 0, 1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0, 0, 0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0, 0, 0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   p/8, p/8, p/8, p/8, p/8, p/8, p/8, p/8],
        [0, 0, 0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0,   0],
        [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0,   0],
        [0, 0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   0]
    ])

    assert_allclose(mat, expected)
