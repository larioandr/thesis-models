import pytest

import numpy as np
from numpy.testing import assert_allclose

from pyqumo import arrivals as ar
from pyqumo.distributions import Erlang
from pyqumo import stats
from pyqumo.arrivals import MAP as MarkovArrival


#
# POISSON PROCESS
# #######################

@pytest.mark.parametrize('rate', (1.0, 2.0))
def test_poisson_process_constructor(rate):
    p1 = ar.PoissonProcess(rate)
    assert_allclose(p1.rate, rate)


def test_poisson_process_moments():
    p = ar.PoissonProcess(2.0)
    assert_allclose(p.mean(), 0.5)
    assert_allclose(p.std(), 0.5)
    assert_allclose(p.var(), 0.25)
    assert_allclose(p.moment(1), 0.5)
    assert_allclose(p.moment(2), 0.5)
    assert_allclose(p.moment(3), 0.75)
    assert_allclose(p.cv(), 1.0)


def test_poisson_process_lag_is_zero():
    p = ar.PoissonProcess(2.0)
    assert p.lag(1) == 0
    assert p.lag(2) == 0


def test_poisson_process_generate():
    p = ar.PoissonProcess(3.0)
    samples = np.asarray(list((p.generate(20000))))
    assert_allclose(p.mean(), samples.mean(), rtol=0.05)
    assert_allclose(p.std(), samples.std(), rtol=0.05)


#
# Markovian Arrival Process
# ##############################
def test_valid_map_constructor():
    m1 = MarkovArrival(
        d0=[[-1, 0.5], [0.5, -1]], 
        d1=[[0, 0.5], [0.2, 0.3]]
    )
    assert_allclose(m1.generator, [[-1, 1], [0.7, -0.7]])


def test_map_constructor__negative_nondiagonal_d0_elements():
    d0 = [[-0.9, -0.1], [0, -1]]
    d1 = [[0, 1.1], [1., 0.]]
    
    # 1) Check that no testing performed if check=False:
    MarkovArrival(d0, d1, check=False)

    # 2) Check that if negative elements are less, then the given atol,
    #    then they are fixed (made equal to zero)
    arrival = MarkovArrival(d0, d1, atol=0.11)
    assert_allclose(arrival.D0, [[-1.0, 0], [0, -1]])
    assert_allclose(arrival.D1, d1)

    # 3) Check that an error is raised if error is greater then tolerance
    with pytest.raises(ValueError):
        MarkovArrival(d0, d1)
    

def test_map_constructor__negative_d1_elements_raise_error():
    with pytest.raises(ValueError):
        MarkovArrival(
            d0=[[-1., 1.], [1.5, -1.]],
            d1=[[0., 0.], [-0.5, 0.0]])


def test_map_constructor__non_infinitesimal_generator_raise_error():
    with pytest.raises(ValueError):
        MarkovArrival(
            d0=[[-1., 1.], [0.5, -1.0]],
            d1=[[0.01, 0.0], [0, 0.5]])


@pytest.mark.parametrize('rate', [1.0, 2.0, 100.0])
def test_map_representation_of_poisson_arrival(rate):
    arrival = MarkovArrival.exponential(rate)

    # Validate matrices:
    assert_allclose(arrival.D0, [[-rate]])
    assert_allclose(arrival.D1, [[rate]])
    
    # Validate moments and variance:
    assert_allclose(arrival.moment(1), 1/rate)
    assert_allclose(arrival.moment(2), 2/rate**2)
    assert_allclose(arrival.mean(), 1/rate)
    assert_allclose(arrival.std(), 1/rate)
    assert_allclose(arrival.cv(), 1.0)
    assert_allclose(arrival.var(), 1/rate**2)
    
    # Make sure autocorrelation is 0:
    assert_allclose(arrival.lag(1), 0)

    # Generate several samples and validate their statistic properties:
    samples = np.asarray([arrival() for _ in range(10000)])
    assert_allclose(samples.mean(), 1/rate, rtol=0.05)
    assert_allclose(samples.std(), 1/rate, rtol=0.05)


@pytest.mark.parametrize('d0,d1', [
    (
        [[-9.0,  0.0,  0.0,   0.0],
         [0.0,  -9.0,  9.0,   0.0],
         [0.0,   0.0, -0.1,   0.0],
         [0.1,   0.0,  0.0,  -0.1]],
        [[8.0, 1.0, 0.00, 0.00],
         [0.0, 0.0, 0.00, 0.00],
         [0.0, 0.0, 0.09, 0.01],
         [0.0, 0.0, 0.00, 0.00]]
    )
])
def test_map_sampling(d0, d1):
    m = ar.MAP(d0, d1)
    NUM_SAMPLES = 100000
    samples = list(m.generate(NUM_SAMPLES))

    assert len(samples) == NUM_SAMPLES
    assert_allclose(np.mean(samples), m.mean(), rtol=0.05)
    assert_allclose(np.std(samples), m.std(), rtol=0.05)
    assert_allclose(np.var(samples), m.var(), rtol=0.05)
    assert_allclose(stats.lag(samples, 2), [m.lag(1), m.lag(2)], rtol=0.05)


# class TestMAP(ut.TestCase):

#     def test_erlang_constructor(self):
#         m1 = ar.MAP.erlang(1, 1.0)
#         m2 = ar.MAP.erlang(2, 5.0)
#         m3 = ar.MAP.erlang(3, 10.0)

#         assert_allclose(m1.D0, [[-1.0]])
#         assert_allclose(m1.D1, [[1.0]])
#         assert_allclose(m2.D0, [[-5, 5], [0, -5]])
#         assert_allclose(m2.D1, [[0, 0], [5, 0]])
#         assert_allclose(m3.D0, [[-10, 10, 0], [0, -10, 10], [0, 0, -10]])
#         assert_allclose(m3.D1, [[0, 0, 0], [0, 0, 0], [10, 0, 0]])

#     def test_moments_like_erlang(self):
#         e1 = Erlang(1, 1.0)
#         e2 = Erlang(2, 5.0)
#         e3 = Erlang(3, 10.0)
#         m1 = ar.MAP.erlang(e1.shape, e1.rate)
#         m2 = ar.MAP.erlang(e2.shape, e2.rate)
#         m3 = ar.MAP.erlang(e3.shape, e3.rate)

#         for k in range(10):
#             self.assertAlmostEqual(m1.moment(k), e1.moment(k))
#             self.assertAlmostEqual(m2.moment(k), e2.moment(k))
#             self.assertAlmostEqual(m3.moment(k), e3.moment(k))

#     # noinspection PyTypeChecker

#     # noinspection PyTypeChecker
#     def test_call(self):
#         D0 = [
#             [-99.0,  0.0,   0.0,   0.0],
#             [0.0,  -99.0,  99.0,   0.0],
#             [0.0,    0.0, -0.01,   0.0],
#             [0.01,   0.0,   0.0, -0.01],
#         ]
#         D1 = [
#             [98.0, 1.00, 0.000, 0.000],
#             [0.00, 0.00, 0.000, 0.000],
#             [0.00, 0.00, 0.009, 0.001],
#             [0.00, 0.00, 0.000, 0.000],
#         ]
#         m = ar.MAP(D0, D1, check=True)
#         NUM_SAMPLES = 1
#         samples = [m() for _ in range(NUM_SAMPLES)]

#         self.assertEqual(len(samples), NUM_SAMPLES)
#         assert_allclose(np.mean(samples), m.mean(), rtol=0.1)
#         assert_allclose(np.std(samples), m.std(), rtol=0.1)
#         assert_allclose(np.var(samples), m.var(), rtol=0.1)
#         assert_allclose(stats.lag(samples, 2), [m.lag(1), m.lag(2)], rtol=0.1)
