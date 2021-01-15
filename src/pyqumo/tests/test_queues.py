import unittest as ut
from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from pyqumo.queues import MM1Queue, MM1NQueue, MapPh1NQueue
from pyqumo.arrivals import PoissonProcess, MarkovArrivalProcess, \
    GenericIndependentProcess
from pyqumo.random import Exponential, PhaseType


QueueProps = namedtuple('QueueProps', [
    'departure_class', 'departure_rate', 'utilization',
    'system_size_avg', 'system_size_var',
    'queue_size_avg', 'queue_size_var',
    'system_size_pmf', 'queue_size_pmf',
    'wait_time', 'resp_time'
], defaults=(None,) * 11)

@pytest.mark.parametrize('queue, props, string', [
    (MM1Queue(2.0, 5.0), QueueProps(
        departure_class=PoissonProcess, departure_rate=2.0, utilization=0.4,
        system_size_avg=0.6667, system_size_var=1.1111,
        queue_size_avg=0.2667, queue_size_var=0.5511,
        system_size_pmf=[0.6, 0.24, 0.096, 0.0384, 0.0154],
        queue_size_pmf=[0.84, 0.096, 0.0384, 0.0154],
        wait_time=2/15
        )
        MM1Queue(2, 5),
        PoissonProcess, 2.0,  # class of departure process and its rate
        0.4, 2/15, 1/3,  # utility (rho), wait_time, response_time
        [0.6, 0.24, 0.096, 0.0384, 0.0154],  # system size PMF
        [0.84, 0.096, 0.0384, 0.0154],       # queue size PMF
        2/3, 4/15, 10/9,  # avg. system size and queue size, system size var.
        '(MM1Queue: arrival=2, service=5)'
    ), (
        MM1Queue(1, 2),
        PoissonProcess, 1.0,  # class of departure process and its rate
        0.5, 2 / 15, 1 / 3,  # utility (rho), wait_time, response_time
        [0.6, 0.24, 0.096, 0.0384, 0.0154],  # system size PMF
        [0.84, 0.096, 0.0384, 0.0154],  # queue size PMF
        2/3, 4/15, 10/9, # avg. system size and queue size, system size var.
        '(MM1Queue: arrival=2, service=5)'
    )
    ]
)
def test_basic_props(queue, departure_class, departure_rate, rho, wait_time,
                     resp_time, ss_pmf, qs_pmf, ss_avg, qs_avg, ss_var, string):
    assert isinstance(queue.arrival, PoissonProcess)
    assert isinstance(queue.service, GenericIndependentProcess)
    assert isinstance(queue.service.dist, Exponential)

    # Check departure process
    assert isinstance(queue.departure, PoissonProcess)
    assert_allclose(queue.departure.rate, departure_rate)

    # Check utility, wait time and response time
    assert_allclose(queue.utilization, rho)
    assert_allclose(queue.wait_time, wait_time)
    assert_allclose(queue.response_time, resp_time)

    # Check system size PMF and props:
    ss_pmf_size = len(ss_pmf)
    assert_allclose(
        [queue.system_size.pmf(i) for i in range(ss_pmf_size)],
        ss_pmf,
        rtol=0.005,
    )
    assert_allclose(queue.system_size.mean, ss_avg)
    assert_allclose(queue.system_size.var, ss_var)

    # Check queue size PMF and props:
    qs_pmf_size = len(qs_pmf)
    assert_allclose(
        [queue.queue_size.pmf(i) for i in range(qs_pmf_size)],
        qs_pmf,
        rtol=0.005,
    )
    assert_allclose(queue.queue_size.mean, qs_avg)

    assert str(queue) == string


# class TestMM1(ut.TestCase):
#     def test_system_size_pmf(self):
#         q1 = MM1(1, 2)
#         q2 = MM1(2, 5)
#         assert_almost_equal(q1.system_size_pmf(size=3), [0.5, 0.25, 0.125])
#         assert_almost_equal(q2.system_size_pmf(size=3), [0.6, 0.24, 0.096])
#
#     def test_system_size_avg(self):
#         q1 = MM1(1, 2)
#         q2 = MM1(2, 5)
#         q3 = MM1(1, 1)
#         q4 = MM1(2, 1)
#         self.assertAlmostEqual(q1.system_size_avg, 1.0)
#         self.assertAlmostEqual(q2.system_size_avg, 2 / 3)
#         self.assertAlmostEqual(q3.system_size_avg, np.inf)
#         self.assertAlmostEqual(q4.system_size_avg, np.inf)
#
#     def test_system_size_var(self):
#         q1 = MM1(1, 2)
#         q2 = MM1(2, 5)
#         q3 = MM1(1, 1)
#         q4 = MM1(2, 1)
#         self.assertAlmostEqual(q1.system_size_var, 2.0)
#         self.assertAlmostEqual(q2.system_size_var, 10 / 9)
#         self.assertAlmostEqual(q3.system_size_var, np.inf)
#         self.assertAlmostEqual(q4.system_size_var, np.inf)
#
#     def test_system_size_std(self):
#         q1 = MM1(1, 2)
#         q2 = MM1(2, 5)
#         q3 = MM1(1, 1)
#         q4 = MM1(2, 1)
#         self.assertAlmostEqual(q1.system_size_std, np.sqrt(2.0))
#         self.assertAlmostEqual(q2.system_size_std, np.sqrt(10) / 3)
#         self.assertAlmostEqual(q3.system_size_std, np.inf)
#         self.assertAlmostEqual(q4.system_size_std, np.inf)
#
#     def test_response_time(self):
#         q1 = MM1(1, 2)
#         q2 = MM1(2, 5)
#         q3 = MM1(1, 1)
#         q4 = MM1(2, 1)
#         self.assertAlmostEqual(q1.response_time, 1.0)
#         self.assertAlmostEqual(q2.response_time, 1 / 3)
#         self.assertAlmostEqual(q3.response_time, np.inf)
#         self.assertAlmostEqual(q4.response_time, np.inf)
#
#     def test_wait_time(self):
#         q1 = MM1(1, 2)
#         q2 = MM1(2, 5)
#         q3 = MM1(1, 1)
#         q4 = MM1(2, 1)
#         self.assertAlmostEqual(q1.wait_time, 0.5)
#         self.assertAlmostEqual(q2.wait_time, 2 / 15)
#         self.assertAlmostEqual(q3.wait_time, np.inf)
#         self.assertAlmostEqual(q4.wait_time, np.inf)
#
#
# class TestMM1N(ut.TestCase):
#
#     def test_valid_creation(self):
#         queue_2_5 = MM1N(arrival_rate=2.0, service_rate=5.0, capacity=1)
#         queue_1_10 = MM1N(arrival_rate=1.0, service_rate=10.0, capacity=5)
#         self.assertAlmostEqual(queue_2_5.arrival_rate, 2.0)
#         self.assertAlmostEqual(queue_2_5.service_rate, 5.0)
#         self.assertEqual(queue_2_5.capacity, 1)
#         self.assertAlmostEqual(queue_1_10.arrival_rate, 1.0)
#         self.assertAlmostEqual(queue_1_10.service_rate, 10.0)
#         self.assertAlmostEqual(queue_1_10.capacity, 5)
#
#     def test_invalid_creation(self):
#         with self.assertRaises(ValueError):
#             MM1N(0.0, 1, 1)
#         with self.assertRaises(ValueError):
#             MM1N(-1, 1, 1)
#         with self.assertRaises(ValueError):
#             MM1N(1, 0, 1)
#         with self.assertRaises(ValueError):
#             MM1N(1, -1, 1)
#         with self.assertRaises(TypeError):
#             MM1N('1', 1, 1)
#         with self.assertRaises(TypeError):
#             MM1N(1, '1', 1)
#         with self.assertRaises(TypeError):
#             MM1N([1, 2], 1, 1)
#         with self.assertRaises(TypeError):
#             MM1N(1, [1, 2], 1)
#         with self.assertRaises(ValueError):
#             MM1N(1, 1, -1)
#         with self.assertRaises(ValueError):
#             MM1N(1, 1, 0)
#         with self.assertRaises(TypeError):
#             MM1N(1, 1, '1')
#         with self.assertRaises(TypeError):
#             MM1N(1, 1, 1.2)
#         with self.assertRaises(TypeError):
#             MM1N(1, 1, [1, 2])
#
#     def test_arrival_process(self):
#         queue = MM1N(1.0, 2.0, 5)
#         self.assertIsInstance(queue.arrival, PoissonProcess)
#         self.assertAlmostEqual(queue.arrival.param, queue.arrival_rate)
#
#     def test_service_distribution(self):
#         queue = MM1N(1, 2, 5)
#         self.assertIsInstance(queue.service, Exp)
#         self.assertAlmostEqual(queue.service.param, queue.service_rate)
#
#     def test_departure_process(self):
#         queue = MM1N(2.0, 5.0, 200)
#         self.assertIsInstance(queue.departure, MAP)
#         # Since queue is very long, arrival rate will be close to departure
#         # rate.
#         self.assertAlmostEqual(queue.departure.param, queue.arrival_rate, 5)
#
#     def test_utilization(self):
#         q1 = MM1N(2, 5, 4)
#         q2 = MM1N(1, 10, 3)
#         q3 = MM1N(5, 5, 2)
#         q4 = MM1N(10, 4, 1)
#         self.assertAlmostEqual(q1.utilization, 0.4)
#         self.assertAlmostEqual(q2.utilization, 0.1)
#         self.assertAlmostEqual(q3.utilization, 1.0)
#         self.assertAlmostEqual(q4.utilization, 2.5)
#
#     def test_system_size_pmf(self):
#         q1 = MM1N(1, 2, 10)
#         q2 = MM1N(2, 5, 5)
#         assert_almost_equal(
#             q1.system_size_pmf(size=3), [0.500244, 0.250122, 0.125061], 6)
#         assert_almost_equal(
#             q2.system_size_pmf(size=3), [0.602468, 0.240987, 0.096395], 6)
#
#     def test_system_size_avg(self):
#         q1 = MM1N(1, 2, 5)
#         q2 = MM1N(2, 5, 5)
#         q3 = MM1N(1, 1, 10)
#         q4 = MM1N(2, 1, 100)
#         self.assertAlmostEqual(q1.system_size_avg, 0.9047619047619048)
#         self.assertAlmostEqual(q2.system_size_avg, 0.6419895893580104)
#         self.assertAlmostEqual(q3.system_size_avg, np.inf)
#         self.assertAlmostEqual(q4.system_size_avg, np.inf)
#
#     def test_system_size_var(self):
#         # TODO: implement first
#         pass
#
#     def test_system_size_std(self):
#         # TODO: implement first
#         pass
#
#     def test_response_time(self):
#         q1 = MM1N(1, 2, 5)
#         q2 = MM1N(2, 5, 6)
#         q3 = MM1N(1, 1, 10)
#         q4 = MM1N(2, 1, 100)
#         self.assertAlmostEqual(q1.response_time, 0.9047619047619048)
#         self.assertAlmostEqual(q2.response_time, 0.3275895226739491)
#         self.assertAlmostEqual(q3.response_time, np.inf)
#         self.assertAlmostEqual(q4.response_time, np.inf)
#
#     def test_wait_time(self):
#         q1 = MM1N(1, 2, 5)
#         q2 = MM1N(2, 5, 6)
#         q3 = MM1N(1, 1, 10)
#         q4 = MM1N(2, 1, 100)
#         self.assertAlmostEqual(q1.wait_time, 0.40476190476190477)
#         self.assertAlmostEqual(q2.wait_time, 0.12758952267394907)
#         self.assertAlmostEqual(q3.wait_time, np.inf)
#         self.assertAlmostEqual(q4.wait_time, np.inf)
#
#
# class TestMapPh1N(ut.TestCase):
#
#     def test_valid_creation(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 4)
#         self.assertAlmostEqual(q1.arrival_rate, 1)
#         self.assertAlmostEqual(q1.service_rate, 2)
#         self.assertEqual(q1.capacity, 5)
#         self.assertAlmostEqual(q2.arrival_rate, 2)
#         self.assertAlmostEqual(q2.service_rate, 5)
#         self.assertAlmostEqual(q2.capacity, 4)
#
#     def test_invalid_creation(self):
#         ar1 = MAP.exponential(1.0)
#         srv1 = PhaseType.exponential(2.0)
#
#         with self.assertRaises(ValueError):
#             MapPh1N(ar1, srv1, -1)
#         with self.assertRaises(ValueError):
#             MapPh1N(ar1, srv1, 0)
#         with self.assertRaises(TypeError):
#             MapPh1N(ar1, srv1, '1')
#         with self.assertRaises(TypeError):
#             MapPh1N(ar1, srv1, 1.2)
#         with self.assertRaises(TypeError):
#             MapPh1N(ar1, srv1, [1, 2])
#
#     def test_arrival_process(self):
#         q = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         self.assertIsInstance(q.arrival, MAP)
#         self.assertAlmostEqual(q.arrival.param, q.arrival_rate)
#
#     def test_service_distribution(self):
#         q = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         self.assertIsInstance(q.service, PhaseType)
#         self.assertAlmostEqual(q.service.param, q.service_rate)
#
#     def test_departure_process(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 20)
#         q2 = MapPh1N(MAP.erlang(4, 1.0), PhaseType.erlang(3, 2.0), 20)
#         self.assertIsInstance(q1.departure, MAP)
#         self.assertIsInstance(q2.departure, MAP)
#         # Since queue is very long, arrival rate will be close to departure
#         # rate.
#         self.assertAlmostEqual(q1.departure.param, q1.arrival_rate, 5)
#         self.assertAlmostEqual(q2.departure.param, q2.arrival_rate, 5)
#
#     def test_utilization(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 20)
#         q2 = MapPh1N(MAP.erlang(4, 1.0), PhaseType.erlang(3, 2.0), 20)
#         self.assertAlmostEqual(q1.utilization, 1 / 2)
#         self.assertAlmostEqual(q2.utilization, 3 / 8)
#
#     def test_system_size_pmf(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
#         exp_q1 = MM1N(1.0, 2.0, 5)
#         exp_q2 = MM1N(2, 5, 10)
#         assert_almost_equal(q1.system_size_pmf(10), exp_q1.system_size_pmf(10))
#         self.assertEqual(len(q1.system_size_pmf()), q1.capacity + 1)
#         assert_almost_equal(q2.system_size_pmf(), exp_q2.system_size_pmf())
#
#     def test_system_size_avg(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
#         exp_q1 = MM1N(1.0, 2.0, 5)
#         exp_q2 = MM1N(2, 5, 10)
#         self.assertAlmostEqual(q1.system_size_avg, exp_q1.system_size_avg)
#         self.assertAlmostEqual(q2.system_size_avg, exp_q2.system_size_avg)
#
#     #
#     # def test_system_size_var(self):
#     #     # TODO: implement first
#     #     pass
#     #
#     # def test_system_size_std(self):
#     #     # TODO: implement first
#     #     pass
#     #
#     def test_response_time(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
#         exp_q1 = MM1N(1.0, 2.0, 5)
#         exp_q2 = MM1N(2, 5, 10)
#         self.assertAlmostEqual(q1.response_time, exp_q1.response_time)
#         self.assertAlmostEqual(q2.response_time, exp_q2.response_time)
#
#     def test_wait_time(self):
#         q1 = MapPh1N(MAP.exponential(1.0), PhaseType.exponential(2.0), 5)
#         q2 = MapPh1N(MAP.exponential(2.0), PhaseType.exponential(5.0), 10)
#         exp_q1 = MM1N(1.0, 2.0, 5)
#         exp_q2 = MM1N(2, 5, 10)
#         self.assertAlmostEqual(q1.wait_time, exp_q1.wait_time)
#         self.assertAlmostEqual(q2.wait_time, exp_q2.wait_time)
