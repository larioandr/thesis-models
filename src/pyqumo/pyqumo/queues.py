from functools import cached_property, lru_cache
from typing import Union, Callable, Mapping

import numpy as np

from pyqumo.matrix import cbdiag
from pyqumo.random import Distribution, CountableDistribution, PhaseType
from pyqumo.arrivals import PoissonProcess, MarkovArrivalProcess, RandomProcess, \
    GenericIndependentProcess


class BasicQueueingSystem:
    def __init__(
            self,
            arrival: Union[Distribution, RandomProcess],
            service: Union[Distribution, RandomProcess],
            queue_capacity: int = np.inf,
            precision: float = 1e-9
    ):
        """
        Queueing system constructor.
        """
        if isinstance(arrival, Distribution):
            arrival = GenericIndependentProcess(arrival)
        if isinstance(service, Distribution):
            service = GenericIndependentProcess(service)
        self._arrival = arrival
        self._service = service
        self._queue_capacity = queue_capacity
        self._precision = precision

    @property
    def arrival(self) -> RandomProcess:
        return self._arrival

    @property
    def service(self) -> RandomProcess:
        return self._service

    @property
    def queue_capacity(self) -> int:
        return self._queue_capacity

    @cached_property
    def capacity(self):
        return self._queue_capacity + 1

    @cached_property
    def system_size(self) -> CountableDistribution:
        props = self._get_system_size_props()
        m1, var = props.get('avg', None), props.get('var', None)
        moments = []
        if m1 is not None:
            moments = [m1, var + m1**2] if var is not None else [m1]
        return CountableDistribution(
            self.get_system_size_prob,
            precision=self._precision,
            moments=moments)

    @cached_property
    def queue_size(self) -> CountableDistribution:
        props = self._get_queue_size_props()
        m1, var = props.get('avg', None), props.get('var', None)
        moments = []
        if m1 is not None:
            moments = [m1, var + m1**2] if var is not None else [m1]

        def fn(x: int) -> float:
            if x > 0:
                return self.get_system_size_prob(x + 1)
            if x == 0:
                return self.get_system_size_prob(0) + \
                       self.get_system_size_prob(1)
            return 0.0

        return CountableDistribution(
            fn, precision=self._precision, moments=moments)

    def _get_system_size_props(self) -> Mapping[str, float]:
        """
        This helper can return props of the system size distribution: avg, var.

        By default, returns an empty dictionary. If overridden, may return
        a dictionary with keys 'avg', 'var'. If they found, they will be
        used to compute 1-st and 2-nd moments precisely.
        """
        return {}

    def _get_queue_size_props(self) -> Mapping[str, float]:
        """
        This helper can return props of the queue size distribution: avg, var.

        By default, returns an empty dictionary. If overridden, may return
        a dictionary with keys 'avg', 'var'. If they found, they will be
        used to compute 1-st and 2-nd moments precisely.
        """
        return {}

    @property
    def wait_time(self):
        return self.response_time - self.service.mean

    @property
    def departure(self) -> RandomProcess:
        raise NotImplementedError

    @property
    def utilization(self) -> float:
        raise NotImplementedError

    @property
    def get_system_size_prob(self) -> Callable[[int], float]:
        raise NotImplementedError

    @property
    def response_time(self) -> float:
        raise NotImplementedError


class MM1Queue(BasicQueueingSystem):
    def __init__(self, arrival_rate: float, service_rate: float,
                 precision: float = 1e-9):
        arrival = PoissonProcess(arrival_rate)
        service = PoissonProcess(service_rate)
        super().__init__(arrival, service, precision=precision)

    @cached_property
    def departure(self):
        return PoissonProcess(self.arrival.rate)

    @cached_property
    def utilization(self):
        return self.arrival.rate / self.service.rate

    @cached_property
    def get_system_size_prob(self) -> Callable[[int], float]:
        rho = self.utilization
        if 0 <= rho <= 1:
            return lambda size: (1 - rho) * pow(rho, size) if size >= 0 else 0
        raise ValueError(f"no system size distribution, utilization = {rho}")

    def _get_system_size_props(self) -> Mapping[str, float]:
        rho = self.utilization
        if rho >= 1:
            return {'avg': np.inf, 'var': np.inf}
        avg = rho / (1 - rho)
        var = rho / (1 - rho)**2
        return {'avg': avg, 'var': var}

    def _get_queue_size_props(self) -> Mapping[str, float]:
        rho = self.utilization
        if rho >= 1:
            return {'avg': np.inf, 'var': np.inf}
        avg = rho**2 / (1 - rho)
        m2 = rho**2 * (1 + rho) / (1 - rho)**2
        var = m2 - avg**2
        return {'avg': avg, 'var': var}

    @cached_property
    def response_time(self):
        service_rate_diff = self.service.rate - self.arrival.rate
        if service_rate_diff < 1e-12:
            return np.inf
        return 1 / service_rate_diff

    def __repr__(self):
        return f"(MM1Queue: arrival={self.arrival.rate:.3g}, " \
               f"service={self.service.rate:.3g})"


class MM1NQueue(BasicQueueingSystem):
    def __init__(self, arrival_rate: float,
                 service_rate: float,
                 queue_capacity: int,
                 precision: float = 1e-9):
        if abs(np.math.modf(queue_capacity)[0]) > 1e-12 or queue_capacity <= 0:
            raise ValueError(f"positive integer expected, "
                             f"but {queue_capacity} found")
        arrival = PoissonProcess(arrival_rate)
        service = PoissonProcess(service_rate)
        super().__init__(arrival, service, queue_capacity=queue_capacity,
                         precision=precision)

    @cached_property
    def departure(self):
        n = self.capacity
        a = self.arrival.rate
        b = self.service.rate
        d0 = cbdiag(n + 1, [
            (0, np.asarray([[-(a + b)]])),
            (1, np.asarray([[a]]))
        ])
        d0[0, 0] += b
        d0[n, n] += a
        d1 = cbdiag(n + 1, [(-1, np.asarray([[b]]))])
        return MarkovArrivalProcess(d0, d1)

    @cached_property
    def utilization(self):
        return 1 - self.system_size.pmf(0)

    @lru_cache
    def get_system_size_prob(self) -> Callable[[int], float]:
        rho = self.arrival.rate / self.service.rate
        if rho >= 1:
            return lambda x: 1 if x == self.capacity else 0
        p0 = (1 - rho) / (1 - rho**(self.capacity + 1))
        return lambda x: rho**x * p0 if 0 <= x <= self.capacity else 0.0

    def _get_system_size_props(self) -> Mapping[str, float]:
        rho = self.arrival.rate / self.service.rate
        n = self.capacity

        if rho >= 1:
            return {}

        p0 = self.system_size.pmf(0)
        rho_n = pow(rho, n)    # rho^n
        rho_np1 = rho_n * rho  # rho^{n+1}
        k = rho / (1 - rho)    # helper: \frac{\rho}{1 - \rho}
        avg = k - (n + 1) * rho_np1 / (1 - rho_np1)
        m2 = k * n * (n + 1) * p0 * rho_n + (1 + rho) / (1 - rho) * avg
        var = m2 - avg**2
        return {'avg': avg, 'm2': m2, 'var': var}

    def _get_queue_size_props(self) -> Mapping[str, float]:
        rho = self.arrival.rate / self.service.rate
        n = self.capacity

        if rho >= 1:
            return {}

        ns = self._get_system_size_props()
        p0 = self.system_size.pmf(0)
        avg = ns['avg'] - (1 - p0)
        m2 = ns['m2'] - 2 * ns['avg'] + (1 - p0)
        var = ns['var'] - 2 * p0 * ns['avg'] + p0 * (1 - p0)

        return {'avg': avg, 'm2': m2, 'var': var}

    @cached_property
    def response_time(self):
        rho = self.arrival.rate / self.service.rate
        if rho >= 1:
            return np.inf

        mu = self.service.rate
        n = self.capacity
        rho_n = pow(rho, n)
        numerator = 1 / mu * (1 - (n + 1) * rho_n + n * rho_n * rho)
        denominator = (1 - rho) * (1 - rho_n * rho)
        return numerator / denominator


class MapPh1NQueue(BasicQueueingSystem):
    def __init__(self,
                 arrival: MarkovArrivalProcess,
                 service: PhaseType,
                 queue_capacity: int):
        if abs(np.math.modf(queue_capacity)[0]) > 1e-12 or queue_capacity <= 0:
            raise ValueError(f"positive integer expected, "
                             f"but {queue_capacity} found")
        super().__init__(arrival, service, queue_capacity=queue_capacity,
                         precision=1e-20)

    def _get_casted_arrival_and_service(self) \
            -> (MarkovArrivalProcess, PhaseType):
        """
        Returns (arrival, service), casted to MarkovArrival and PH.

        This method is mostly needed to avoid problems with Python linter,
        that warns about wrong types (RandomProcess instead of
        MarkovArrivalProcess for arrival, and RandomProcess instead of
        GenericIndependentProcess for service).

        However, most of the methods are cached, so this doesn't add much
        overhead job.
        """
        arrival = self.arrival
        assert isinstance(arrival, MarkovArrivalProcess)
        service_process = self.service
        assert isinstance(service_process, GenericIndependentProcess)
        service = service_process.dist
        assert isinstance(service, PhaseType)
        return arrival, service

    @cached_property
    def departure(self) -> MarkovArrivalProcess:
        arrival, service = self._get_casted_arrival_and_service()

        # Aliasing matrices from arrival MAP and service PH
        d0 = arrival.d0
        d1 = arrival.d1
        w = arrival.order
        iw = np.eye(w)
        s = service.s
        tau = service.init_probs
        v = service.order
        iv = np.eye(v)
        ev = np.ones((v, 1))
        m = self.capacity - 1
        b = v * w
        ob = np.zeros((b, b))

        # Building blocks
        d0_iv = np.kron(d0, iv)
        d1_iv = np.kron(d1, iv)
        d0_s = np.kron(d0, iv) + np.kron(iw, s)
        ct = np.kron(-s.dot(ev), tau)
        iw_ct = np.kron(iw, ct)
        r0 = np.kron(d1, np.kron(tau, ev))
        ra = np.kron(d0 + d1, iv) + np.kron(iw, s)

        # Building departure D0 and D1
        d0_dep = cbdiag(self.capacity, ((0, d0_s), (1, d1_iv)))
        d0_dep[m*b:, m*b:] = ra
        d0_left_col = np.vstack((d0_iv,) + (ob,) * self.capacity)
        d0_top_row = np.hstack((r0,) + (ob,) * m)
        d0_dep = np.hstack((d0_left_col, np.vstack((d0_top_row, d0_dep))))
        D1_dep = cbdiag(self.capacity + 1, ((-1, iw_ct),))

        return MarkovArrivalProcess(d0_dep, D1_dep)

    @property
    def utilization(self):
        return self.arrival.rate / self.service.rate

    def get_system_size_prob(self) -> Callable[[int], float]:
        arrival, service = self._get_casted_arrival_and_service()
        departure = self.departure

        b = arrival.order * service.order
        ctmc = departure.ctmc
        pmf = ctmc.steady_pmf

        return lambda x: ctmc.steady_pmf[x*b: (x+1)*b].sum() \
            if 0 <= x <= self.capacity else 0.0

    @cached_property
    def response_time(self):
        return self.system_size.mean / self.arrival.rate
