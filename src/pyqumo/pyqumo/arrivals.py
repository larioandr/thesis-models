from functools import cached_property, lru_cache

import numpy as np

from pyqumo import chains
from pyqumo.matrix import is_infinitesimal, cached_method, cbdiag, order_of
from pyqumo.distributions import Exp
from pyqumo.stochastic import Rnd


class ArrivalProcess:
    def mean(self):
        raise NotImplementedError

    @property
    @lru_cache
    def rate(self):
        return 1.0 / self.mean()

    def var(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError

    def cv(self):
        raise NotImplementedError

    def lag(self, k):
        raise NotImplementedError

    def moment(self, k):
        raise NotImplementedError

    def generate(self, size):
        raise NotImplementedError

    def next(self):
        return self.generate(1)


class GIProcess(ArrivalProcess):
    def __init__(self, dist):
        if dist is None:
            raise ValueError('distribution required')
        self._dist = dist

    @property
    def dist(self):
        return self._dist

    def mean(self):
        return self._dist.mean()

    def var(self):
        return self._dist.var()

    def std(self):
        return self._dist.std()

    def cv(self):
        return self._dist.std() / self._dist.mean()

    def lag(self, k):
        return 0.0

    def moment(self, k):
        return self._dist.moment(k)

    def generate(self, size):
        return self._dist.generate(size)


class PoissonProcess(GIProcess):
    def __init__(self, rate):
        if rate <= 0.0:
            raise ValueError("positive rate expected, '{}' found".format(rate))
        super().__init__(Exp(rate))


class MAP(ArrivalProcess):
    @staticmethod
    def erlang(shape, rate):
        """MAP representation of Erlang process with the given shape and rate.

        :param shape number of phases
        :param rate rate at each phase
        """
        d0 = cbdiag(shape, [(0, [[-rate]]), (1, [[rate]])])
        d1 = np.zeros((shape, shape))
        d1[shape-1, 0] = rate
        return MAP(d0, d1)

    @staticmethod
    def exponential(rate):
        """MAP representation of a Poisson process with the given rate.

        :param rate exponential distribution rate ($1/\\mu$)
        """
        if rate <= 0:
            raise ValueError("positive rate expected, '{}' found".format(rate))
        d0 = [[-rate]]
        d1 = [[rate]]
        return MAP(d0, d1)

    def __init__(self, d0, d1, check=True, rtol=1e-5, atol=1e-6):
        """Create MAP process with the given D0 and D1 matrices.

        If `check == True` is set (this is default), then we validate matrices
        D0 and D1. These matrices must comply three requirements:

        1) Sum `D0 + D1` must be an infinitesimal matrix
        2) All elements of D1 must be non-negative
        3) All elements of D0 except the main diagonal must be non-negative

        To avoid problems with "1e-9 is not zero", constructor accepts
        parameters `rtol` (relative tolerance) and `atol` (absolute tolerance):

        - any value `x: |x| <= atol` is treated as `0`
        - any value `x: x >= -atol` is treated as non-negative
        - any value `x: x <= atol` is treated as non-positive

        Sometimes it is useful to avoid matrices validation (e.g., when MAP
        matrices are obtained as a result of another computation). To disable
        validation, set `check = False`.

        :param d0 matrix of "invisible" transitions
        :param d1 matrix of "visible" transitions
        :param check set to `False` to skip matrices D0 and D1 validation
        :param rtol relative tolerance used when validating D0 and D1
        :param atol absolute tolerance used when validating D0 and D1
        """
        super().__init__()
        d0 = np.asarray(d0)
        d1 = np.asarray(d1)
        self._generator = d0 + d1
        self._order = order_of(d0)
        if check:
            if not is_infinitesimal(self._generator, rtol=rtol, atol=atol):
                raise ValueError("D0 + D1 must be infinitesimal")

            if not np.all(d1 >= -atol):
                raise ValueError("all D1 elements must be non-negative")

            if not (np.all(d0 - np.diag(d0.diagonal()) >= -atol)):
                raise ValueError("all non-diagonal D0 elements must be "
                                 "non-negative")

        self._d = [d0, d1]
        self.__inv_d0 = np.linalg.inv(self.D0)

        # Since we will use Mmap for data generation, we need to generate
        # special matrices for transitions probabilities and rates.

        # 1) We need to store rates (we will use them when choosing random
        #    time we spend in the state):
        self._rates = -self._d[0].diagonal()

        # 2) Then, we need to store cumulative transition probabilities P.
        #    We store them in a stochastic matrix of shape N x (K*N):
        #      P[I, J] is a probability, that:
        #      - new state will be J (mod N), and
        #      - if J < N, then no packet is generated, or
        #      - if J >= N, then packet of type J // N is generated.
        self._trans_pmf = np.hstack((
            self._d[0] + np.diag(self._rates),  # leftmost diagonal is zero
            self._d[1]
        )) / self._rates[:, None]

        # Build embedded DTMC and CTMC:
        self.__ctmc = chains.CTMC(self._generator, check=False)
        self.__dtmc = chains.DTMC(-self.__inv_d0.dot(self.D1), check=False)

        # Define random variables generators:
        # -----------------------------------
        # - random generators for time in each state:
        self.__rate_rnd = [Rnd(
            lambda n, r=r: np.random.exponential(1/r, size=n),
            label=f"exp({r:.3f})"
        ) for r in self._rates]

        # - random generators of state transitions:
        n_trans = self._order * len(self._d)
        self.__trans_rnd = [Rnd(
            lambda n, p0=p: np.random.choice(np.arange(n_trans), p=p0, size=n),
            label=f"choose({n_trans}, p={p})"
        ) for p in self._trans_pmf]

        # Since we have the initial distribution, we find the initial state:
        self._state = np.random.choice(
            np.arange(self._order), p=self.__dtmc.steady_pmf())

    def copy(self):
        """Build a new MAP with the same matrices D0 and D1 without validation.
        """
        return MAP(self.D0, self.D1, check=False)

    # noinspection PyPep8Naming
    @property
    def D0(self):
        """Get matrix D0."""
        return self._d[0]

    # noinspection PyPep8Naming
    @property
    def D1(self):
        """Get matrix D1."""
        return self._d[1]

    # noinspection PyPep8Naming
    def D(self, n):
        """Get matrix Dn for n = 0 or n = 1."""
        self._d[n]

    @cached_property
    def generator(self):
        return self._generator

    @cached_property
    def order(self) -> int:
        return self._order

    @lru_cache
    def d0n(self, k: int) -> np.ndarray:
        """
        Returns $(-D0)^{k}$.
        """
        if k == -1:
            return -self.__inv_d0
        if k == 0:
            return np.eye(self.order)
        if k > 0:
            return self.d0n(k - 1).dot(-self.D0)
        # If we are here, degree <= -2
        return self.d0n(k + 1).dot(self.d0n(-1))

    @lru_cache
    def moment(self, k: int) -> np.ndarray:
        pi = self.embedded_dtmc().steady_pmf()
        x = np.math.factorial(k) * pi.dot(self.d0n(-k)).dot(np.ones(self.order))
        return x.item()

    @cached_property
    def rate(self):
        return 1.0 / self.moment(1)

    @lru_cache
    def mean(self):
        return self.moment(1)

    @lru_cache
    def var(self):
        return self.moment(2) - pow(self.moment(1), 2)

    @lru_cache
    def std(self):
        return self.var() ** 0.5

    @lru_cache
    def cv(self):
        return self.std() / self.mean()

    @lru_cache
    def lag(self, k):
        #
        # Computing lag-k as:
        #
        #   r^2 * pi * (-D0)^(-1) * P^k * (-D0)^(-1) * 1s - 1
        #   -------------------------------------------------- ,
        #   2 * r^2 * pi * (-D0)^(-2) * 1s - 1
        #
        # where r - rate (\lambda), pi - stationary distribution of the
        # embedded DTMC, 1s - vector of ones of MAP order
        #
        dtmc_matrix_k = self._pow_dtmc_matrix(k)
        pi = self.embedded_dtmc().steady_pmf()
        rate2 = pow(self.rate, 2.0)
        e = np.ones(self.order)
        d0ni = self.d0n(-1)
        d0ni2 = self.d0n(-2)

        numerator = (rate2 *
                     pi.dot(d0ni).dot(dtmc_matrix_k).dot(d0ni).dot(e)) - 1
        denominator = (2 * rate2 * pi.dot(d0ni2).dot(e) - 1)
        return numerator / denominator

    @lru_cache
    def background_ctmc(self):
        return self.__ctmc

    @lru_cache
    def embedded_dtmc(self):
        return self.__dtmc

    def generate(self, size):
        for _ in range(size):
            pkt_type = 0
            interval = 0.0
            # print('> start in state ', self._state)
            i = self._state
            while pkt_type == 0:
                interval += self.__rate_rnd[i]()
                j = self.__trans_rnd[i]()
                pkt_type, i = divmod(j, self._order)
            self._state = i
            yield interval
    
    def __call__(self):
        return next(self.generate(1))
    
    def reset_generate_state(self):
        self._state = np.random.choice(
            np.arange(self._order), p=self.__dtmc.steady_pmf())

    def compose(self, other):
        # TODO:  write unit tests
        if not isinstance(other, MAP):
            raise TypeError
        self_eye = np.eye(self.order)
        other_eye = np.eye(other.order)
        d0_out = np.kron(self.D0, other_eye) + np.kron(other.D0, self_eye)
        d1_out = np.kron(self.D1, other_eye) + np.kron(other.D1, self_eye)
        return MAP(d0_out, d1_out)
    
    @lru_cache
    def _pow_dtmc_matrix(self, k):
        if k == 0:
            return np.eye(self.order)
        elif k > 0:
            return self._pow_dtmc_matrix(k - 1).dot(self.embedded_dtmc().matrix)
        else:
            raise ValueError("k='{}' must be non-negative".format(k))
