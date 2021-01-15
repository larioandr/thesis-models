from functools import lru_cache, cached_property
from typing import Union, Sequence, Callable, Mapping, Tuple, Iterator, \
    Optional, Iterable

import numpy as np
from scipy import linalg, integrate
import scipy.stats
from scipy.special import ndtr

from pyqumo import stats
from pyqumo.errors import MatrixShapeError
from pyqumo.matrix import is_pmf, order_of, cbdiag, fix_stochastic, \
    is_subinfinitesimal, fix_infinitesimal, is_square, is_substochastic, \
    str_array


class Rnd:
    def __init__(self, fn, cache_size=10000, label=""):
        self.__fn = fn
        self.__cache_size = cache_size
        self.__samples = []
        self.__index = cache_size
        self.label = label

    def __call__(self):
        if self.__index >= self.__cache_size:
            self.__samples = self.__fn(self.__cache_size)
            self.__index = 0
        x = self.__samples[self.__index]
        self.__index += 1
        return x

    def __repr__(self):
        return f"<Rnd: '{self.label}'>"


class Distribution:
    """
    Base class for all continuous distributions.
    """
    @cached_property
    def mean(self) -> float:
        """
        Get mean value of the random variable.
        """
        return self._moment(1)

    @cached_property
    def rate(self) -> float:
        """
        Get rate (1/mean)
        """
        return 1 / self.mean

    @cached_property
    def var(self) -> float:
        """
        Get variance (dispersion) of the random variable.
        """
        return self._moment(2) - self._moment(1)**2

    @cached_property
    def std(self):
        """
        Get standard deviation of the random variable.
        """
        return self.var ** 0.5

    @cached_property
    def cv(self):
        """
        Get coefficient of variation (relation of std.dev. to mean value)
        """
        return self.std / self.mean

    def moment(self, n: int) -> float:
        """
        Get n-th moment of the random variable.

        Parameters
        ----------
        n : int
            moment degree, for n=1 it is mean value

        Returns
        -------
        value : float

        Raises
        ------
        ValueError
            raised if n is not an integer or is non-positive
        """
        if n < 0 or (n - np.floor(n)) > 0:
            raise ValueError(f'positive integer expected, but {n} found')
        if n == 0:
            return 1
        return self._moment(n)

    def _moment(self, n: int) -> float:
        """
        Compute n-th moment.
        """
        raise NotImplementedError

    def __call__(self, size: int = 1) -> Union[float, np.ndarray]:
        """
        Generate random samples of the random variable with this distribution.

        Parameters
        ----------
        size : int, optional
            number of values to generate (default: 1)

        Returns
        -------
        value : float or ndarray
            if size > 1, then returns a 1D array, otherwise a float scalar
        """
        if size == 1:
            return self._eval(1)[0]
        return self._eval(size)

    def _eval(self, size: int) -> np.ndarray:
        """
        Generate random samples. This method should be defined in inherited
        classes.
        """
        raise NotImplementedError


class AbstractCdfMixin:
    """
    Mixin that adds cumulative distribution function property prototype.
    """
    @property
    def cdf(self) -> Callable[[float], float]:
        """
        Get cumulative distribution function (CDF).
        """
        raise NotImplementedError


class ContinuousDistributionMixin:
    """
    Base mixin for continuous distributions, provides `pdf` property.
    """
    @property
    def pdf(self) -> Callable[[float], float]:
        """
        Get probability density function (PDF).
        """
        raise NotImplementedError


class DiscreteDistributionMixin:
    """
    Base mixin for discrete distributions, provides `pmf` prop and iterator.
    """
    @property
    def pmf(self) -> Callable[[float], float]:
        """
        Get probability mass function (PMF).
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """
        Iterate over (value, prob) pairs.
        """
        raise NotImplementedError


class Const(ContinuousDistributionMixin, DiscreteDistributionMixin,
            AbstractCdfMixin, Distribution):
    """
    Constant distribution that always results in a given constant value.
    """
    def __init__(self, value: float):
        self._value = value
        self._next = None

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        return lambda x: np.inf if x == self._value else 0

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        return lambda x: 0 if x < self._value else 1

    @cached_property
    def pmf(self) -> Callable[[float], float]:
        return lambda x: 1 if x == self._value else 0

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        yield self._value, 1.0

    @lru_cache
    def _moment(self, n: int) -> float:
        return self._value ** n

    def _eval(self, size: int) -> np.ndarray:
        return np.asarray([self._value] * size)

    def __repr__(self):
        return f'(Const: value={self._value:g})'


class Uniform(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Uniform random distribution.

    Notes
    -----

    PDF function :math:`f(x) = 1/(b-a)` anywhere inside ``[a, b]``,
     and zero otherwise. CDF function `F(x)` is equal to 0 for ``x < a``,
     1 for ``x > b`` and :math:`F(x) = (x - a)/(b - a)` anywhere inside
     ``[a, b]``.

    Moment :math:`m_n` for any natural number n is computed as:

    .. math:: m_n = 1/(n+1) (a^0 b^n + a^1 b^{n-1} + ... + a^n b^0).

    Variance :math:`Var(x) = (b - a)^2 / 12.
    """
    def __init__(self, a: float = 0, b: float = 1):
        self._a, self._b = a, b

    @property
    def min(self) -> float:
        return self._a if self._a < self._b else self._b

    @property
    def max(self) -> float:
        return self._b if self._a < self._b else self._a

    @lru_cache
    def _moment(self, n: int) -> float:
        a_degrees = np.power(self._a, np.arange(n + 1))
        b_degrees = np.power(self._b, np.arange(n, -1, -1))
        return 1 / (n + 1) * a_degrees.dot(b_degrees)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        k = 1 / (self.max - self.min)
        return lambda x: k if self.min <= x <= self.max else 0

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        a, b = self.min, self.max
        k = 1 / (b - a)
        return lambda x: 0 if x < a else 1 if x > b else k * (x - a)

    def _eval(self, size: int) -> np.ndarray:
        return np.random.uniform(self.min, self.max, size=size)

    def __repr__(self):
        return f'(Uniform: a={self.min:g}, b={self.max:g})'


class Normal(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Normal random distribution.
    """
    def __init__(self, mean, std):
        self._mean, self._std = mean, std

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @cached_property
    def var(self):
        return self._std**2

    @lru_cache
    def _moment(self, n: int) -> float:
        m, s = self._mean, self._std

        if n == 1:
            return m
        elif n == 2:
            return m**2 + s**2
        elif n == 3:
            return m**3 + 3 * m * (s**2)
        elif n == 4:
            return m**4 + 6 * (m**2) * (s**2) + 3 * (s**4)

        # If moment order is too large, try to numerically solve it using
        # `scipy.integrate` module `quad()` routine:

        # noinspection PyTypeChecker
        return integrate.quad(lambda x: x**n * self.pdf(x), -np.inf, np.inf)[0]

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        k = 1 / np.sqrt(2 * np.pi * self.var)
        return lambda x: k * np.exp(-(x - self.mean)**2 / (2 * self.var))

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        k = 1 / (self.std * 2**0.5)
        return lambda x: 0.5 * (1 + np.math.erf(k * (x - self.mean)))

    def _eval(self, size: int) -> np.ndarray:
        return np.random.normal(self._mean, self._std, size=size)

    def __repr__(self):
        return f'(Normal: mean={self._mean:.3g}, std={self._std:.3g})'


class Exponential(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Exponential random distribution.
    """
    def __init__(self, rate: float):
        super().__init__()
        if rate <= 0.0:
            raise ValueError("exponential parameter must be positive")
        self._param = rate

    @property
    def param(self):
        return self._param

    @lru_cache
    def _moment(self, n: int) -> float:
        return np.math.factorial(n) / (self.param**n)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        r = self.rate
        base = np.e ** -r
        return lambda x: r * base**x if x >= 0 else 0.0

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        r = self.rate
        base = np.e ** -r
        return lambda x: 1 - base**x if x >= 0 else 0.0

    def _eval(self, size: int) -> np.ndarray:
        return np.random.exponential(1 / self.rate, size=size)

    def __str__(self):
        return f"(Exp: rate={self.rate:g})"


class Erlang(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Erlang random distribution.

    To create a distribution its shape (k) and rate (lambda) parameters must
    be specified. Its density function is defined as:

    .. math::

        f(x; k, l) = l^k x^(k-1) e^(-l * x) / (k-1)!
    """

    def __init__(self, shape: int, param: float):
        super().__init__()
        if (shape <= 0 or shape == np.inf or
                np.abs(np.round(shape) - shape) > 0):
            raise ValueError("shape must be positive integer")
        if param <= 0.0:
            raise ValueError("rate must be positive")
        self._shape, self._param = int(np.round(shape)), param

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def param(self) -> float:
        return self._param

    @lru_cache
    def _moment(self, n: int) -> float:
        """
        Return n-th moment of Erlang distribution.

        N-th moment of Erlang distribution with shape `K` and rate `R` is
        computed as: :math:`k (k+1) ... (k + n - 1) / r^n`
        """
        k, r = self.shape, self.param
        return k / r if n == 1 else (k + n - 1) / r * self._moment(n - 1)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        r, k = self.param, self.shape
        koef = r**k / np.math.factorial(k - 1)
        base = np.e**(-r)
        return lambda x: 0 if x < 0 else koef * x**(k - 1) * base**x

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        # Prepare data
        r, k = self.param, self.shape
        factorials = np.cumprod(np.concatenate(([1], np.arange(1, k))))
        # Summation coefficients are: r^0 / 0!, r^1 / 1!, ... r^k / k!:
        koefs = np.power(r, np.arange(self.shape)) / factorials
        base = np.e**(-r)

        # CDF is given with:
        #   1 - e^(-r*x) ((r^0/0!) * x^0 + (r^1/1!) x^1 + ... + (r^k/k!) x^k):
        return lambda x: 0 if x < 0 else \
            1 - base**x * koefs.dot(np.power(x, np.arange(k)))

    def _eval(self, size: int) -> np.ndarray:
        return np.random.exponential(1 / self.param, size=(size * self.shape))\
            .reshape((self.shape, size))\
            .sum(axis=0)

    def __repr__(self):
        return f"(Erlang: shape={self.shape:.3g}, rate={self.param:.3g})"


# noinspection PyUnresolvedReferences
class MixtureDistribution(ContinuousDistributionMixin, AbstractCdfMixin,
                          Distribution):
    """
    Mixture of continuous distributions.

    This is defined by:

    - a sequence of distributions `xi_0, xi_1, ..., xi_{N-1}`
    - a sequence of weights `w_0, w_1, ..., w_{N-1}`

    The resulting probability is a weighted sum of the given distributions:

    :math:`f(x) = w_0 f_{xi_0}(x) + ... + w_{N-1} f_{xi_{N-1}}(x)`
    """
    def __init__(self, states: Sequence[Distribution],
                 weights: Optional[Sequence[float]] = None):
        order = len(states)
        if order == 0:
            raise ValueError("no distributions provided")
        if weights is not None:
            if len(states) != len(weights):
                raise ValueError(
                    f"expected equal number of states and weights, "
                    f"but {len(states)} and {weights} found")
            weights = np.asarray(weights)
            if (weights < 0).any():
                raise ValueError(f"negative weights disallowed: {weights}")
            self._probs = weights / weights.sum()
        else:
            weights = np.ones(order)
            self._probs = 1/order * weights
        # Store distributions as a new tuple:
        self._states = tuple(states)
        self._order = order

    @property
    def states(self) -> Sequence[Distribution]:
        return self._states

    @property
    def probs(self) -> np.ndarray:
        return self._probs

    @property
    def order(self) -> int:
        return self._order

    @lru_cache
    def _moment(self, n: int) -> float:
        moments = np.asarray([st.moment(n) for st in self._states])
        return self._probs.dot(moments)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        fns = [state.pdf for state in self._states]
        return lambda x: sum(p * f(x) for (p, f) in zip(self.probs, fns))

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        fns = [state.cdf for state in self._states]
        return lambda x: sum(p * f(x) for (p, f) in zip(self.probs, fns))

    def _eval(self, size: int) -> np.ndarray:
        state_indexes = np.random.choice(self.order, size, p=self._probs)
        values = [self.states[si]() for si in state_indexes]
        return np.asarray(values)

    def __repr__(self):
        states_str = "[" + ", ".join(str(state) for state in self.states) + "]"
        probs_str = str_array(self._probs)
        return f"(Mixture: states={states_str}, probs={probs_str})"


class HyperExponential(MixtureDistribution):
    """Hyper-exponential distribution.

    Hyper-exponential distribution is defined by:

    - a vector of rates (a1, ..., aN)
    - probabilities mass function (p1, ..., pN)

    Then the resulting probability is a weighted sum of exponential
    distributions Exp(ai) with weights pi:

    $X = \\sum_{i=1}^{N}{p_i X_i}$, where $X_i ~ Exp(ai)$
    """
    def __init__(self, rates: Sequence[float], probs: Sequence[float]):
        exponents = [Exponential(rate) for rate in rates]
        super().__init__(exponents, probs)

    # noinspection PyUnresolvedReferences
    @cached_property
    def rates(self) -> np.ndarray:
        return np.asarray([state.rate for state in self.states])

    def __repr__(self):
        return f"(HyperExponential: " \
               f"probs={str_array(self.probs)}, " \
               f"rates={str_array(self.rates)})"


# noinspection PyUnresolvedReferences
class AbsorbMarkovPhasedEvalMixin:
    """
    Mixin for _eval() for phased distributions with Markovian transitions.

    To use this mixin, distribution need to implement:

    - `states` (property): returns an iterable. Calling an item should
    return a sample value of the time spent in the state, size `N`. Signature
    of each item `__call__()` method should support `size` parameter.

    - `init_probs` (property): initial probability distribution, should have
    the same dimension as `time` sequence (`N`).

    - `trans_probs` (property): matrix of transitions probabilities, should
    have shape `(N+1)x(N+1)`, last state - absorbing.

    - `order` (property): should return the number of transient states (`N`)
    """
    @property
    def order(self) -> int:
        raise NotImplementedError

    @property
    def states(self) -> Sequence[Callable[[Optional[int]], np.ndarray]]:
        raise NotImplementedError

    @property
    def init_probs(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def trans_probs(self) -> np.ndarray:
        raise NotImplementedError

    def _init_eval(self):
        """
        Initialize random values generators
        """
        assert hasattr(self, 'order')
        assert hasattr(self, 'states')
        assert hasattr(self, 'init_probs')
        assert hasattr(self, 'trans_probs')

        # Random generator for initial state:
        self.__init_rnd = Rnd(
            lambda n: np.random.choice(
                all_states[:-1],
                p=self.init_probs,
                size=n
            ))

        # Random generators for time in each state:
        self.__time_rnd = [
            Rnd(lambda n, fn=state: fn(n))
            for state in self.states
        ]

        # Random generators of state transitions:
        all_states = np.arange(self.order + 1)
        self.__trans_rnd = [
            Rnd(lambda n, p=row: np.random.choice(all_states, p=p, size=n))
            for row in self.trans_probs
        ]

    def is_absorbing_state(self, state_index: int) -> bool:
        """
        Check whether the state is absorbing.
        """
        return state_index >= self.order

    def _eval(self, size: int) -> np.ndarray:
        # Initialize random generators if they were not inited earlier
        if not hasattr(self, '__init_rnd'):
            self._init_eval()

        # Generate required number of items:
        samples_array = np.zeros(size)
        for i in range(size):
            # Select random initial state and jump over the chain
            # till getting into absorbing state:
            sample = 0.0
            j = self.__init_rnd()
            while not self.is_absorbing_state(j):
                sample += self.__time_rnd[j]()
                j = self.__trans_rnd[j]()
            samples_array[i] = sample
        return samples_array


class PhaseType(ContinuousDistributionMixin,
                AbstractCdfMixin,
                AbsorbMarkovPhasedEvalMixin,
                Distribution):
    """
    Phase-type (PH) distribution.

    This distribution is specified with a subinfinitesimal matrix and
    initial states probability distribution.

    PH distribution is a generalization of exponential, Erlang,
    hyperexponential, hypoexponential and hypererlang distributions, so
    they can be defined using PH distribution means. However, PH distribution
    operates with matrices, incl. matrix-exponential operations, so it is
    less efficient then custom implementations.
    """
    def __init__(self, sub: np.ndarray, p: np.ndarray, safe: bool = False):
        # Validate and fix data:
        # ----------------------
        if not safe:
            if (sub_order := order_of(sub)) != order_of(p):
                raise MatrixShapeError(f'({sub_order},)', p.shape, 'PMF')
            if not is_subinfinitesimal(sub):
                sub = fix_infinitesimal(sub, sub=True)
            if not is_pmf(p):
                p = fix_stochastic(p)

        # Store data in fields:
        # ---------------------
        self._subgenerator = sub
        self._pmf0 = p
        self._sni = -np.linalg.inv(sub)  # (-S)^{-1} - negated inverse of S

        # Build internal representations for transitions PMFs and rates:
        # --------------------------------------------------------------
        self._order = order_of(self._pmf0)
        self._rates = -self._subgenerator.diagonal()
        self._trans_probs = np.hstack((
            self._subgenerator + np.diag(self._rates),
            -self._subgenerator.sum(axis=1)[:, None]
        )) / self._rates[:, None]
        self._states = [Exponential(r) for r in self._rates]

    @staticmethod
    def exponential(rate: float) -> 'PhaseType':
        sub = np.asarray([[-rate]])
        p = np.asarray([1.0])
        return PhaseType(sub, p, safe=True)

    @staticmethod
    def erlang(shape: int, rate: float) -> 'PhaseType':
        blocks = [
            (0, np.asarray([[-rate]])),
            (1, np.asarray([[rate]]))
        ]
        sub = cbdiag(shape, blocks)
        p = np.zeros(shape)
        p[0] = 1.0
        return PhaseType(sub, p, safe=True)

    @staticmethod
    def hyperexponential(rates: Sequence[float], probs: Sequence[float]):
        order = len(rates)
        sub = np.zeros((order, order))
        for i, rate in enumerate(rates):
            sub[(i, i)] = -rate
        if not isinstance(probs, np.ndarray):
            probs = np.asarray(probs)
        return PhaseType(sub, probs, safe=False)

    @cached_property
    def order(self) -> int:
        return order_of(self._subgenerator)

    @property
    def s(self):
        return self._subgenerator

    @property
    def init_probs(self):
        return self._pmf0

    @property
    def trans_probs(self):
        return self._trans_probs

    @property
    def states(self):
        return self._states

    @property
    def sni(self) -> np.ndarray:
        return self._sni

    @lru_cache
    def _moment(self, n: int) -> float:
        sni_powered = np.linalg.matrix_power(self.sni, n)
        ones = np.ones(shape=(self.order, 1))
        x = np.math.factorial(n) * self.init_probs.dot(sni_powered).dot(ones)
        return x.item()

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        p = np.asarray(self._pmf0)
        s = np.asarray(self._subgenerator)
        tail = -s.dot(np.ones(self.order))
        return lambda x: 0 if x < 0 else p.dot(linalg.expm(x * s)).dot(tail)

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        p = np.asarray(self._pmf0)
        ones = np.ones(self.order)
        s = np.asarray(self._subgenerator)
        return lambda x: 0 if x < 0 else 1 - p.dot(linalg.expm(x * s)).dot(ones)

    def __repr__(self):
        return f"(PH: s={str_array(self.s)}, p={str_array(self.init_probs)})"


class Choice(DiscreteDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Discrete distribution of values with given non-negative weights.
    """
    def __init__(self, values: Sequence[float],
                 weights: Union[Mapping[float, float], Sequence[float]] = None):
        """
        Discrete distribution constructor.

        Different values probabilities are computed based on weights as
        :math:`p_i = w_i / (w_1 + w_2 + ... + w_N)`.

        Parameters
        ----------
        values : sequence of values
        weights : mapping of values to weights or a sequence of weights, opt.
            if provided as a sequence, then weights length should be equal
            to values length; if not provided, all values are expected to
            have the same weight.
        """
        if len(values) == 0:
            raise ValueError('expected non-empty values')
        weights_, values_ = [], []
        try:
            # First we assume that values is dictionary. In this case we
            # expect that it stores values in pairs like `value: weight` and
            # iterate through it using `items()` method to fill value and
            # weights arrays:
            # noinspection PyUnresolvedReferences
            for key, weight in values.items():
                values_.append(key)
                weights_.append(weight)
            weights_ = np.asarray(weights_)
        except AttributeError:
            # If `values` doesn't have `items()` attribute, we treat as an
            # iterable holding values only. Then we check whether its size
            # matches weights (if provided), and fill weights it they were
            # not provided:
            values_.extend(values)
            if weights is not None and len(weights) > 0:
                if len(values) != len(weights):
                    raise ValueError('values and weights size mismatch')
            else:
                weights = (1. / len(values),) * len(values)
            weights_ = np.asarray(weights)

        # Check that all weights are non-negative and their sum is positive:
        if np.any(weights_ < 0):
            raise ValueError('weights must be non-negative')
        total_weight = sum(weights_)
        if np.allclose(total_weight, 0):
            raise ValueError('weights sum must be positive')

        # Store values and probabilities
        probs_ = weights_ / total_weight
        self._data = [(v, p) for v, p in zip(values_, probs_)]
        self._data.sort(key=lambda item: item[0])

    @cached_property
    def values(self) -> np.ndarray:
        return np.asarray([item[0] for item in self._data])

    @cached_property
    def probs(self) -> np.ndarray:
        return np.asarray([item[1] for item in self._data])

    @lru_cache()
    def __len__(self):
        return len(self._data)

    def find_left(self, value: float) -> int:
        """
        Searches for the value and returns the closest left side index.

        Examples
        --------
        >>> choices = Choice([1, 3, 5], [0.2, 0.5, 0.3])
        >>> choices.find_left(1)
        >>> 0
        >>> choices.find_left(2)  # not in values, return leftmost value index
        >>> 0
        >>> choices.find_left(5)
        >>> 2
        >>> choices.find_left(-1)  # for too small values return -1
        >>> -1

        Parameters
        ----------
        value : float
            value to search for

        Returns
        -------
        index : int
            if `value` is found, return its index; if not, but there is value
            `x < value` and there are no other values `y: x < y < value` in
            data, return index of `x`. If for any `x` in data `x > value`,
            return `-1`.
        """
        def _find(start: int, end: int) -> int:
            delta = end - start
            if delta < 1:
                return -1
            if delta == 1:
                return start if value >= self.values[start] else -1
            middle = start + delta // 2
            middle_value = self.values[middle]
            if np.allclose(value, middle_value):
                return middle
            if value < middle_value:
                return _find(start, middle)
            return _find(middle, end)
        return _find(0, len(self))

    def get_prob(self, value: float) -> float:
        """
        Get probability of a given value.

        Parameters
        ----------
        value : float

        Returns
        -------
        prob : float
        """
        index = self.find_left(value)
        stored_value = self.values[index]
        if index >= 0 and np.allclose(value, stored_value):
            return self.probs[index]
        return 0.0

    @lru_cache
    def _moment(self, n: int) -> float:
        return (self.values**n).dot(self.probs).sum()

    def _eval(self, size: int) -> np.ndarray:
        return np.random.choice(self.values, p=self.probs, size=size)

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        cum_probs = np.cumsum(self.probs)

        def fn(x):
            index = self.find_left(x)
            return cum_probs[index] if index >= 0 else 0.0

        return fn

    @cached_property
    def pmf(self) -> Callable[[float], float]:
        return lambda x: self.get_prob(x)

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        for value, prob in self._data:
            yield value, prob

    def __repr__(self):
        return f"(Choice: values={self.values.tolist()}, " \
               f"p={self.probs.tolist()})"


class CountableDistribution(DiscreteDistributionMixin,
                            AbstractCdfMixin,
                            Distribution):
    """
    Distribution with set of values {0, 1, 2, ...} - all non-negative numbers.

    An example of this kind of distributions is geometric distribution.
    Since a set of values is infinite, we specify this distribution with
    a probability function that takes value = 0, 1, ..., and returns its
    probability.

    To compute properties we will need to find sum of infinite series.
    We specify precision as maximum tail probability, and use only first
    values (till this tail) when estimating sums.
    """
    def __init__(self, prob: Callable[[int], float],
                 precision: float = 1e-9,
                 moments: Sequence[float] = ()):
        """
        Constructor.

        Parameters
        ----------
        prob : callable, (int) -> float
            Probability function. Should accept arguments 0, 1, 2, ...
            and return their probability.
        precision : float, optional
            Maximum tail probability. If this tail starts at value X=N,
            properties will be estimated over values 0, 1, ..., N only,
            without respect to the tail. Note, that when estimating
            moments of high order the error will grow due to the growth
            of tail weight (for n > N > 1, n**K > n for K > 1).
            By default, 1e-9.
        moments : sequence of floats, optional
            Optional explicit moments values. If given, they will be
            used instead of estimating over first 0, 1, ..., N values.
            By default, empty tuple.
        """
        self._prob = prob
        self._precision = precision
        self._moments = moments

        # Find number I, such that P[X > I] < precision:
        trunc_pmf = []
        head_prob = 0.0
        max_value = -1
        while head_prob < 1 - precision:
            max_value += 1
            p = prob(max_value)
            trunc_pmf.append(p)
            head_prob += p
        self._max_value = max_value

        values = np.arange(self._max_value + 1)
        self._trunc_pmf = np.asarray(trunc_pmf)
        self._trunc_cdf = np.cumsum(self._trunc_pmf)
        self._trunc_choice = Choice(tuple(values), self._trunc_pmf)

    @lru_cache
    def get_prob_at(self, x: int) -> float:
        """
        Get probability of a given value.

        This method is cached, so only first access at a given value may
        be long. For the values 0, 1, ..., `truncated_at` this method is
        fast even at the first run, since these values are computed in
        constructor when computing `truncated_at` value itself.
        For greater values access to `prob(x)` may take time.

        Returns 0.0 for negative arguments, no matter of `prob(x)`.

        Parameters
        ----------
        x : int, non-negative

        Returns
        -------
        probability : float
        """
        return self._trunc_pmf[x] if 0 <= x <= self._max_value else \
            (self._prob(x) if x > self._max_value else 0.0)

    @property
    def prob(self) -> Callable[[int], float]:
        """
        Returns probability function.
        """
        return self._prob

    @property
    def precision(self) -> float:
        """
        Returns precision of this distribution.
        """
        return self._precision

    @property
    def truncated_at(self) -> int:
        """
        Returns a value, such that tail probability is less then precision.

        If `truncated_at = N`, then total probability of values
        N+1, N+2, ... is less then `precision`.
        """
        return self._max_value

    @lru_cache
    def _moment(self, n: int) -> float:
        """
        Computes n-th moment.

        If moments were provided in the constructor, use them. Otherwise
        estimate moment as the weighted sum of the first N elements, where
        `truncated_at = N`. Note, that for large `n` error may be large,
        and, if needed, distribution should be built with high precision.

        Parameters
        ----------
        n : int
            Order of the moment.

        Returns
        -------
        value : float
        """
        if n <= len(self._moments) and self._moments[n-1] is not None:
            return self._moments[n-1]
        values = np.arange(self._max_value + 1)
        degrees = np.power(values, n)
        probs = np.asarray([self.get_prob_at(x) for x in values])
        return probs.dot(degrees)

    @cached_property
    def pmf(self) -> Callable[[float], float]:
        """
        This function returns probability mass function.

        For values, those are NOT non-negative integers, returns 0.0
        Works very fast for values between 0 and `truncated_at` (incl.),
        but requires call to `prob(x)` for greater values.

        Returns
        -------
        pmf : callable (float) -> float
        """
        def fn(x: float) -> float:
            fl, num = np.math.modf(x)
            if abs(fl) > 1e-12:
                return 0
            num = int(num)
            if num < 0:
                return 0.0
            return self.get_prob_at(num)

        return fn

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        """
        This function returns cumulative distribution function.

        For values, those are NOT non-negative integers, returns 0.0
        Works very fast for values between 0 and `truncated_at` (incl.),
        but requires call to `prob(x)` for greater values.

        Returns
        -------
        pmf : callable (float) -> float
        """
        def fn(x: float) -> float:
            _, num = np.math.modf(x)
            num = int(num)
            if num < 0:
                return 0.0
            if num <= self._max_value:
                return self._trunc_cdf[num]
            p = self._trunc_cdf[self._max_value]
            for i in range(self._max_value + 1, num + 1):
                p += self.get_prob_at(i)
            return p

        return fn

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        # To avoid infinite loops, we iterate over 10-times max value.
        total_prob = 0.0
        for i in range(10 * (self._max_value + 1)):
            if total_prob >= 1 - 1e-12:
                return
            p = self.get_prob_at(i)
            total_prob += p
            yield i, p

    def _eval(self, size: int) -> np.ndarray:
        """
        Generate a random array of the given size.

        When generating random values, use `Choice` distribution with values
        0, 1, ..., `truncated_at`. Thus, no values from tail (which prob. is
        less then precision) will be generated.

        Parameters
        ----------
        size : array size

        Returns
        -------
        array : np.ndarray
        """
        return self._trunc_choice(size)

    def __repr__(self):
        values = ', '.join([f"{self.get_prob_at(x):.3g}" for x in range(5)])
        return f"(Countable: p=[{values}, ...], precision={self.precision})"


# noinspection PyUnresolvedReferences
class EstStatsMixin:
    """
    Mixin for distributions without analytic form for moments computation.

    This mixin estimates moments, variance and standard deviation based on
    sampled data. It expects that the derived class provides:

    - `num_stats_samples` property: number of samples should to be used
    in moments estimation.
    - `_eval()` implementation.
    """
    @property
    def num_stats_samples(self) -> int:
        raise NotImplementedError

    @cached_property
    def _stats_samples(self) -> np.ndarray:
        """
        Get samples cache to estimate moments and/or other properties.

        If cache doesn't exist, it will be created`.
        """
        if not hasattr(self, '__stats_samples'):
            self.__stats_samples = self._eval(self.num_stats_samples)
        return self.__stats_samples

    @lru_cache
    def _moment(self, n: int) -> float:
        return stats.moment(self._stats_samples, minn=n, maxn=n)[0]


# noinspection PyUnresolvedReferences
class KdePdfMixin:
    """
    Mixin for distributions without analytic form for PDF and CDF computation.

    This mixin estimates PDF and CDF functions using Gaussian KDE from scipy.
    It requires:

    - `num_kde_samples` property: number of samples should to be used
    in KDE building. Should not be too large since KDE will work VERY slowly.
    - `_eval()` implementation.
    """
    @property
    def num_kde_samples(self) -> int:
        raise NotImplementedError

    @cached_property
    def _kde(self) -> scipy.stats.gaussian_kde:
        if not hasattr(self, '__kde'):
            kde_samples = self._eval(self.num_kde_samples)
            self.__kde = scipy.stats.gaussian_kde(kde_samples)
        return self.__kde

    @property
    def pdf(self):
        return lambda x: self._kde.pdf(x)[0]

    @property
    def cdf(self):
        dataset = self._kde.dataset
        factor = self._kde.factor
        return lambda x: ndtr(np.ravel(x - dataset) / factor).mean()


class SemiMarkovAbsorb(AbsorbMarkovPhasedEvalMixin,
                       EstStatsMixin,
                       KdePdfMixin,
                       Distribution):
    """
    Semi-Markov process with absorbing states.

    Process with `N` states is specified with three parameters:

    - random distribution of time in each state `T[i], i = 0, 1, ..., N-1`
    - transition strictly sub-stochastic matrix `P` of shape `(N, N)`
    - initial probability distribution `p0` of shape `(N,)`

    Process starts in one of its states as defined with `p0`. It spends time
    in each state as defined by the corresponding distribution `T[i]` and
    after that move to another state selected with probabilities `P'[i]`.
    Here `P'` = `[[P; I - P*1], [0, 1]]` - a stochastic matrix built from
    sub-stochastic matrix `P` by appending a column to the right with missing
    probabilities of transiting to absorbing state:
    `P'[i,N] = 1 - (P[i,0] + P[i,1] + ... + P[i,N-1])`. We require P
    to be strictly sub-stochastic, so at least at one row this value should
    be non-zero.

    To generate random samples, we model the behavior of this process
    multiple times. We don't analyze reachability, so loops are possible.

    Note: this method doesn't try to fix sub-stochastic transition matrix if
    it is not valid, in contrast to phase-type distributions or MAP processes.
    """
    def __init__(self, trans: Union[np.ndarray, Sequence[Sequence[float]]],
                 time_dist: Sequence[Distribution],
                 probs: Union[np.ndarray, Sequence[float], int] = 0,
                 num_samples: int = 10000,
                 num_kde_samples: int = 1000):
        """
        Constructor.

        Parameters
        ----------
        trans : array_like
            sub-stochastic transition matrix between non-absorbing states
            with shape `(N, N)`.
        time_dist : sequence of distributions
            distribution of time spent in state, should have length `N`.
        probs : array_like or int
            initial distribution of length `N`, or the number of the first
            state. If the state number is provided (int), then process will
            start from this state with probability 1.0.
        num_samples : int, optional
            number of samples that is used for moments estimation.
            As a rule of thumb, should be large enough, especially if
            high order moments needed.
            By default: 10'000
        num_kde_samples : int, optional
            number of samples that is used for Gaussian KDE building
            for PDF and CDF estimation.
            As a rule of thumb, should NOT be too large, since using PDF
            of KDE built from too large number of samples is very slow.
            By default: 1'000
        """
        # Convert to ndarray and validate transitions matrix:
        if not isinstance(trans, np.ndarray):
            trans = np.asarray(trans)
        else:
            trans = trans.copy()  # copy to avoid changes from outside

        if not is_square(trans):
            raise MatrixShapeError("(N, N)", trans.shape, "transitions matrix")
        order = order_of(trans)
        if not is_substochastic(trans):
            raise ValueError(f"expected sub-stochastic matrix, "
                             f"but {trans} found")

        # Validate time_dist and p0, convert p0 to ndarray if needed:
        if len(time_dist) != order:
            raise ValueError(f"need {order} time distributions, "
                             f"{len(time_dist)} found")

        if isinstance(probs, Iterable):
            if not isinstance(probs, np.ndarray):
                probs = np.asarray(probs)
            else:
                probs = probs.copy()  # copy to avoid changes from outside
            if not is_pmf(probs):
                raise ValueError(f"PMF expected, but {probs} found")
            if (p0_order := order_of(probs)) != order:
                raise ValueError(f"expected P0 vector with order {order}, "
                                 f"but {p0_order} found")
        else:
            # If here, assume p0 is an integer - number of state. Build
            # initial PMF with 1.0 set for this state.
            if not (0 <= probs < order):
                raise ValueError(f"semi-Markov process of order {order} "
                                 f"doesn't have transient state {probs}")
            p0_ = np.zeros(order)
            p0_[probs] = 1.0
            probs = p0_

        # Store matrices and parameters:
        self._order = order
        self._states = tuple(time_dist)  # copy to avoid changes
        self._init_probs = probs
        self._num_samples = num_samples
        self._num_kde_samples = num_kde_samples

        # Build full transitions stochastic matrix:
        self._trans_probs = np.vstack((
            np.hstack((
                trans,
                np.ones((order, 1)) - trans.sum(axis=1).reshape((order, 1))
            )),
            np.asarray([[0] * order + [1]]),
        ))

        # Define cache that is used for moments estimations:
        # --------------------------------------------------
        self.__samples = None
        self.__kde = None

    @property
    def init_probs(self):
        return self._init_probs

    @property
    def trans_probs(self):
        return self._trans_probs

    @property
    def order(self):
        return self._order

    @property
    def states(self):
        return self._states

    @property
    def num_stats_samples(self):
        return self._num_samples

    @property
    def num_kde_samples(self) -> int:
        return self._num_kde_samples

    def __repr__(self):
        trans = str_array(self._trans_probs)
        time_ = "[" + ', '.join([str(td) for td in self._states]) + "]"
        probs = str_array(self._init_probs)
        return f"(SemiMarkovAbsorb: trans={trans}, time={time_}, p0={probs})"
