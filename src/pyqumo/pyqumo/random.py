from functools import lru_cache, cached_property
from typing import Union, Sequence, Callable, Any, Mapping, Tuple, Iterator, \
    Optional, Iterable

import numpy as np
from scipy import linalg, integrate
import scipy.stats
from scipy.special import ndtr

from pyqumo import stats
from pyqumo.errors import MatrixShapeError
from pyqumo.matrix import is_pmf, order_of, cbdiag, fix_stochastic, \
    is_subinfinitesimal, fix_infinitesimal, is_square, is_substochastic


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


def _str_array(array: np.ndarray):
    num_axis = len(array.shape)
    if num_axis == 1:
        return "[" + ", ".join([f"{x:.3g}" for x in array]) + "]"
    return "[" + ", ".join(
        [_str_array(array[i]) for i in range(array.shape[0])]
    ) + "]"


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
        if n <= 0 or (n - np.floor(n)) > 0:
            raise ValueError(f'positive integer expected, but {n} found')
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

    @lru_cache()
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
        self._rate = rate

    @property
    def rate(self) -> float:
        """Distribution parameter"""
        return self._rate

    @lru_cache
    def _moment(self, n: int) -> float:
        return np.math.factorial(n) / (self.rate**n)

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

    def __init__(self, shape: int, rate):
        super().__init__()
        if (shape <= 0 or shape == np.inf or
                np.abs(np.round(shape) - shape) > 0):
            raise ValueError("shape must be positive integer")
        if rate <= 0.0:
            raise ValueError("rate must be positive")
        self._shape, self._rate = int(np.round(shape)), rate

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def rate(self) -> float:
        return self._rate

    @lru_cache
    def _moment(self, n: int) -> float:
        """
        Return n-th moment of Erlang distribution.

        N-th moment of Erlang distribution with shape `K` and rate `R` is
        computed as: :math:`k (k+1) ... (k + n - 1) / r^n`
        """
        k, r = self.shape, self.rate
        return k / r if n == 1 else (k + n - 1) / r * self._moment(n - 1)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        r, k = self.rate, self.shape
        koef = r**k / np.math.factorial(k - 1)
        base = np.e**(-r)
        return lambda x: 0 if x < 0 else koef * x**(k - 1) * base**x

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        # Prepare data
        r, k = self.rate, self.shape
        factorials = np.cumprod(np.concatenate(([1], np.arange(1, k))))
        # Summation coefficients are: r^0 / 0!, r^1 / 1!, ... r^k / k!:
        koefs = np.power(r, np.arange(self.shape)) / factorials
        base = np.e**(-r)

        # CDF is given with:
        #   1 - e^(-r*x) ((r^0/0!) * x^0 + (r^1/1!) x^1 + ... + (r^k/k!) x^k):
        return lambda x: 0 if x < 0 else \
            1 - base**x * koefs.dot(np.power(x, np.arange(k)))

    def _eval(self, size: int) -> np.ndarray:
        return np.random.exponential(1 / self.rate, size=(size * self.shape))\
            .reshape((self.shape, size))\
            .sum(axis=0)

    def __repr__(self):
        return f"(Erlang: shape={self.shape:.3g}, rate={self.rate:.3g})"


class HyperExponential(ContinuousDistributionMixin, AbstractCdfMixin,
                       Distribution):
    """Hyper-exponential distribution.

    Hyper-exponential distribution is defined by:

    - a vector of rates (a1, ..., aN)
    - probabilities mass function (p1, ..., pN)

    Then the resulting probability is a weighted sum of exponential
    distributions Exp(ai) with weights pi:

    $X = \\sum_{i=1}^{N}{p_i X_i}$, where $X_i ~ Exp(ai)$
    """
    def __init__(self, rates: Sequence[float], probs: Sequence[float]):
        rates, probs = np.asarray(rates), np.asarray(probs)
        if len(rates) != len(probs):
            raise ValueError("rates and probs vectors order mismatch")
        if not is_pmf(probs):
            probs = fix_stochastic(probs)
        if not (np.all(rates >= 0.0)):
            raise ValueError("rates must be non-negative")
        self._rates, self._probs = rates, probs
        self._exponents = [Exponential(rate) for rate in rates]

    @property
    def rates(self) -> np.ndarray:
        return self._rates

    @property
    def probs(self) -> np.ndarray:
        return self._probs

    @property
    def order(self):
        return len(self._rates)

    @lru_cache
    def _moment(self, n: int) -> float:
        return np.math.factorial(n) * (self.probs / (self.rates ** n)).sum()

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        fns = [exp.pdf for exp in self._exponents]
        return lambda x: sum(p * f(x) for (p, f) in zip(self.probs, fns))

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        fns = [exp.cdf for exp in self._exponents]
        return lambda x: sum(p * f(x) for (p, f) in zip(self.probs, fns))

    def _eval(self, size: int) -> np.ndarray:
        states = np.random.choice(self.order, size, p=self._probs)
        values = [self._exponents[state]() for state in states]
        return np.asarray(values)

    def __repr__(self):
        return f"(HyperExponential: " \
               f"probs={self._probs.tolist()}, " \
               f"rates={self._rates.tolist()})"


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

    @property
    def rate(self) -> float:
        return 1 / self.mean

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
        return f"(PhaseType: s={self.s.tolist()}, p={self.init_probs.tolist()})"


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
        if not values:
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
            if weights:
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
        trans = _str_array(self._trans_probs)
        time_ = "[" + ', '.join([str(td) for td in self._states]) + "]"
        probs = _str_array(self._init_probs)
        return f"(SemiMarkovAbsorb: trans={trans}, time={time_}, p0={probs})"


#
# class LinComb:
#     def __init__(self, dists, w=None):
#         self._dists = dists
#         self._w = w if w is not None else np.ones(len(dists))
#         assert len(self._w) == len(dists)
#
#     def mean(self):
#         acc = 0
#         for w, d in zip(self._w, self._dists):
#             try:
#                 x = w * d.mean()
#             except AttributeError:
#                 x = w * d
#             acc += x
#         return acc
#
#     def var(self):
#         acc = 0
#         for w, d in zip(self._w, self._dists):
#             try:
#                 x = (w ** 2) * d.var()
#             except AttributeError:
#                 x = 0
#             acc += x
#         return acc
#
#     def std(self):
#         return self.var() ** 0.5
#
#     def __call__(self):
#         acc = 0
#         for w, d in zip(self._w, self._dists):
#             try:
#                 x = w * d()
#             except TypeError:
#                 x = w * d
#             acc += x
#         return acc
#
#     def generate(self, size=None):
#         if size is None:
#             return self()
#         acc = np.zeros(size)
#         for w, d in zip(self._w, self._dists):
#             try:
#                 row = d.generate(size)
#             except AttributeError:
#                 row = d * np.ones(size)
#             acc += w * row
#         return acc
#
#     def __str__(self):
#         return ' + '.join(f'{w}*{d}' for w, d in zip(self._w, self._dists))
#
#     def __repr__(self):
#         return str(self)
#
#
# class VarChoice:
#     def __init__(self, dists, w=None):
#         assert w is None or len(w) == len(dists)
#         if w is not None:
#             w = np.asarray([round(wi, 5) for wi in w])
#             if not np.all(np.asarray(w) >= 0):
#                 print(w)
#                 assert False
#             assert np.all(np.asarray(w) >= 0)
#             assert sum(w) > 0
#
#         self._dists = dists
#         self._w = np.asarray(w) if w is not None else np.ones(len(dists))
#         self._p = self._w / sum(self._w)
#
#     @property
#     def order(self):
#         return len(self._dists)
#
#     def mean(self):
#         acc = 0
#         for p, d in zip(self._p, self._dists):
#             try:
#                 x = p * d.mean()
#             except AttributeError:
#                 x = p * d
#             acc += x
#         return acc
#
#     def var(self):
#         return self.moment(2) - self.mean() ** 2
#
#     def std(self):
#         return self.var() ** 0.5
#
#     def moment(self, k):
#         acc = 0
#         for p, d in zip(self._p, self._dists):
#             try:
#                 x = p * d.moment(k)
#             except AttributeError:
#                 x = p * (d ** k)
#             acc += x
#         return acc
#
#     def __call__(self):
#         index = np.random.choice(self.order, p=self._p)
#         try:
#             return self._dists[index]()
#         except TypeError:
#             return self._dists[index]
#
#     def generate(self, size=None):
#         if size is None:
#             return self()
#         return np.asarray([self() for _ in range(size)])
#
#     def __str__(self):
#         return (
#                 '{{' +
#                 ', '.join(f'{w}: {d}' for w, d in zip(self._p, self._dists)) +
#                 '}}')
#
#     def __repr__(self):
#         return str(self)
#
#
# class LinearTransform(Distribution):
#     def __init__(self, xi, k, b):
#         super().__init__()
#         self.__xi = xi
#         self.__k = k
#         self.__b = b
#
#     @property
#     def xi(self):
#         return self.__xi
#
#     @property
#     def k(self):
#         return self.__k
#
#     @property
#     def b(self):
#         return self.__b
#
#     def mean(self):
#         return self.k * self.xi.mean() + self.b
#
#     def var(self):
#         return (self.k ** 2) * self.xi.var()
#
#     def std(self):
#         return self.k * self.xi.std()
#
#     def moment(self, n: int):
#         if n <= 0 or np.abs(n - np.round(n)) > 0:
#             raise ValueError('positive integer expected')
#         if n == 1:
#             return self.mean()
#         elif n == 2:
#             return self.mean() ** 2 + self.var()
#         else:
#             raise ValueError('two moments supported')
#
#     def generate(self, num):
#         for i in range(num):
#             yield self.xi() * self.k + self.b
#
#     def pdf(self, x):
#         return self.xi.pdf((x - self.b) / self.k) / self.k
#
#     def cdf(self, x):
#         return self.xi.cdf((x - self.b) / self.k)
#
#     def __call__(self) -> float:
#         return self.xi() * self.k + self.b
#
#     def sample(self, shape):
#         size = np.prod(shape)
#         return np.asarray(list(self.generate(size))).reshape(shape)
