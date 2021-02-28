"""
This module contains an implementation of the PH fitting algorithm using a
mixture of two Erlang distribution with common order N by three moments match.
The method is defined in paper [1].

[1] Mary A. Johnson & Michael R. Taaffe (1989) Matching moments to phase
    distributions: Mixtures of erlang distributions of common order,
    Communications in Statistics. Stochastic Models, 5:4, 711-743,
    DOI: 10.1080/15326348908807131
"""
from typing import Sequence, Tuple
import numpy as np

from pyqumo.random import HyperErlang
from pyqumo.stats import get_cv, get_skewness, get_noncentral_m3
from pyqumo.errors import BoundsError


def fit_mern2(
    moments: Sequence[float],
    strict: bool = True
) -> Tuple[HyperErlang, np.ndarray]:
    """
    Fit moments with a mixture of two Erlang distributions with common order.

    The algorithm is defined in [1].

    Parameters
    ----------
    moments : sequence of float
        Only the first three moments are taken into account
    """
    # [ ] TODO: write good documentation

    if (num_moments := len(moments)) == 3:
        m1, m2, m3 = moments[:3]
        cv = get_cv(m1, m2)
    else:
        if (strict and num_moments < 3) or num_moments == 0:
            raise ValueError(f"Expected 3 moments, but {num_moments} found")
        m1 = moments[0]
        m2 = moments[1] if num_moments > 1 else 2*pow(m1, 2)
        cv = get_cv(m1, m2)
        if cv < 1 - 1e-5:
            m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) / 2)
        elif abs(cv - 1) <= 1e-4:
            m3 = moments[2] if num_moments > 2 else 6*pow(m1, 3)
        else:
            m3 = get_noncentral_m3(m1, cv, cv - 1/cv + 0.5)

    gamma = get_skewness(m1, m2, m3)

    # Check boundaries and raise BoundsError if fail:
    if strict and (min_skew := cv - 1/cv) >= gamma:
        raise BoundsError(
            f"Skewness = {gamma:g} is too small for CV = {cv:g}\n"
            f"\tmin. skewness = {min_skew:g}\n"
            f"\tm1 = {m1:g}, m2 = {m2:g}, m3 = {m3:g}")

    # Compute N:
    shape_base = int(max(
        np.ceil(1 / cv**2),
        np.ceil((-gamma + 1/cv**3 + 1/cv + 2*cv) / (gamma - (cv - 1/cv)))
    )) + (2 if cv <= 1 else 0)

    def get_ratio(l1_, l2_):
        """
        Helper to get ratio between Erlang rates. Always return value >= 1.
        """
        if l2_ >= l1_ > 0:
            return l2_ / l1_
        return l1_ / l2_ if l1_ > l2_ > 0 else np.inf

    n = shape_base
    l1, l2, p1 = get_mern2_props(m1, m2, m3, n)
    dist = HyperErlang([l1, l2], [n, n], [p1, 1 - p1])

    # Estimate errors:
    errors = np.asarray([
        abs(m - dist.moment(i+1)) / abs(m) for i, m in enumerate(moments)
    ])

    return dist, errors


def get_mern2_props(
        m1: float,
        m2: float,
        m3: float,
        n: int) -> Tuple[float, float, float]:
    """
    Helper function to estimate Erlang distributions rates and
    probabilities from the given moments and Erlang shape (n).

    See theorem 3 in [1] for details about A, B, C, p1, x, y and lambdas
    computation.

    Parameters
    ----------
    m1 : float
        mean value
    m2 : float
        second non-central moment
    m3 : float
        third non-central moment
    n : int
        shape of the Erlang distributions

    Returns
    -------
    l1 : float
        parameter of the first Erlang distribution
    l2 : float
        parameter of the second Erlang distribution
    n : int
        shape of the Erlang distributions

    Raises
    ------
    BoundsError
        raise this if skewness is below CV - 1/CV (CV - coef. of variation)
    """
    # Check boundaries:
    cv = get_cv(m1, m2)
    gamma = get_skewness(m1, m2, m3)
    if (min_skew := cv - 1/cv) >= gamma:
        raise BoundsError(
            f"Skewness = {gamma:g} is too small for CV = {cv:g}\n"
            f"\tmin. skewness = {min_skew:g}\n"
            f"\tm1 = {m1:g}, m2 = {m2:g}, m3 = {m3:g}")

    # Compute auxiliary variables:
    x = m1 * m3 - (n + 2) / (n + 1) * pow(m2, 2)
    y = m2 - (n + 1) / n * pow(m1, 2)
    c = m1 * x
    b = -(
        n * x +
        n * (n + 2) / (n + 1) * pow(y, 2) +
        (n + 2) * pow(m1, 2) * y
    )
    a = n * (n + 2) * m1 * y
    d = pow(b**2 - 4 * a * c, 0.5)

    # Compute Erlang mixture parameters:
    em1, em2 = (-b - d) / (2*a), (-b + d) / (2*a)
    p1 = (m1 / n - em2) / (em1 - em2)
    l1, l2 = 1 / em1, 1 / em2
    return l1, l2, p1
