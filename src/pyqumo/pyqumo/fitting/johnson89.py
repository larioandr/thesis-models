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

from pyqumo.random import PhaseType
from pyqumo.stats import get_cv, get_skewness, get_noncentral_m3
from pyqumo.errors import BoundsError


def fit_mern2(
    moments: Sequence[float],
    strict: bool = True
) -> Tuple[PhaseType, np.ndarray]:
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
    n_floor = int(max(
        np.ceil(1 / cv**2),
        np.ceil((-gamma + 1/cv**3 + 1/cv + 2*cv) / (gamma - (cv - 1/cv)))
    )) + (2 if cv <= 1 else 0)

    def get_mern_props(m1_, m2_, m3_, n_):
        # Compute auxiliary variables:
        x = m1_ * m3_ - (n_ + 2) / (n_ + 1) * pow(m2_, 2)
        y = m2_ - (n_ + 1) / n_ * pow(m1_, 2)
        C = m1_ * x
        B = -(
            n_ * x +
            n_ * (n_+2) / (n_+1) * pow(y, 2) +
            (n_+2) * pow(m1_, 2) * y
        )
        A = n_ * (n_+2) * m1_ * y
        D = (B**2 - 4*A*C)**0.5

        # Compute Erlang mixture parameters:
        em1, em2 = (-B - D) / (2*A), (-B + D) / (2*A)
        p1_ = (m1_/n_ - em2) / (em1 - em2)
        l1_, l2_ = 1/em1, 1/em2
        return (l1_, l2_), p1_

    (l1_0, l2_0), p1_0 = get_mern_props(m1, m2, m3, n_floor)
    n = n_floor
    r0 = l2_0 / l1_0 if l1_0 > 0 else np.inf

    if r0 < 1e-3 or r0 > 1000 or p1_0 < 1e-3 or p1_0 > (1 - 1e-3):
        n += 1
        (l1, l2), p1 = get_mern_props(m1, m2, m3, n)
    else:
        l1, l2, p1 = l1_0, l2_0, p1_0

    # Build PH matrix and initial prob. dist.:
    mat = np.zeros((2*n, 2*n))
    for i in range(n):
        mat[i, i] = -l1
        mat[n+i, n+i] = -l2
        if i < n-1:
            mat[i, i+1] = l1
            mat[n+i, n+i+1] = l2
    probs = np.zeros(2*n)
    probs[0] = p1
    probs[n] = 1 - p1
    ph = PhaseType(mat, probs)

    # Estimate errors:
    errors = np.asarray([
        abs(m - ph.moment(i+1)) / abs(m) for i, m in enumerate(moments)
    ])

    return ph, errors
