import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyqumo.fitting import fit_mern2


def get_m2(m1, c):
    """
    Helper: get M2 from M1 and Cv.
    """
    return (c**2 + 1) * m1**2


def get_m3(m1, c, gamma):
    """
    Helper: get M3 value from M1, Cv and skewness.
    """
    m2 = get_m2(m1, c)
    std = c * m1
    var = std**2
    return gamma * var * std + 3 * m1 * var + m1**3


@pytest.mark.parametrize('m1, cv, gamma', [
    # Best case: coef. of variation > 1.0
    (1, 1.5, 1.8),
    (20, 1.5, 1.8),  # the same as above, but with another mean
    (1, 1.1, 0.2),
    (1, 10, 12),
    
    # Close to exponential distribution:
    (1, 1, 0.01),  # very small skewness
    (2, 1, 100),  # large skewness

    # Worse case: coef. of variation < 1.0
    (1, 0.9, -0.01),
    (1, 0.1, -7),  # lots of state here
    (1, 0.1, 9),   # shold be less states then above (however, no check here)
])
def test_fit_mern2_with_strict_good_data(m1, cv, gamma):
    """
    Test Johnson and Taaffe algorithm implementation from [1] on reasonably
    good data. Since the feasible regions are defined in (C - 1/C, Gamma)
    axis, we specify arguments with C and Gamma, and get M2, M3 values
    from these.
    """
    m2, m3 = get_m2(m1, cv), get_m3(m1, cv, gamma)
    ph, errors = fit_mern2([m1, m2, m3])

    # Validate PH distribution properties:
    assert_allclose(ph.moment(1), m1)
    assert_allclose(ph.moment(2), m2)
    assert_allclose(ph.moment(3), m3)
    assert_allclose(ph.cv, cv)
    assert_allclose(ph.skewness, gamma)

    # Validate errors are close to zero:    
    assert_allclose(errors, np.zeros(3), atol=1e-5)
