import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyqumo.fitting import fit_mern2
from pyqumo.errors import BoundsError


#
# HELPERS
# ----------------------------------------------------------------------------
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


def get_boundary_gamma(c):
    return c - 1/c


# 
# TESTS FOR fit_mern2()
# ----------------------------------------------------------------------------
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
def test_fit_mern2__strict__good_data(m1, cv, gamma):
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


@pytest.mark.parametrize('m1, cv, gamma, err_str', [
    (1.0, 1.0, -0.01, "Skewness = -0.01 is too small for CV = 1"), 
    (1.0, 2.0, 1.4, "Skewness = 1.4 is too small for CV = 2"),
    (1.0, 0.2, -5, "Skewness = -5 is too small for CV = 0.2"),
])
def test_fit_mern2__strict__infeasible_raise_error(m1, cv, gamma, err_str):
    """
    Test that for values in infeasible region (see Fig. 1 in [1]) 
    `BoundsError` exception is raised.
    """
    m2, m3 = get_m2(m1, cv), get_m3(m1, cv, gamma)
    with pytest.raises(BoundsError) as err:
        fit_mern2([m1, m2, m3])
    assert str(err.value).startswith(err_str)


@pytest.mark.parametrize('moments, err_str', [
    ([1.0], "Expected three moments, but 1 found"),
    ([1.0, 2.0], "Expected three moments, but 2 found"),
])
def test_fit_mern2__strict__require_at_least_three_moments(moments, err_str):
    """
    Validate that calling fit_mer2() with less then three moments raises
    ValueError.
    """
    with pytest.raises(ValueError) as err:
        fit_mern2(moments)
    assert str(err.value) == err_str


@pytest.mark.parametrize('moments, min_errors, comment', [
    ([1., 2., 6., 20., 110.], [0., 0., 0., 0.05, 0.05], "Exp. with 5 momnets"),
])
def test_fit_mern2__strict__compute_errors(moments, min_errors, comment):
    """
    Validate that fit_mern2() computes errors when even more then three
    moments passed. These errors are returned in the call.
    """
    ph, real_errors = fit_mern2(moments)
    assert all(real_errors >= min_errors)
