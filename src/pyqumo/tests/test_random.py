import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyqumo.random import Const, Exponential, Uniform, Normal, Erlang, \
    HyperExponential, PhaseType


#
# TESTING VARIOUS DISTRIBUTIONS STATISTIC AND ANALYTIC PROPERTIES
# ----------------------------------------------------------------------------
# First of all we test statistical and analytic properties:
# - moments, k = 1, 2, 3, 4
# - variance
# - standard deviation
# - PDF (probability density function) at 3-5 points grid
# - CDF (cumulative distribution function) at 3-5 points grid
# - random samples generation
#
# To valid random samples generation we draw a large number of samples
# (100'000) and compute this sample set mean and variance using NumPy. Then
# we compare these estimation with expected mean and standard variance.
#
# We also validate that distribution is nicely printed.
#
# Since all distributions extend `Distribution` class and implement its
# methods, e.g. properties `var`, `mean`, methods `_moment()` and `_eval()`,
# we need to specify only the distribution itself and expected values.
# The test algorithm is the same. To achieve this, we use PyTest
# parametrization and define only one common test with a large number of
# parameters.
#
# Each parameters tuple specify one test: distribution, expected moments
# (four values), a grid for PDF, a grid for CDF and expected string form
# of the distribution.
@pytest.mark.parametrize('dist, m1, m2, m3, m4, xy_pdf, xy_cdf, string', [
    # Constant distribution:
    (
        Const(2), 2, 4, 8, 16,
        [(1.9, 0), (2.0, np.inf), (2.1, 0)], [(1.9, 0), (2.0, 1), (2.1, 1)],
        '(Const: value=2)'
    ), (
        Const(3), 3, 9, 27, 81,
        [(2, 0), (3, np.inf), (3.1, 0)], [(2, 0), (3.0, 1), (3.1, 1)],
        '(Const: value=3)'
    ),
    # Uniform distribution:
    (
        Uniform(0, 1), 0.5, 1/3, 1/4, 1/5,
        [(-1, 0), (0, 1), (0.5, 1), (1, 1), (2, 0)],
        [(-1, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1)],
        '(Uniform: a=0, b=1)'
    ), (
        Uniform(2, 10), 6, 124/3, 312, 2499.2,
        [(1, 0), (2, 0.125), (6, 0.125), (10, 0.125), (11, 0)],
        [(1, 0), (2, 0), (6, 0.5), (10, 1), (11, 1)],
        '(Uniform: a=2, b=10)'
    ),
    # Normal distribution:
    (
        Normal(0, 1), 0, 1, 0, 3,
        [(-2, 0.054), (-1, 0.242), (0, 0.399), (1, 0.242), (2, 0.054)],  # PDF
        [(-2, 0.023), (-1, 0.159), (0, 0.500), (1, 0.841), (2, 0.977)],  # CDF
        '(Normal: mean=0, std=1)'
    ), (
        Normal(1, 0.5), 1, 1.25, 1.75, 2.6875,
        [(0, 0.108), (1, 0.798), (1.25, 0.704), (1.5, 0.484), (2, 0.108)],
        [(0, 0.023), (1, 0.500), (1.25, 0.691), (1.5, 0.841), (2, 0.977)],
        '(Normal: mean=1, std=0.5)'
    ),
    # Exponential distribution:
    (
        Exponential(1.0), 1, 2, 6, 24,
        [(0, 1.0), (1, 1/np.e), (2, 0.135)],      # PDF
        [(0, 0.0), (1, 1 - 1/np.e), (2, 0.865)],  # CDF
        '(Exp: rate=1)'
    ), (
        Exponential(2.0), 1/2, 1/2, 6/8, 24/16,
        [(0, 2.0), (1, 0.271), (2, 0.037)],  # PDF
        [(0, 0.0), (1, 0.865), (2, 0.982)],  # CDF
        '(Exp: rate=2)'
    ),
    # Erlang distribution:
    (
        Erlang(1, 1), 1, 2, 6, 24,
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)],  # PDF
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)],  # CDF
        '(Erlang: shape=1, rate=1)'
    ), (
        Erlang(shape=5, rate=2.5), 2.0, 4.8, 13.44, 43.008,
        [(0, 0.000), (1, 0.334), (2, 0.439), (3, 0.182)],  # PDF
        [(0, 0.000), (1, 0.109), (2, 0.559), (3, 0.868)],  # CDF
        '(Erlang: shape=5, rate=2.5)'
    ),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]), 1, 2, 6, 24,
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)],  # PDF
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)],  # CDF
        '(HyperExponential: probs=[1], rates=[1])'
    ), (
        HyperExponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        0.39167, 0.33194, 0.4475694, 0.837384,
        [(0, 2.800), (0.25, 1.331), (0.5, 0.664), (1, 0.187)],
        [(0, 0.000), (0.25, 0.492), (0.5, 0.731), (1, 0.917)],
        '(HyperExponential: probs=[0.5, 0.2, 0.3], rates=[2, 3, 4])'
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0), 1, 2, 6, 24,
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)],  # PDF
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)],  # CDF
        '(PhaseType: s=[[-1.0]], p=[1.0])'
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        0.714286, 0.680272, 0.809848, 1.15693,
        [(0, 0.00), (0.5, 1.134), (1, 0.555), (1.5, 0.153)],  # PDF
        [(0, 0.00), (0.5, 0.350), (1, 0.790), (1.5, 0.950)],  # CDF
        '(PhaseType: '
        's=[[-4.2, 4.2, 0.0], [0.0, -4.2, 4.2], [0.0, 0.0, -4.2]], '
        'p=[1.0, 0.0, 0.0])'
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        0.39167, 0.33194, 0.4475694, 0.837384,
        [(0, 2.800), (0.25, 1.331), (0.5, 0.664), (1, 0.187)],  # PDF
        [(0, 0.000), (0.25, 0.492), (0.5, 0.731), (1, 0.917)],  # CDF
        '(PhaseType: '
        's=[[-2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, -4.0]], '
        'p=[0.5, 0.2, 0.3])'
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        0.718362, 1.01114, 2.12112, 5.92064,
        [(0, 1.30), (0.5, 0.710), (1, 0.355), (1.5, 0.174)],  # PDF
        [(0, 0.00), (0.5, 0.495), (1, 0.752), (1.5, 0.879)],  # CDF
        '(PhaseType: '
        's=[[-2.0, 1.0, 0.2], [0.5, -3.0, 1.0], [0.5, 0.5, -4.0]], '
        'p=[0.5, 0.4, 0.1])'
    ),
])
def test_distributions_props(dist, m1, m2, m3, m4, xy_pdf, xy_cdf, string):
    """
    Validate Uniform distribution.
    """
    var = m2 - m1**2
    std = var**0.5

    # Validate statistic properties:
    assert_allclose(dist.mean, m1, atol=1e-3, err_msg=string)
    assert_allclose(dist.var, var, atol=1e-3, err_msg=string)
    assert_allclose(dist.std, std, atol=1e-3, err_msg=string)
    assert_allclose(dist.moment(1), m1, atol=1e-3, err_msg=string)
    assert_allclose(dist.moment(2), m2, atol=1e-3, err_msg=string)
    assert_allclose(dist.moment(3), m3, atol=1e-3, err_msg=string)
    assert_allclose(dist.moment(4), m4, atol=1e-3, err_msg=string)

    # Validate PDF on the given grid:
    pdf = dist.pdf
    for x, y in xy_pdf:
        assert_allclose(pdf(x), y, atol=1e-3, err_msg=f'{string} PDF, x={x}')

    # Validate CDF on the given grid:
    cdf = dist.cdf
    for x, y in xy_cdf:
        assert_allclose(cdf(x), y, atol=1e-3, err_msg=f'{string} CDF, x={x}')

    # Validate that random generated sequence have expected mean and std:
    samples = dist(100000)
    assert_allclose(samples.mean(), m1, atol=0.01, rtol=5e-2, err_msg=string)
    assert_allclose(samples.std(), std, atol=0.01, rtol=5e-2, err_msg=string)

    # Validate string output:
    assert str(dist) == string


#
#
# def test_semi_markov_absorb():
#     mat = [[0, 0.5, 0.5], [0, 0, 0], [0, 0, 0]]
#     time = [Constant(10), Exponential(5), Exponential(20)]
#     initial_prob = [1, 0, 0]
#     dist = SemiMarkovAbsorb(mat=mat, time=time, p0=initial_prob)
#
#     # First we make sure that the (full) transitional matrix and absorbing
#     # state are calculated properly:
#     assert_allclose(
#         dist.trans_matrix,
#         [[0, 0.5, 0.5, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
#     )
#     assert dist.absorbing_state == 3
#     assert dist.order == 3
#
#     # Check that the mean value matches the expected:
#     expected_mean = 10 + 0.5 * 5 + 0.5 * 20
#     # assert_almost_equal(dist.mean(), expected_mean)
#
#     # Then we check that we can generate samples and estimate their properties:
#     samples = np.asarray([dist() for _ in range(500)])
#     assert_allclose(samples.mean(), expected_mean, rtol=0.15)
#
#
# @pytest.mark.parametrize('dists, w, mean, var, std', [
#     ([Exponential(10)], None, 10, 100, 10),
#     ([Exponential(3), Exponential(4)], None, 7, 25, 5),
#     ([Constant(1), 2, Exponential(3)], [4, 8, 2], 26, 36, 6),
# ])
# def test_lin_comb(dists, w, mean, var, std):
#     lc = LinComb(dists, w)
#
#     assert_allclose(lc.mean(), mean, rtol=0.05)
#     assert_allclose(lc.var(), var, rtol=0.05)
#     assert_allclose(lc.std(), std, rtol=0.05)
#
#     samples = lc.generate(2000)
#     assert_allclose(samples.mean(), mean, rtol=0.2)
#     assert_allclose(samples.var(), var, rtol=0.2)
#     assert_allclose(samples.std(), std, rtol=0.2)
#
#
# @pytest.mark.parametrize('dists, w', [
#     ([Exponential(1)], None),
#     ([Exponential(2), Exponential(3)], None),
#     ([Exponential(2), 4, Constant(3)], [0.1, 0.4, 0.5]),
#     ([Exponential(3), Normal(10, 1)], [10, 2])
# ])
# def test_var_choice(dists, w):
#     vc = VarChoice(dists, w)
#
#     n = len(dists)
#     p = [1/n] * n if w is None else list(np.asarray(w) / sum(w))
#     d = [dist if hasattr(dist, 'moment') else Constant(dist) for dist in dists]
#     m1 = sum(p[i] * d[i].mean() for i in range(n))
#     m2 = sum(p[i] * d[i].moment(2) for i in range(n))
#     m3 = sum(p[i] * d[i].moment(3) for i in range(n))
#     m4 = sum(p[i] * d[i].moment(4) for i in range(n))
#     var = m2 - (m1 ** 2)
#
#     assert_allclose(vc.mean(), m1)
#     assert_allclose(vc.var(), var)
#     assert_allclose(vc.std(), var ** 0.5)
#
#     assert_allclose(vc.moment(1), m1)
#     assert_allclose(vc.moment(2), m2)
#     assert_allclose(vc.moment(3), m3)
#     assert_allclose(vc.moment(4), m4)
#
#     # To be consistent, validate that moment argument is a natural number:
#     for k in (0, -1, 2.3):
#         with pytest.raises(ValueError) as excinfo:
#             vc.moment(k)
#         assert 'positive integer expected' in str(excinfo.value).lower()
#
#     samples = vc.generate(1000)
#     assert_allclose(samples.mean(), m1, rtol=0.1)
#
#
# def test_var_choice_generate_provides_values_from_distributions_only():
#     vc = VarChoice([Constant(34), Constant(42)])
#     samples = [vc() for _ in range(100)]
#     values = set(samples)
#     assert values == {34, 42}
#
#
# @pytest.mark.parametrize('mean,std', [(10, 2), (2.3, 1.1)])
# def test_normal_distribution(mean, std):
#     dist = Normal(mean, std)
#
#     m1 = mean
#     m2 = mean ** 2 + std ** 2
#     m3 = mean ** 3 + 3 * mean * (std ** 2)
#     m4 = mean ** 4 + 6 * (mean ** 2) * (std ** 2) + 3 * (std ** 4)
#
#     assert_allclose(dist.mean(), mean)
#     assert_allclose(dist.var(), std**2)
#     assert_allclose(dist.std(), std)
#     assert_allclose(dist.moment(1), m1)
#     assert_allclose(dist.moment(2), m2)
#     assert_allclose(dist.moment(3), m3)
#     assert_allclose(dist.moment(4), m4)
#
#     with pytest.raises(ValueError) as excinfo:
#         dist.moment(5)
#     assert 'four moments supported' in str(excinfo.value).lower()
#
#     # To be consistent, validate that moment argument is a natural number:
#     for k in (0, -1, 2.3):
#         with pytest.raises(ValueError) as excinfo:
#             dist.moment(k)
#         assert 'positive integer expected' in str(excinfo.value).lower()
#
#     samples = dist.generate(2000)
#     assert_allclose(samples.mean(), mean, rtol=0.2)
#     assert_allclose(samples.std(), std, rtol=0.2)
#
#     samples = np.asarray([dist() for _ in range(1000)])
#     assert_allclose(samples.mean(), mean, rtol=0.2)
#     assert_allclose(samples.std(), std, rtol=0.2)
#
#     assert str(dist) == f'N({mean},{std})'
#
#
# @pytest.mark.parametrize('a,b', [(0, 10), (23, 24), (30, 20)])
# def test_uniform_distribution(a, b):
#     dist = Uniform(a, b)
#
#     m1 = 0.5 * (a + b)
#     var = np.abs(b - a) ** 2 / 12
#     std = var ** 0.5
#     m2 = m1 ** 2 + var
#
#     assert_allclose(dist.mean(), m1)
#     assert_allclose(dist.var(), var)
#     assert_allclose(dist.std(), std)
#     assert_allclose(dist.moment(1), m1)
#     assert_allclose(dist.moment(2), m2)
#
#     with pytest.raises(ValueError) as excinfo:
#         dist.moment(3)
#     assert 'two moments supported' in str(excinfo.value).lower()
#
#     # To be consistent, validate that moment argument is a natural number:
#     for k in (0, -1, 2.3):
#         with pytest.raises(ValueError) as excinfo:
#             dist.moment(k)
#         assert 'positive integer expected' in str(excinfo.value).lower()
#
#     samples = dist.generate(2000)
#     assert_allclose(samples.mean(), m1, rtol=0.2)
#     assert_allclose(samples.std(), std, rtol=0.2)
#
#     samples = np.asarray([dist() for _ in range(1000)])
#     assert_allclose(samples.mean(), m1, rtol=0.2)
#     assert_allclose(samples.std(), std, rtol=0.2)
#
#     assert str(dist) == f'U({min(a, b)},{max(a, b)})'
#
#
# @pytest.mark.parametrize('xi,koef,offset', [
#     (Exponential(1), 1.0, 0.0),
#     (Exponential(2), 4.0, 3.0),
# ])
# def test_linear_transform(xi, koef, offset):
#     dist = LinearTransform(xi, koef, offset)
#
#     mean, std = xi.mean() * koef + offset, xi.std() * koef
#     moments = [mean, mean ** 2 + std ** 2]
#
#     assert_allclose(dist.mean(), mean)
#     assert_allclose(dist.std(), std)
#     assert_allclose(dist.var(), std ** 2)
#     assert_allclose([dist.moment(1), dist.moment(2)], moments)
#
#     with pytest.raises(ValueError) as excinfo:
#         dist.moment(3)
#     assert 'two moments supported' in str(excinfo.value).lower()
#
#     # To be consistent, validate that moment argument is a natural number:
#     for k in (0, -1, 2.3):
#         with pytest.raises(ValueError) as excinfo:
#             dist.moment(k)
#         assert 'positive integer expected' in str(excinfo.value).lower()
#
#     assert_allclose(dist.cdf(offset), xi.cdf(0))
#     assert_allclose(dist.cdf(offset + koef), xi.cdf(1))
#
#     print(f'dist.pdf(b) = {dist.pdf(offset)}')
#     print(f'xi.pdf(0) = {xi.pdf(0)}, k = {koef}')
#
#     assert_allclose(dist.pdf(offset), xi.pdf(0) / koef)
#     assert_allclose(dist.pdf(offset + koef), xi.pdf(1) / koef)
#
#     samples = np.asarray(list(dist.generate(50000)))
#     assert_allclose(samples.mean(), dist.mean(), rtol=0.1)
#     assert_allclose(samples.std(), dist.std(), rtol=0.1)
#
#     samples = np.asarray([dist() for _ in range(50000)])
#     assert_allclose(samples.mean(), dist.mean(), rtol=0.1)
#     assert_allclose(samples.std(), dist.std(), rtol=0.1)
#
