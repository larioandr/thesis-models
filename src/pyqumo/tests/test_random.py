import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyqumo.random import Const, Exponential, Uniform, Normal, Erlang, \
    HyperExponential, PhaseType, Choice, SemiMarkovAbsorb, MixtureDistribution


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
# Some properties are specific to continuous or discrete distributions,
# e.g. PMF or PMF. To test them, we define separate tests.
#
# Each parameters tuple specify one test: distribution, expected moments
# (four values), a grid for CDF and expected string form
# of the distribution.
@pytest.mark.parametrize('dist, m1, m2, m3, m4, string, atol, rtol', [
    # Constant distribution:
    (Const(2), 2, 4, 8, 16, '(Const: value=2)', 1e-2, 2e-2),
    (Const(3), 3, 9, 27, 81, '(Const: value=3)', 1e-2, 2e-2),
    # Uniform distribution:
    (Uniform(0, 1), 0.5, 1/3, 1/4, 1/5, '(Uniform: a=0, b=1)', 1e-2, 2e-2),
    (Uniform(2, 10), 6, 124/3, 312, 2499.2, '(Uniform: a=2, b=10)', 1e-2, 2e-2),
    # Normal distribution:
    (Normal(0, 1), 0, 1, 0, 3, '(Normal: mean=0, std=1)', 1e-2, 2e-2),
    (
        Normal(1, 0.5),
        1, 1.25, 1.75, 2.6875,
        '(Normal: mean=1, std=0.5)',
        1e-2, 2e-2
    ),
    # Exponential distribution:
    (Exponential(1.0), 1, 2, 6, 24, '(Exp: rate=1)', 1e-2, 2e-2),
    (Exponential(2.0), 1/2, 1/2, 6/8, 24/16, '(Exp: rate=2)', 1e-2, 2e-2),
    # Erlang distribution:
    (Erlang(1, 1), 1, 2, 6, 24, '(Erlang: shape=1, rate=1)', 1e-2, 2e-2),
    (
        Erlang(5, rate=2.5),
        2, 4.8, 13.44, 43.008,
        '(Erlang: shape=5, rate=2.5)',
        1e-2, 2e-2
    ),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]), 1, 2, 6, 24,
        '(HyperExponential: probs=[1], rates=[1])',
        1e-2, 2e-2
    ), (
        HyperExponential([2, 3, 4], probs=[0.5, 0.2, 0.3]),
        0.39167, 0.33194, 0.4475694, 0.837384,
        '(HyperExponential: probs=[0.5, 0.2, 0.3], rates=[2, 3, 4])',
        1e-2, 2e-2
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0),
        1, 2, 6, 24,
        '(PhaseType: s=[[-1.0]], p=[1.0])',
        1e-2, 2e-2
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        0.714286, 0.680272, 0.809848, 1.15693,
        '(PhaseType: '
        's=[[-4.2, 4.2, 0.0], [0.0, -4.2, 4.2], [0.0, 0.0, -4.2]], '
        'p=[1.0, 0.0, 0.0])',
        1e-2, 2e-2
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        0.39167, 0.33194, 0.4475694, 0.837384,
        '(PhaseType: '
        's=[[-2.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, -4.0]], '
        'p=[0.5, 0.2, 0.3])',
        1e-2, 2e-2
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        0.718362, 1.01114, 2.12112, 5.92064,
        '(PhaseType: '
        's=[[-2.0, 1.0, 0.2], [0.5, -3.0, 1.0], [0.5, 0.5, -4.0]], '
        'p=[0.5, 0.4, 0.1])',
        1e-2, 2e-2
    ),
    # Choice (discrete) distribution
    (Choice([2]), 2, 4, 8, 16, '(Choice: values=[2], p=[1.0])', 1e-2, 2e-2),
    (
        Choice([3, 7, 5], weights=[1, 4, 5]),
        5.6, 33.0, 202.4, 1281.0,
        '(Choice: values=[3, 5, 7], p=[0.1, 0.5, 0.4])',
        1e-2, 2e-2
    ),
    # Semi-Markov Absorbing Process
    (
        SemiMarkovAbsorb(
            [[0]], [Exponential(1.0)], probs=0, num_samples=250000),
        1, 2, 6, 24,
        '(SemiMarkovAbsorb: trans=[[0, 1], [0, 1]], '
        'time=[(Exp: rate=1)], p0=[1])',
        1e-1, 1e-1
    ), (
        SemiMarkovAbsorb([[0]], [Normal(1, 0.5)], num_samples=250000),
        1, 1.25, 1.75, 2.6875,
        '(SemiMarkovAbsorb: trans=[[0, 1], [0, 1]], '
        'time=[(Normal: mean=1, std=0.5)], p0=[1])',
        1e-1, 1e-1
    ), (
        SemiMarkovAbsorb(
            trans=[[0, 0.5, 0.1], [1/6, 0, 1/3], [1/8, 1/8, 0]],
            time_dist=[Exponential(2), Exponential(3), Exponential(4)],
            probs=[0.5, 0.4, 0.1],
            num_samples=250000
        ),
        0.718362, 1.01114, 2.12112, 5.92064,
        '(SemiMarkovAbsorb: trans=['
        '[0, 0.5, 0.1, 0.4], [0.167, 0, 0.333, 0.5], '
        '[0.125, 0.125, 0, 0.75], [0, 0, 0, 1]'
        '], time=[(Exp: rate=2), (Exp: rate=3), (Exp: rate=4)], '
        'p0=[0.5, 0.4, 0.1])',
        1e-1, 1e-1
    ),
    # Mixture of constant distributions (choice):
    (
        MixtureDistribution(
            states=[Const(3), Const(7), Const(5)],
            weights=[1, 4, 5]
        ),
        5.6, 33.0, 202.4, 1281.0,
        '(Mixture: '
        'states=[(Const: value=3), (Const: value=7), (Const: value=5)], '
        'probs=[0.1, 0.4, 0.5])',
        1e-2, 2e-2,
    )
])
def test_common_props(dist, m1, m2, m3, m4, string, atol, rtol):
    """
    Validate common distributions properties: first four moments and repr.
    """
    var = m2 - m1**2
    std = var**0.5

    # Validate statistic properties:
    assert_allclose(dist.mean, m1, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.var, var, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.std, std, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(1), m1, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(2), m2, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(3), m3, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(dist.moment(4), m4, atol=atol, rtol=rtol, err_msg=string)

    # Validate that random generated sequence have expected mean and std:
    samples = dist(100000)
    assert_allclose(samples.mean(), m1, atol=atol, rtol=rtol, err_msg=string)
    assert_allclose(samples.std(), std, atol=atol, rtol=rtol, err_msg=string)

    # Validate string output:
    assert str(dist) == string


#
# VALIDATE CUMULATIVE DISTRIBUTION FUNCTIONS
# ------------------------------------------
@pytest.mark.parametrize('dist, grid', [
    # Constant distribution:
    (Const(2), [(1.9, 0), (2.0, 1), (2.1, 1)]),
    (Const(3), [(2, 0), (3, 1), (3.1, 1)]),
    # Uniform distribution:
    (Uniform(0, 1), [(-1, 0), (0, 0), (0.5, 0.5), (1, 1), (2, 1)]),
    (Uniform(2, 10), [(1, 0), (2, 0), (6, 0.5), (10, 1), (11, 1)]),
    # Normal distribution:
    (
        Normal(0, 1),
        [(-2, 0.023), (-1, 0.159), (0, 0.500), (1, 0.841), (2, 0.977)],
    ), (
        Normal(1, 0.5),
        [(0, 0.023), (1, 0.500), (1.25, 0.691), (1.5, 0.841), (2, 0.977)],
    ),
    # Exponential distribution:
    (Exponential(1.0), [(0, 0.0), (1, 1 - 1/np.e), (2, 0.865)]),
    (Exponential(2.0), [(0, 0.0), (1, 0.865), (2, 0.982)]),
    # Erlang distribution:
    (Erlang(1, 1), [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]),
    (Erlang(5, rate=2.5), [(0, 0.000), (1, 0.109), (2, 0.559), (3, 0.868)]),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]),
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]
    ), (
        HyperExponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 0.000), (0.25, 0.492), (0.5, 0.731), (1, 0.917)]
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0),
        [(0, 0.000), (1, 0.632), (2, 0.865), (3, 0.950)]
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        [(0, 0.00), (0.5, 0.350), (1, 0.790), (1.5, 0.950)]
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 0.000), (0.25, 0.492), (0.5, 0.731), (1, 0.917)]
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        [(0, 0.00), (0.5, 0.495), (1, 0.752), (1.5, 0.879)]
    ),
    # Choice (discrete) distribution
    (Choice([5]), [(4.9, 0), (5.0, 1), (5.1, 1)]),
    (
        Choice([3, 5, 7], weights=[1, 5, 4]),
        [(2, 0), (3, 0.1), (4.9, 0.1), (5, 0.6), (6.9, 0.6), (7, 1), (8, 1)]
    ),
])
def test_cdf(dist, grid):
    """
    Validate cumulative distribution function.
    """
    cdf = dist.cdf
    for x, y in grid:
        assert_allclose(cdf(x), y, atol=1e-3, err_msg=f'{dist} CDF, x={x}')


@pytest.mark.parametrize('dist, grid', [
    (
        SemiMarkovAbsorb([[0]], [Exponential(2.0)], probs=0,
                         num_kde_samples=20000),
        [(0.1, 0.181), (0.5, 0.632), (1.0, 0.865), (1.4, 0.939)]
    )
])
def test_gaussian_kde_cdf(dist, grid):
    """
    Validate cumulative distribution function when it is evaluated from
    sample data using Gaussian kernel.
    """
    cdf = dist.cdf
    for x, y in grid:
        assert_allclose(cdf(x), y, rtol=0.1, err_msg=f'{dist} CDF, x={x}')


#
# VALIDATE PROBABILITY DENSITY FUNCTIONS (CONT. DIST.)
# ----------------------------------------------------
@pytest.mark.parametrize('dist, grid', [
    # Constant distribution:
    (Const(2), [(1.9, 0), (2.0, np.inf), (2.1, 0)]),
    (Const(3), [(2, 0), (3, np.inf), (3.1, 0)]),
    # Uniform distribution:
    (Uniform(0, 1), [(-1, 0), (0, 1), (0.5, 1), (1, 1), (2, 0)]),
    (Uniform(2, 10), [(1, 0), (2, 0.125), (6, 0.125), (10, 0.125), (11, 0)]),
    # Normal distribution:
    (Normal(0, 1), [(-2, 0.054), (-1, 0.242), (0, 0.399), (1, 0.242)]),
    (Normal(1, 0.5), [(0, 0.108), (1, 0.798), (1.25, 0.704), (2, 0.108)]),
    # Exponential distribution:
    (Exponential(1.0), [(0, 1.0), (1, 1/np.e), (2, 0.135)]),
    (Exponential(2.0), [(0, 2.0), (1, 0.271), (2, 0.037)]),
    # Erlang distribution:
    (Erlang(1, 1), [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]),
    (Erlang(5, rate=2.5), [(0, 0.000), (1, 0.334), (2, 0.439), (3, 0.182)]),
    # Hyperexponential distribution
    (
        HyperExponential([1], [1]),
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]
    ), (
        HyperExponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 2.800), (0.25, 1.331), (0.5, 0.664), (1, 0.187)]
    ),
    # Phase-type distribution
    (
        PhaseType.exponential(1.0),
        [(0, 1.000), (1, 0.368), (2, 0.135), (3, 0.050)]
    ), (
        PhaseType.erlang(shape=3, rate=4.2),
        [(0, 0.00), (0.5, 1.134), (1, 0.555), (1.5, 0.153)]
    ), (
        PhaseType.hyperexponential(rates=[2, 3, 4], probs=[0.5, 0.2, 0.3]),
        [(0, 2.800), (0.25, 1.331), (0.5, 0.664), (1, 0.187)],  # PDF
    ), (
        PhaseType(np.asarray([[-2, 1, 0.2], [0.5, -3, 1], [0.5, 0.5, -4]]),
                  np.asarray([0.5, 0.4, 0.1])),
        [(0, 1.30), (0.5, 0.710), (1, 0.355), (1.5, 0.174)]
    ),
])
def test_pdf(dist, grid):
    """
    Validate continuous distribution probability density function.
    """
    pdf = dist.pdf
    for x, y in grid:
        assert_allclose(pdf(x), y, atol=1e-3, err_msg=f'{dist} PDF, x={x}')


@pytest.mark.parametrize('dist, grid', [
    (
        SemiMarkovAbsorb([[0]], [Exponential(2.0)], probs=0,
                         num_kde_samples=20000),
        [(0.1, 1.637), (0.5, 0.736), (1.0, 0.271), (1.4, 0.122), (2.0, 0.0366)]
    )
])
def test_gaussian_kde_pdf(dist, grid):
    """
    Validate probability density function when it is evaluated from
    sample data using Gaussian kernel.
    """
    pdf = dist.pdf
    for x, y in grid:
        assert_allclose(pdf(x), y, rtol=0.1, err_msg=f'{dist} PDF, x={x}')


#
# VALIDATE PROBABILITY MASS FUNCTIONS AND ITERATORS (DISCRETE DIST.)
# ------------------------------------------------------------------
@pytest.mark.parametrize('dist, grid', [
    # Constant distribution:
    (Const(2), [(2, 1.0)]),
    (Const(3), [(3, 1.0)]),
    # Choice (discrete) distribution:
    (Choice([10]), [(10, 1.0)]),
    (Choice([5, 7, 9], weights=[1, 5, 4]), [(5, 0.1), (7, 0.5), (9, 0.4)]),
])
def test_pmf_and_iterators(dist, grid):
    """
    Validate discrete distribution probability mass function and iterator.
    """
    pmf = dist.pmf
    for x, y in grid:
        assert_allclose(pmf(x), y, atol=1e-3, err_msg=f'{dist} PMF, x={x}')
    for i, (desired, actual) in enumerate(zip(grid, dist)):
        assert_allclose(actual[0], desired[0],
                        err_msg=f'{i}-th values mismatch, dist: {dist}')
        assert_allclose(actual[1], desired[1],
                        err_msg=f'{i}-th probability mismatch, dist: {dist}')


#
# CUSTOM PROPERTIES OF SIMPLE DISTRIBUTIONS
# ----------------------------------------------------------------------------
@pytest.mark.parametrize('choice, value, expected_index, comment', [
    (Choice([1]), 1, 0, 'choice of length 1, search for existing value'),
    (Choice([1]), 0, -1, 'choice of length 1, search for too small value'),
    (Choice([1]), 2, 0, 'choice of length 1, search for large value'),
    (Choice([1, 2]), 2, 1, 'choice of length 2, search for existing value'),
    (Choice([1, 2]), 0, -1, 'choice of length 2, search for too small value'),
    (Choice([1, 2]), 1.5, 0, 'choice of length 2, s.f. value in the middle'),
    (Choice([10, 20, 30, 40, 50]), 10, 0, 'choice len. 5, existing value #1'),
    (Choice([10, 20, 30, 40, 50]), 30, 2, 'choice len. 5, existing value #2'),
    (Choice([10, 20, 30, 40, 50]), 40, 3, 'choice len. 5, existing value #3'),
    (Choice([10, 20, 30, 40, 50]), 9, -1, 'choice len. 5, too small value'),
    (Choice([10, 20, 30, 40, 50]), 51, 4, 'choice len. 5, too large value'),
    (Choice([10, 20, 30, 40, 50]), 22, 1, 'choice len. 5, val inside'),
])
def test_choice_find_left(choice, value, expected_index, comment):
    assert choice.find_left(value) == expected_index, comment


#
# CUSTOM PROPERTIES OF SEMI-MARKOV ABSORBING PROCESS
# ----------------------------------------------------------------------------


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
