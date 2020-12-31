import pytest
from numpy import float64, asarray, ndarray, int64, inf
from numpy.testing import assert_allclose

from pyqumo.errors import RowSumsError
from pyqumo.matrix import fix_stochastic, CellValueError, \
    RowSumError, row2string, matrix2string, array2string, \
    parse_array, is_square, is_vector, MatrixShapeError, order_of, \
    is_stochastic, is_infinitesimal, is_subinfinitesimal, \
    check_markovian_arrival, fix_infinitesimal, fix_markovian_arrival


# ############################################################################
# TEST MATRIX-STRING CONVERTING ROUTINES
# ############################################################################

@pytest.mark.parametrize('value, string, sep', [
    ([], "", ","),
    ([1], "1", ","),
    ([1.0, 2.0], "1.0,2.0", ","),
    ([1, 2], "1;2", ";"),
    (asarray([1.2, 3.4, 5.6]), "1.2:3.4:5.6", ":")
])
def test_row2string(value, string, sep):
    assert row2string(value, sep) == string


@pytest.mark.parametrize('value, string, row_sep, col_sep', [
    ([[]], "", ",", ";"),
    ([[1]], "1", ";", ","),
    ([[1, 2], [3, 4]], "1,2;3,4", ";", ","),
    (asarray([[1.1, 2.2], [3.3, 4.4]]), "1.1-2.2:3.3-4.4", ":", "-")
])
def test_matrix2string(value, string, row_sep, col_sep):
    assert matrix2string(value, row_sep=row_sep, col_sep=col_sep) == string


@pytest.mark.parametrize('value, string, row_sep, col_sep', [
    ([], "", ";", ","),
    ([[]], "", ";", ","),
    ([1], "1", ";", ","),
    ([[2]], "2", ";", ","),
    ([1, 2], "1,2", ";", ","),
    ([[1, 2]], "1,2", ";", ","),
    ([[1], [2]], "1;2", ";", ","),
    (asarray([[1, 2], [3, 4]]), "1:2\n3:4", "\n", ":")
])
def test_array2string(value, string, row_sep, col_sep):
    assert array2string(value, row_sep=row_sep, col_sep=col_sep) == string


@pytest.mark.parametrize('string, value, row_sep, col_sep, dtype', [
    ('', [], ';', ',', int64),
    ('1', [1], ';', ',', int64),
    ('1,2', [1, 2], ';', ',', int64),
    ('1,2', [[1], [2]], ',', ';', int64),
    ('1->2||3->4', [[1, 2], [3, 4]], '||', '->', float64)
])
def test_parse_array(string, value, row_sep, col_sep, dtype):
    result = parse_array(string, row_sep=row_sep, col_sep=col_sep, dtype=dtype)
    assert isinstance(result, ndarray)
    assert result.dtype == dtype
    assert_allclose(result, value, rtol=1e-12)


# ############################################################################
# TEST MATRIX SHAPE ROUTINES
# ############################################################################
@pytest.mark.parametrize('mat, expected, comment', [
    ([[1]], True, '1x1 matrix is a square matrix'),
    ([[]], False, '1x0 matrix is NOT a square matrix'),
    ([1], False, '1d matrix with one element is NOT a square'),
    ([1, 2, 3], False, '1d matrix with N > 1 elements is NOT a square'),
    ([[1], [2]], False, '2x1 is a column vector, NOT a square'),
    ([[1, 2], [3, 4]], True, '2x2 matrix is a square')
])
def test_is_square(mat, expected, comment):
    mat = asarray(mat)
    result = is_square(mat)
    assert result == expected, comment


@pytest.mark.parametrize('mat, expected, comment', [
    ([[1]], True, '1x1 matrix is a vector'),
    ([[]], True, 'formally, 1x0 matrix is a vector'),
    ([1, 2, 3], True, '1d array is a vector'),
    ([[1], [2]], True, 'Nx1 matrix is a column vector'),
    ([[1, 2], [3, 4]], False, '2d matrix (2x2) is NOT a vector')
])
def test_is_vector(mat, expected, comment):
    mat = asarray(mat)
    result = is_vector(mat)
    assert result == expected, comment


@pytest.mark.parametrize('mat, expected, error_string, comment', [
    ([[1]], 1, None, 'number of elements in a row'),
    ([2, 4, 6], 3, None, 'number of elements in a vector'),
    ([[3], [4], [5]], 3, None, 'number of elements in a column'),
    ([[1, 2], [3, 4]], 2, None, 'order of square matrix (2)'),
    # In the last test, raise an error due to incorrect matrix shape:
    ([[1, 2, 3], [4, 5, 6]], None,
     "matrix shape error: expected (N,) or (N, N), but (2, 3) found", None)
])
def test_order_of(mat, expected, error_string, comment):
    mat = asarray(mat)
    if error_string is None:
        result = order_of(mat)
        assert result == expected, comment
    else:
        with pytest.raises(MatrixShapeError) as e:
            order_of(mat)
        assert error_string == str(e.value)


# ############################################################################
# TEST MATRIX INSPECTION ROUTINES
# ############################################################################
@pytest.mark.parametrize('mat, expected, comment', [
    ([[]], False, 'empty matrix (1x0) is NOT stochastic'),
    ([[1.0]], True, 'degenerate case of 1x1 stochastic matrix'),
    ([0.2, 0.8], True, 'probability mass function is stochastic'),
    ([[0.2, 0.8, 0.0], [0.5, 0.2, 0.3]], True, 'stochastic 2x3 matrix'),
    ([[0.2, 0.8], [0.5, 0.4]], False, 'not stochastic - 2nd row sum is 0.9')
])
def test_is_stochastic(mat, expected, comment):
    mat = asarray(mat)
    assert is_stochastic(mat) == expected, comment


@pytest.mark.parametrize('mat, expected, comment', [
    ([[]], False, 'empty matrix (1x0) is NOT infinitesimal'),
    ([[0]], True, 'degenerate case of 1x1 infinitesimal matrix'),
    ([[-1.0, 1.0], [0.0, 0.0]], True, 'infinitesimal 2x2 matrix'),
    ([0], False, 'vector can NOT be infinitesimal'),
    ([[-1, 0.5, 0.5], [0, 0, 0]], False, 'not square => NOT infinitesimal'),
    ([[-1.0, 1.1], [0.9, -1.0]], False, '2nd row sum is NOT zero'),
])
def test_is_infinitesimal(mat, expected, comment):
    mat = asarray(mat)
    assert is_infinitesimal(mat) == expected, comment


@pytest.mark.parametrize('mat, expected, comment', [
    ([[]], False, 'empty matrix (1x0) is NOT subinfinitesimal'),
    ([[0]], False, 'row with sum < 0 required (1x1 case)'),
    ([[-1, 1], [2, -2]], False, 'row with sum < 0 required (2x2 case)'),
    ([[-1.0, 1.0], [0.5, -1.0]], True, 'good 2x2 subinfinitesimal matrix'),
    ([-2, 1], False, 'vector can not be subinfinitesimal'),
    ([[-2, 1]], False, 'only square matrix can be subinfinitesimal')
])
def test_is_subinfinitesimal(mat, expected, comment):
    mat = asarray(mat)
    assert is_subinfinitesimal(mat) == expected, comment


@pytest.mark.parametrize('matrices, expected, comment', [
    (([[0]],), False, 'at least two matrices required'),
    (([[-1]], [[1]]), True, 'Poisson arrival with rate 1 - degenerate case'),
    (([-1], [1]), False, 'all matrices must not be vectors'),
    (([[-1, 0]], [[0, 1]]), False, 'require square matrices'),
    (([[-1, 0], [1, -1]], [[1]]), False, 'all orders must be the same'),
    (([[0]], [[0]]), False, 'first matrix must be strictly subinfinitesimal'),
    (([[-2, 1], [1, -1]], [[0.5, 0.5], [0, 0]]), True, 'valid MAP(2)'),
    (([[-2, 1], [0, 0]], [[1, 0], [0, 0]]), True, 'valid MAP with zero row'),
    (
        ([[-3, 0], [0, -2]], [[1, 1], [1, 0]], [[1, 0], [0, 1]]),
        True,
        'valid MMAP(2) or batch MAP(2) matrices'
    ), (
        ([[-2, 0], [0, -2]], [[1, 1], [1, 0]]),
        False,
        'not MAP matrices - 2nd row sum is not zero'
    ), (
        ([[-2, 0], [0, -2]], [[-1, 3], [2, 0]]),
        False,
        'not MAP matrices - negative element in D1'
    ), (
        ([[1, 0], [0, -1]], [[0, 0], [0, 1]]),
        False,
        'not MAP matrices - first matrix must be subinfinitesimal'
    )
])
def test_check_markovian_arrival_matrices(matrices, expected, comment):
    matrices = [asarray(mat) for mat in matrices]
    assert check_markovian_arrival(matrices) == expected, comment


# ############################################################################
# TEST SPECIAL MATRIX FIXING ROUTINES
# ############################################################################

#
# TEST fix_stochastic
# ----------------------------------------------------------------------------

def test_fix_stochastic__fixable_matrix():
    """
    Validate test_stochastic() function with a fixable matrix.
    """
    mat = asarray([
        [-0.08,  0.02, 0.72, 0.12],
        [-0.04, -0.01, 0.03, 0.76],
        [1.09, 0.00, 0.02, 0.00],
        [0.40, 0.40, 0.00, 0.20]
    ])
    max_err = float64(0.12)

    # The rows will be fixed in this way:
    # 1) row 0:
    #    [...] => {+0.08} => [0, 0.10, 0.80, 0.20] =>
    #          => {/1.10} => [0, 0.10/1.10, 0.80/1.10, 0.20/1.10]
    # 2) row 1:
    #    [...] => {+0.04} => [0, 0.03, 0.07, 0.80] =>
    #          => {/0.90} => [0, 0.03/0.90, 0.07/0.90, 0.80/0.90]
    # 3) row 2:
    #    [...] => {/1.11} => [1.09/1.11, 0, 0.01/1.11, 0]
    # 4) row 3: is ok without modifications
    #
    # Maximum error of elements is: 0.08
    # Maximum error of row sums is: 0.11

    fixed_mat, max_elem_err, max_row_err = fix_stochastic(mat, max_err=max_err)

    assert_allclose(fixed_mat, [
        [0, 0.1/1.1, 0.8/1.1, 0.2/1.1],
        [0, 0.03/0.9, 0.07/0.9, 0.8/0.9],
        [1.09/1.11, 0, 0.02/1.11, 0],
        [0.4, 0.4, 0, 0.2]
    ], rtol=1e-9)
    assert_allclose(max_elem_err, 0.08)
    assert_allclose(max_row_err, 0.11)


@pytest.mark.parametrize('mat, result, elem_err, row_err, tol, comment', [
    ([-0.02, 0.28, 0.73], [0, 0.3/1.05, 0.75/1.05], 0.02, 0.05, 0.051,
     'negative elements are added, then row is divided'),
    ([0.99], [1.0], 0.0, 0.01, 0.02,
     'row with sum less then 1.0 should be divided')
])
def test_fix_stochastic__fixable_vector(mat, result, elem_err, row_err, tol,
                                        comment):
    """
    Validate test_stochastic() correctly processes a almost stochastic vector.
    """
    mat = asarray(mat)
    fixed_mat, max_elem_err, max_row_err = fix_stochastic(mat, max_err=tol)
    assert_allclose(fixed_mat, result, err_msg=comment)
    assert_allclose(max_elem_err, elem_err, err_msg=comment)
    assert_allclose(max_row_err, row_err, err_msg=comment)


def test_fix_stochastic__too_small_element_raise_error():
    """
    Validate test_stochastic() raises error when element is too negative.
    """
    with pytest.raises(CellValueError) as excinfo:
        fix_stochastic(asarray([[0.02, -0.08, 0.72, 0.12]]), max_err=0.07)
    ex = excinfo.value
    assert ex.row == 0
    assert ex.col == 1
    assert_allclose(ex.value, -0.08)
    assert_allclose(ex.error, 0.08)
    assert_allclose(ex.tol, 0.07)
    assert_allclose(ex.upper, 1.0)
    assert_allclose(ex.lower, 0.0)
    assert str(ex) == \
           f"cell (0, 1) value {ex.value} is out of bounds (0.0, 1.0)\n" \
           f"\terror: {ex.error}, tolerance: 0.07"


def test_fix_stochastic__row_sum_too_large():
    """
    Validate test_stochastic() raises error when row sum is too large.
    """
    with pytest.raises(RowSumError) as excinfo:
        fix_stochastic(asarray([[0, 1], [0.8, 0.25]]), max_err=0.04)
    ex = excinfo.value
    assert ex.row == 1
    assert_allclose(ex.actual, 1.05)
    assert_allclose(ex.desired, 1.0)
    assert_allclose(ex.error, 0.05)
    assert_allclose(ex.tol, 0.04)
    assert str(ex) == \
        f"sum of row 1 elements is {ex.actual}, expected 1.0\n" \
        f"\terror: {ex.error}, tolerance: 0.04"


#
# TEST fix_infinitesimal
# ----------------------------------------------------------------------------
def test_fix_infinitesimal__good_infinitesimal_matrix():
    """
    Validate fix_infinitesimal() fixes a matrix that is almost infinitesimal.
    """
    mat = asarray([
        [-2.0, 1.5, 0.5, 0.0, 0.0],    # good row, no modifications
        [-0.1, -0.9, 0.4, 0.5, 0.0],   # set -0.1 to 0.0, err = 0.1
        [0.0, 2.0, -3.0, 1.2, 0.0],    # set diagonal to -3.2, err = 0.2
        [0.5, -0.15, 3.4, -4.0, 0.0],  # -0.15 => 0.0 -4.0 => -3.9, err = 0.15
        [0.0, 0.0, 0.0, 0.0, 0.0]      # good row, no modifications
    ])
    TOL = 0.21  # maximum error is 0.21
    fixed_mat, err = fix_infinitesimal(mat, tol=TOL)
    assert_allclose(fixed_mat, [
        [-2.0, 1.5, 0.5, 0.0, 0.0],
        [0.0, -0.9, 0.4, 0.5, 0.0],
        [0.0, 2.0, -3.2, 1.2, 0.0],
        [0.5, 0.0, 3.4, -3.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.0]
    ])
    assert_allclose(err, 0.2)


def test_fix_infinitesimal__good_subinfinitesimal_matrix():
    """
    Validate fix_infinitesimal() fixes a matrix that is almost subinfinitesimal.
    """
    mat = asarray([
        [-2.0, 1.7, 0.6],    # set diagonal to -2.3, err = 0.3
        [-0.1, -1.0, 0.4],   # set -0.1 to 0.0, err = 0.1
        [0.0, 2.0, -5.0],    # good row
    ])
    TOL = 0.31  # maximum error is 0.21
    fixed_mat, err = fix_infinitesimal(mat, tol=TOL, sub=True)
    assert_allclose(fixed_mat, [
        [-2.3, 1.7, 0.6],
        [0.0, -1.0, 0.4],
        [0.0, 2.0, -5.0]
    ])
    assert_allclose(err, 0.3)


def test_fix_infinitesimal__raise_error_for_too_small_elements():
    """
    Validate fix_infinitesimal() raises CellValueError
    when some element is less then -tol
    """
    mat = [[-1.0, -0.101, 1.0], [1.0, -1.0, -0.11], [0, -0.09, 0]]
    with pytest.raises(CellValueError) as excinfo:
        fix_infinitesimal(asarray(mat), tol=0.1)
    ex = excinfo.value
    assert ex.row == 1
    assert ex.col == 2
    assert_allclose(ex.value, -0.11)
    assert_allclose(ex.lower, 0)
    assert ex.upper == inf
    assert_allclose(ex.error, 0.11)
    assert_allclose(ex.tol, 0.10)
    assert str(ex) == \
        f"cell (1, 2) value {ex.value} is out of bounds (0, inf)\n" \
        f"\terror: {ex.error}, tolerance: 0.1"


def test_fix_infinitesimal__raise_error_for_wrong_diagonal_elements():
    """
    Validate fix_infinitesimal() raises RowSumError when
    sum of row elements differs from zero for more then tolerance.
    """
    mat = [[-1.21, 1.0], [5, -5.23]]
    with pytest.raises(RowSumError) as excinfo:
        fix_infinitesimal(asarray(mat), tol=0.2)
    ex = excinfo.value
    assert ex.row == 1
    assert_allclose(ex.actual, -0.23)
    assert_allclose(ex.desired, 0)
    assert_allclose(ex.error, 0.23)
    assert_allclose(ex.tol, 0.2)
    assert str(ex) == \
           f"sum of row 1 elements is {ex.actual}, expected 0.0\n" \
           f"\terror: {ex.error}, tolerance: 0.2"


def test_fix_infinitesimal__raise_error_for_non_square_matrix():
    """
    Validate fix_infinitesimal() raise MatrixShapeError if the given matrix
    shape is not square.
    """
    with pytest.raises(MatrixShapeError) as excinfo:
        fix_infinitesimal(asarray([[-1, 1, 0], [0, 0, 0]]))
    ex = excinfo.value
    assert ex.expected == '(N, N)'
    assert ex.shape == (2, 3)
    assert str(ex) == \
           'matrix shape error: expected (N, N), but (2, 3) found'
    # Also check that 1D array with 1 element is not good:
    with pytest.raises(MatrixShapeError):
        fix_infinitesimal(asarray([0]))


def test_fix_infinitesimal__raise_error_when_row_sums_is_zero_and_sub_true():
    """
    Validate fix_infinitesimal() raise ZeroRowSumsError if the given matrix
    doesn't contain rows with negative sum when `sub = True`.
    """
    # This matrix becomes infinitesimal after fixing:
    mat = [[-1, 1.1, 0], [-0.1, -2, 2], [0, 0, 0]]
    with pytest.raises(RowSumsError) as excinfo:
        fix_infinitesimal(asarray(mat), tol=0.2, sub=True)
    assert str(excinfo.value) == 'all rows sums are zero: [0. 0. 0.]'


#
# TEST fix_infinitesimal
# ----------------------------------------------------------------------------
def test_fix_markov_arrival__good_fixable_map():
    """
    Validate fix_markovian_arrival() with D0 and D1 properly fix them when
    errors are not too large.
    """
    d0 = [
        [-2, 1, 0, 0, 0],    # row sum is 0.0, error = 0.0
        [0.2, -1, 0, 0, 0],  # row sum is 0.1, error = 0.1
        [-0.1, 0, -1, 0, 0],  # negative element, error = 0.1
        [-0.05, -0.15, 0.5, -2, 0],  # neg. elements, error = 0.15
        [0, 0, 0, 0, 0]
    ]
    d1 = [
        [0, 0, 1, 0, 0],
        [0.4, 0, 0.5, 0, 0],  # row sum is 0.1, error = 0.1
        [0, -0.02, 0, 0.9, 0],  # negative element, error = 0.02
        [0.5, 0.5, 0.02, 0.5, 0],  # row sum is 0.05, error = 0.02
        [0, 0, 0, 0, 0]
    ]
    TOL = 0.2

    matrices = [asarray(d0), asarray(d1)]
    fixed_matrices, error = fix_markovian_arrival(matrices, tol=TOL)

    assert_allclose(fixed_matrices[0], [
        [-2, 1, 0, 0, 0],
        [0.2, -1.1, 0, 0, 0],
        [0, 0, -0.9, 0, 0],
        [0, 0, 0.5, -2.02, 0],
        [0, 0, 0, 0, 0]
    ])
    assert_allclose(fixed_matrices[1], [
        [0, 0, 1, 0, 0],
        [0.4, 0, 0.5, 0, 0],
        [0, 0, 0, 0.9, 0],
        [0.5, 0.5, 0.02, 0.5, 0],
        [0, 0, 0, 0, 0]
    ])
    assert_allclose(error, 0.15)


def test_fix_markov_arrival__raise_error_for_too_small_elements():
    """
    Validate fix_markov_arrival() raise CellValueError when either
    non-diagonal element of D0, or any element of Di, i > 0, is smaller
    then -tol.
    """
    d0 = [[-1, -0.1], [0, -2]]
    d1 = [[0, 1], [1, -0.15]]
    d2 = [[0, 0], [-0.2, 1]]
    matrices = [asarray(di) for di in (d0, d1, d2)]
    with pytest.raises(CellValueError) as excinfo:
        fix_markovian_arrival(matrices, tol=0.11)
    ex = excinfo.value
    assert ex.row == 1
    assert ex.col == 0
    assert ex.label == 'D2'
    assert_allclose(ex.error, 0.2)
    assert_allclose(ex.value, -0.2)
    assert_allclose(ex.tol, 0.11)
    assert_allclose(ex.lower, 0)
    assert ex.upper == inf
    assert str(ex) == \
           f"D2 cell (1, 0) value {ex.value} is out of bounds (0, inf)\n" \
           f"\terror: {ex.error}, tolerance: 0.11"


def test_fix_markov_arrival__raise_error_for_wrong_diagonal_elements():
    """
    Validate fix_markov_arrival() raises RowSumError when
    sum of row elements differs from zero for more then tolerance.
    """
    d0 = [[-2.2, 1.0], [0, -1.1]]
    d1 = [[1.0, 0.0], [0.5, 0.5]]
    matrices = [asarray(di) for di in (d0, d1)]
    with pytest.raises(RowSumError) as excinfo:
        fix_markovian_arrival(matrices, tol=0.05)
    ex = excinfo.value
    assert ex.row == 0
    assert_allclose(ex.actual, -0.2)
    assert_allclose(ex.desired, 0)
    assert_allclose(ex.error, 0.2)
    assert_allclose(ex.tol, 0.05)
    assert str(ex) == \
           f"sum of row 0 elements is {ex.actual}, expected 0.0\n" \
           f"\terror: {ex.error}, tolerance: 0.05"


def test_fix_markov_arrival__raise_error_when_less_then_two_matrices_given():
    """
    Validate fix_markov_arrival() raises ValueError when less then 2 matrices
    provided.
    """
    with pytest.raises(ValueError):
        fix_markovian_arrival([], tol=0.1)
    with pytest.raises(ValueError):
        fix_markovian_arrival((asarray([[0]]),), tol=0.1)


def test_fix_markov_arrival__raise_error_for_non_square_d0():
    """
    Validate fix_markov_arrival() raises MatrixShapeError when D0 is not square.
    """
    d0 = [[-1, 0.5]]
    d1 = [[0, 0.5]]
    matrices = [asarray(di) for di in (d0, d1)]
    with pytest.raises(MatrixShapeError) as excinfo:
        fix_markovian_arrival(matrices, tol=0.1)
    ex = excinfo.value
    assert ex.expected == "(N, N)"
    assert ex.shape == (1, 2)
    assert ex.label == "D0"
    assert str(ex) == "D0 matrix shape error: expected (N, N), but (1, 2) found"


def test_fix_markov_arrival__raise_error_if_matrices_have_different_shape():
    """
    Validate fix_markov_arrival() raises MatrixShapeError when any Di, i > 0,
    has different shape then D0.
    """
    d0 = [[-1, 0], [0.2, -0.5]]
    d1 = [[1, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    matrices = [asarray(di) for di in (d0, d1)]
    with pytest.raises(MatrixShapeError) as excinfo:
        fix_markovian_arrival(matrices, tol=0.1)
    ex = excinfo.value
    assert ex.expected == (2, 2)
    assert ex.shape == (3, 3)
    assert ex.label == "D1"
    assert str(ex) == "D1 matrix shape error: expected (2, 2), but (3, 3) found"


# class TestIsPDF(TestCase):
#     def test_correct_list_matrices(self):
#         a1 = [1.0]
#         a2 = [0.5, 1.0]
#         a23 = [[0.2, 0.7, 1.0], [0.1, 0.9, 1.0]]
#         self.assertTrue(qumo.is_pdf(a1))
#         self.assertTrue(qumo.is_pdf(a2))
#         self.assertTrue(qumo.is_pdf(a23))
#
#     def test_correct_ndarray_matrices(self):
#         a1 = np.array([1.0])
#         a2 = np.array([0.5, 1.0])
#         a23 = np.array([[0.2, 0.7, 1.0], [0.1, 0.9, 1.0]])
#         self.assertTrue(qumo.is_pdf(a1))
#         self.assertTrue(qumo.is_pdf(a2))
#         self.assertTrue(qumo.is_pdf(a23))
#
#     def test_returns_false_when_some_elements_are_negative(self):
#         a2 = [-0.1, 0.2, 1.0]
#         a23 = [[0.0, 0.5, 1.0], [-0.2, 0.4, 1.0]]
#         self.assertFalse(qumo.is_pdf(a2))
#         self.assertTrue(qumo.is_pdf(a2, atol=0.11))
#         self.assertFalse(qumo.is_pdf(a23))
#         self.assertTrue(qumo.is_pdf(a23, atol=0.21))
#
#     def test_returns_false_when_not_ascending(self):
#         a = [0.5, 0.4, 1.0]
#         self.assertFalse(qumo.is_pdf(a))
#         self.assertTrue(qumo.is_pdf(a, atol=0.11))
#
#     def test_returns_false_when_rightmost_element_less_than_one(self):
#         a = [0.5, 0.6, 0.95]
#         self.assertFalse(qumo.is_pdf(a))
#         self.assertTrue(qumo.is_pdf(a, atol=0.1))
#
#     def test_returns_false_when_rightmost_element_greater_than_one(self):
#         a = [0.5, 0.6, 1.05]
#         self.assertFalse(qumo.is_pdf(a))
#         self.assertTrue(qumo.is_pdf(a, atol=0.1))
#
#
# class TestPmfPdfConverters(TestCase):
#     def test_pmf2pdf_list_matrices(self):
#         p1 = [1.0]
#         p2 = [0.1, 0.9]
#         p3 = [0.4, 0.0, 0.3, 0.3]
#         p23 = [[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]]
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p1), [1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p2), [0.1, 1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p3),
#                                           [0.4, 0.4, 0.7, 1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p23),
#                                           [[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]]))
#
#     def test_pmf2pdf_ndarray_matrices(self):
#         p1 = np.array([1.0])
#         p2 = np.array([0.1, 0.9])
#         p3 = np.array([0.4, 0.0, 0.3, 0.3])
#         p23 = np.array([[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]])
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p1), [1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p2), [0.1, 1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p3),
#                                           [0.4, 0.4, 0.7, 1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pmf2pdf(p23),
#                                           [[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]]))
#
#     def test_pdf2pmf_list_matrices(self):
#         p1 = [1.0]
#         p2 = [0.1, 1.0]
#         p3 = [0.4, 0.4, 0.7, 1.0]
#         p23 = [[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]]
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p1), [1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p2), [0.1, 0.9]))
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p3),
#                                           [0.4, 0.0, 0.3, 0.3]))
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p23),
#                                           [[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]]))
#
#     def test_pdf2pmf_ndarray_matrices(self):
#         p1 = np.array([1.0])
#         p2 = np.array([0.1, 1.0])
#         p3 = np.array([0.4, 0.4, 0.7, 1.0])
#         p23 = np.array([[0.2, 0.3, 1.0], [0.4, 0.4, 1.0]])
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p1), [1.0]))
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p2), [0.1, 0.9]))
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p3),
#                                           [0.4, 0.0, 0.3, 0.3]))
#         self.assertTrue(qumo.almost_equal(qumo.pdf2pmf(p23),
#                                           [[0.2, 0.1, 0.7], [0.4, 0.0, 0.6]]))
#
#
#
# class TestBlockMatrixBuilders(TestCase):
#
#     def test_cbmat_1x1(self):
#         b0 = [[0.0]]
#         b1 = [[1.0]]
#         b2 = [[2.0]]
#         b3 = [[3.0]]
#         b4 = [[4.0]]
#         blocks1 = [(0, 0, b0), (0, 1, b1), (1, 0, b2), (1, 1, b3)]
#         blocks2 = [(0, 0, b3), (0, 1, b2), (1, 0, b1), (1, 1, b0)]
#         blocks3 = [(0, 0, b1), (0, 2, b2), (3, 0, b3), (4, 2, b4)]
#
#         m1 = qumo.cbmat(blocks1)
#         m2 = qumo.cbmat(blocks2)
#         m3 = qumo.cbmat(blocks3)
#
#         np.testing.assert_allclose(m1, [[0., 1.], [2., 3.]])
#         np.testing.assert_allclose(m2, [[3., 2.], [1., 0.]])
#         np.testing.assert_allclose(m3, [[1, 0, 2], [0, 0, 0],
#                                         [0, 0, 0], [3, 0, 0],
#                                         [0, 0, 4]])
#
#     def test_cbmat_2x2(self):
#         b1 = [[1, 1], [1, 1]]
#         b2 = [[2, 2], [2, 2]]
#         b3 = [[3, 3], [3, 3]]
#         b4 = [[4, 4], [4, 4]]
#         blocks1 = [(0, 0, b1), (0, 1, b2), (1, 0, b3), (1, 1, b4)]
#         blocks2 = [(0, 0, b4), (0, 1, b3), (1, 0, b2), (1, 1, b1)]
#         blocks3 = [(0, 0, b1), (0, 3, b2), (2, 0, b3), (2, 1, b4)]
#
#         m1 = qumo.cbmat(blocks1)
#         m2 = qumo.cbmat(blocks2)
#         m3 = qumo.cbmat(blocks3)
#
#         np.testing.assert_allclose(
#             m1, [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
#         np.testing.assert_allclose(
#             m2, [[4, 4, 3, 3], [4, 4, 3, 3], [2, 2, 1, 1], [2, 2, 1, 1]])
#         np.testing.assert_allclose(
#             m3, [[1, 1, 0, 0, 0, 0, 2, 2],
#                  [1, 1, 0, 0, 0, 0, 2, 2],
#                  [0, 0, 0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0],
#                  [3, 3, 4, 4, 0, 0, 0, 0],
#                  [3, 3, 4, 4, 0, 0, 0, 0]])
#
#     def test_cbdiag_1x2(self):
#         b1 = [[1, 1]]
#         b2 = [[2, 2]]
#         b3 = [[3, 3]]
#
#         m1 = qumo.cbdiag(1, [(0, b1), (1, b2), (-1, b3)])
#         m2 = qumo.cbdiag(2, [(0, b1), (1, b2), (-1, b3)])
#         m3 = qumo.cbdiag(3, [(0, b1), (1, b2), (-1, b3)])
#         m4 = qumo.cbdiag(3, [(-2, b2), (1, b3), (2, b1)])
#
#         np.testing.assert_allclose(m1, [[1, 1]])
#         np.testing.assert_allclose(m2, [[1, 1, 2, 2], [3, 3, 1, 1]])
#         np.testing.assert_allclose(m3, [[1, 1, 2, 2, 0, 0],
#                                         [3, 3, 1, 1, 2, 2],
#                                         [0, 0, 3, 3, 1, 1]])
#         np.testing.assert_allclose(m4, [[0, 0, 3, 3, 1, 1],
#                                         [0, 0, 0, 0, 3, 3],
#                                         [2, 2, 0, 0, 0, 0]])
#
#     def test_cbdiag_2x2(self):
#         b1 = [[1, 1], [1, 1]]
#         b2 = [[2, 2], [2, 2]]
#         b3 = [[3, 3], [3, 3]]
#
#         m1 = qumo.cbdiag(1, [(0, b1), (1, b2), (-1, b3)])
#         m2 = qumo.cbdiag(2, [(0, b1), (1, b2), (-1, b3)])
#         m3 = qumo.cbdiag(3, [(0, b1), (1, b2), (-1, b3)])
#         m4 = qumo.cbdiag(3, [(0, b2), (1, b3), (2, b1)])
#
#         np.testing.assert_allclose(m1, [[1, 1], [1, 1]])
#         np.testing.assert_allclose(m2, [[1, 1, 2, 2], [1, 1, 2, 2],
#                                         [3, 3, 1, 1], [3, 3, 1, 1]])
#         np.testing.assert_allclose(m3, [[1, 1, 2, 2, 0, 0],
#                                         [1, 1, 2, 2, 0, 0],
#                                         [3, 3, 1, 1, 2, 2],
#                                         [3, 3, 1, 1, 2, 2],
#                                         [0, 0, 3, 3, 1, 1],
#                                         [0, 0, 3, 3, 1, 1]])
#         np.testing.assert_allclose(m4, [[2, 2, 3, 3, 1, 1],
#                                         [2, 2, 3, 3, 1, 1],
#                                         [0, 0, 2, 2, 3, 3],
#                                         [0, 0, 2, 2, 3, 3],
#                                         [0, 0, 0, 0, 2, 2],
#                                         [0, 0, 0, 0, 2, 2]])
