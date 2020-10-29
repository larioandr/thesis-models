import pytest

from corelib.config import ValRange, ValArray, ValEval


#
# TESTING ValRange
# ----------------
def test_val_range__values__multiple_ints():
    """
    Test that ValRange.values() returns a list with values.

    GIVEN:
    - a ValRange instance with left < right, such that right - left > step

    VALIDATE:
    - values() call returns [left, left + step , left + 2*step, ..., right]
    - make sure that left and right are both included
    """
    range_ = ValRange(left=10, right=47, step=10)
    assert range_.values() == [10, 20, 30, 40, 47]


def test_val_range_values__single_int():
    """
    Test that ValRange.values() returns a single value when left == right.

    GIVEN:
    - a ValRange instance with left == right

    VALIDATE:
    - values() call returns [left,] tuple
    """
    range_ = ValRange(left=42, right=42, step=1)
    assert range_.values() == [42]


def test_val_range__raises_error_when_left_is_greater_than_right():
    """
    Test that ValRange constructor raises ValueError when left > right.
    """
    with pytest.raises(ValueError):
        ValRange(left=50, right=40, step=-5)


def test_val_range__raises_error_for_zero_step():
    """
    Test that ValRange constructor raises ValueError when step = 0
    """
    with pytest.raises(ValueError):
        ValRange(10, 20, 0)


def test_val_range__raises_error_for_negative_step():
    """
    Test that ValRange constructor raises ValueError when step < 0
    """
    with pytest.raises(ValueError):
        ValRange(10, 20, -1)


def test_val_range__default_step_is_1():
    """
    Test that ValRange created without explicit step defaults to step = 1.
    """
    range_ = ValRange(5, 8)
    assert range_.values() == [5, 6, 7, 8]


def test_val_range__implements_repr():
    """
    Test stringification of ValRange.
    """
    range_ = ValRange(left=23, right=49, step=8)
    assert str(range_) == "<ValRange: left=23, right=49, step=8>"


def test_val_range__implements_iter_magic():
    """
    Test that array implements __iter__ magic method.
    """
    range_ = ValRange(left=23, right=49, step=8)
    assert tuple(range_) == (23, 31, 39, 47, 49)


#
# TESTING ValArray
# ----------------
def test_val_array__accepts_any_iterable():
    """
    Test that ValArray can be constructed from list, tuple or iterator.
    """
    assert ValArray([]).values() == []
    assert ValArray([10, 20, 30]).values() == [10, 20, 30]
    assert ValArray((10, 20, 30)).values() == [10, 20, 30]
    assert ValArray(x**2 for x in range(5)).values() == [0, 1, 4, 9, 16]


def test_val_array__copies_list():
    """
    Test that ValArray copes list and values() returns another list instance.
    """
    values = [1, 2, 3]
    assert ValArray(values).values() is not values


def test_val_array__default_is_empty():
    """
    Test that ValArray created without arguments returns an empty list.
    """
    assert ValArray().values() == []


def test_val_array__implements_iter_magic():
    """
    Test that ValArray implements __iter__ magic method.
    """
    assert tuple(ValArray([1, 2, 9])) == (1, 2, 9)


def test_val_array__implements_repr_magic():
    """
    Test that ValArray implements __repr__ magic method.
    """
    array = ValArray([34, 42])
    assert str(array) == "<ValArray: [34, 42] (len: 2)>"

    array = ValArray(range(1, 101))
    assert str(array) == "<ValArray: [1, 2 ... 100] (len: 100)>"


#
# TESTING ValEval
# ---------------
def test_val_eval__builds_values_product():
    """
    Test that ValEval builds a product of all values.
    """
    ve = ValEval({
        "speed": ValRange(30, 50, 10),
        "m": ValArray([4, 8]),
        "tari": ValArray(["6.25us"])
    })
    assert ve.all() == [
        {"speed": 30, "m": 4, "tari": "6.25us"},
        {"speed": 30, "m": 8, "tari": "6.25us"},
        {"speed": 40, "m": 4, "tari": "6.25us"},
        {"speed": 40, "m": 8, "tari": "6.25us"},
        {"speed": 50, "m": 4, "tari": "6.25us"},
        {"speed": 50, "m": 8, "tari": "6.25us"},
    ]


def test_val_eval__accepts_any_iterable():
    """
    Test that generic iterable can be passed as keys values.
    """
    ve = ValEval({"a": (10, 20), "b": (x**2 for x in range(1, 4))})
    assert ve.all() == [
        {"a": 10, "b": 1},
        {"a": 10, "b": 4},
        {"a": 10, "b": 9},
        {"a": 20, "b": 1},
        {"a": 20, "b": 4},
        {"a": 20, "b": 9},
    ]


def test_val_eval__default_is_empty():
    """
    Test that by default ValEval defines an empty set of parameters.
    """
    assert ValEval().all() == []


def test_val_eval__implements_iter_magic():
    """
    Test that ValEval can be used as an iterator.
    """
    ve = ValEval({"a": (34, 42)})
    assert tuple(ve) == ({"a": 34}, {"a": 42})


def test_val_eval__implements_repr_magic():
    """
    Test that ValEval can be casted to string.
    """
    ve = ValEval({"a": (34, 42), "b": ValRange(10, 20, 2)})
    assert str(ve) == \
           "<ValEval: a=(34, 42) b=<ValRange: left=10, right=20, step=2>>"


def test_val_eval__raise_error_when_key_value_not_iterable():
    """
    Test that ValEval constructor raises TypeError when dict contains
    non-iterable values.
    """
    with pytest.raises(TypeError):
        ValEval({"a": (10, 20), "b": 5})


# noinspection PyArgumentList,PyTypeChecker
def test_val_eval__raise_error_for_non_dict():
    """
    Test that passing a non-dictionary argument raises error.
    """
    with pytest.raises(TypeError):
        ValEval(a=(10, 20))
    with pytest.raises(TypeError):
        ValEval([{"a": (10, 20)}])


def test_val_eval__empty_when_some_key_empty():
    """
    Test that ValEval produces an empty set if some value is empty.
    """
    assert ValEval({"a": (10, 20), "b": ()}).all() == []
