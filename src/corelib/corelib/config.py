"""
Generic functions and classes for processing of models configuration.

This module provides difference classes and functions for processing models
configuration. In particular, it provides means for variadic parameters
specification - `ValRange`, `ValArray`, `ValProd`, `ValEval`, `ValZip`
and `ValJoin`.

Author: Andrey Larionov <larioandr@gmail.com>
License: MIT
"""
from itertools import product
from typing import TypeVar, Any
from collections.abc import Iterable

_Num = TypeVar('_Num', int, float)


class ValRange(Iterable[_Num]):
    """
    Represents a range of parameter values from `left` to `right` with `step`.

    Class `ValRange` provides `values()` method that builds a list of all
    values. Besides that, it implements `__iter__` magic method.

    By default, step is equal to 1.

    In contrast to Python range(), right side is always included. Also, it
    doesn't support iteration with negative steps.

    Examples
    --------
    >>> ValRange(10, 20, 3).values()
    [10, 13, 16, 19, 20]
    >>> ValRange(10, 10).values()
    [10]
    >>> tuple(ValArray(10, 14))
    (10, 11, 12, 13, 14)
    """
    def __init__(self, left: _Num, right: _Num, step: _Num = 1):
        """
        Create `ValRange` instance.

        Parameters
        ----------
        left : float or int
        right : float or int
        step : float or int, optional (default: 1)
        """
        if left > right:
            raise ValueError(
                f"left bound is greater then right ({left} > {right})")
        if step <= 0:
            raise ValueError(f"illegal step ({step})")

        self.left: _Num = left
        self.right: _Num = right
        self.step: _Num = step

    def values(self) -> list[_Num]:
        """
        Convert range to a list of values.

        Returns
        -------
        list[float or int]
            a list of values from the range with left and right included.
        """
        curr_value = self.left
        next_value = self.left + self.step
        ret = [curr_value]
        while next_value <= self.right:
            curr_value = next_value
            ret.append(curr_value)
            next_value += self.step
        if curr_value != self.right:
            ret.append(self.right)
        return ret

    def __iter__(self):
        values = self.values()
        for i in values:
            yield i

    def __repr__(self) -> str:
        return f"<ValRange: left={self.left}, right={self.right}, " \
               f"step={self.step}>"


class ValArray(Iterable[_Num]):
    """
    Represents an array of floats or ints.

    Examples
    --------
    >>> ValArray([10, 20, 30]).values()
    [10, 20, 30]
    >>> ValArray(x**2 for x in range(1, 4)).values()
    [1, 4, 9]
    >>> for i in ValArray((34, 42)):
    >>>     print(i)
    34
    42
    """
    def __init__(self, data: Iterable[_Num] = ()):
        self._values: tuple[_Num] = tuple(data)

    def values(self) -> list[_Num]:
        """
        Returns a list with all array values.
        """
        return list(self._values)

    def __iter__(self):
        for i in self._values:
            yield i

    def __str__(self) -> str:
        size = len(self._values)
        if size > 5:
            s_values = (
                ", ".join(f"{val}" for val in self._values[:2]) +
                " ... " +
                f"{self._values[-1]}"
            )
        else:
            s_values = ", ".join(f"{val}" for val in self._values)
        return f"<ValArray: [{s_values}] (len: {len(self._values)})>"


class ValEval(Iterable):
    """
    Represents a set of records, where each key is defined with an iterable.

    Examples
    --------
    >>> ve = ValEval({"a": (10, 20), "b": ValArray([34, 42])})
    >>> for record in ve:
    >>>     print(record)
    {"a": 10, "b": 34}
    {"a": 10, "b": 42}
    {"a": 20, "b": 34}
    {"a": 20, "b": 42}
    """
    def __init__(self, data: dict[str, Iterable] = None):
        if data is None or len(data) == 0:
            self._data = {}
            self._all = []
            return

        if not isinstance(data, dict):
            raise TypeError("dict required")

        keys, values = [], []
        for key, value in data.items():
            keys.append(key)
            values.append(value)
        records = product(*values)
        self._data = dict(data)
        self._all = tuple(dict(zip(keys, record)) for record in records)

    def __iter__(self):
        for value in self.all():
            yield value

    def all(self) -> list[dict[str, Any]]:
        """
        Get a list of all records with atomic key values.
        """
        return list(self._all)

    def __repr__(self) -> str:
        args = " ".join(f'{key}={val}' for key, val in self._data.items())
        return f"<ValEval: {args}>"
