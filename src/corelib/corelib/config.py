"""
Generic functions and classes for processing of models configuration.

This module provides difference classes and functions for processing models
configuration. In particular, it provides means for variadic parameters
specification - `ValRange`, `ValArray`, `ValSetProd`, `ValSetEval`,
`ValSetZip` and `ValSetJoin`.

Author: Andrey Larionov <larioandr@gmail.com>
License: MIT
"""
import functools
import itertools
from functools import reduce
from itertools import product
from typing import TypeVar, Any
from collections.abc import Iterable

_Num = TypeVar('_Num', int, float)
_T = TypeVar('_T')
_DST = dict[str, _T]


class ValRange(Iterable[_Num]):
    """
    Represents a range of parameter values from `left` to `right` with `step`.

    Class `ValRange` provides `values()` method that builds a tuple of all
    values. Besides that, it implements `__iter__` and `__len__` magic methods.

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

    __next_id = itertools.count(1)

    def __init__(self, left: _Num, right: _Num, step: _Num = 1):
        """
        Create `ValRange` instance.

        Parameters
        ----------
        left : float or int
        right : float or int
        step : float or int, optional (default: 1)

        Raises
        ------
        ValueError
            raised when left > right or step <= 0
        """
        if left > right:
            raise ValueError(
                f"left bound is greater then right ({left} > {right})")
        if step <= 0:
            raise ValueError(f"illegal step ({step})")

        self._left: _Num = left
        self._right: _Num = right
        self._step: _Num = step
        self.__id = next(ValRange.__next_id)

    @functools.cache
    def values(self) -> tuple[_Num, ...]:
        """
        Convert range to a list of values.

        Returns
        -------
        tuple[float or int]
            a tuple of values from the range with left and right included.
        """
        curr_value = self.left
        next_value = self.left + self.step
        ret: list[_Num] = [curr_value]
        while next_value <= self.right:
            curr_value = next_value
            ret.append(curr_value)
            next_value += self.step
        if curr_value != self.right:
            ret.append(self.right)
        return tuple(ret)

    def __iter__(self):
        values = self.values()
        for i in values:
            yield i

    def __len__(self):
        return len(self.values())

    @property
    def id(self) -> int:  # pylint: disable=C0103
        """Unique ID of this ValRange instance."""
        return self.__id

    @property
    def left(self) -> _Num:
        """Left boundary."""
        return self._left

    @property
    def right(self) -> _Num:
        """Right boundary."""
        return self._right

    @property
    def step(self) -> _Num:
        """Grid step value."""
        return self._step

    def __repr__(self) -> str:
        return f"<ValRange #{self.id}: left={self.left}, right={self.right}, " \
               f"step={self.step}>"


class ValArray(Iterable[_T]):
    """
    Represents an array of any type (expected to be mono-typed).

    Examples
    --------
    >>> ValArray([10, 20, 30]).values()
    [10, 20, 30]
    >>> # noinspection PyUnresolvedReferences
    >>> ValArray((x**2 for x in range(1, 4))).values()
    [1, 4, 9]
    >>> for i in ValArray((34, 42)):
    >>>     print(i)
    34
    42
    """

    __next_id = itertools.count(1)

    def __init__(self, data: Iterable[_T] = ()):
        """
        Initialize a ValArray.

        Parameters
        ----------
        data : Iterable, optional
            default is empty tuple
        """
        self._values: tuple[_T] = tuple(data)
        self.__id = next(ValArray.__next_id)

    def values(self) -> tuple[_T, ...]:
        """
        Returns a list with all array values.
        """
        return self._values

    def __iter__(self):
        for i in self._values:
            yield i

    def __len__(self):
        return len(self._values)

    @property
    def id(self):  # pylint: disable=C0103
        """Unique ID of this ValArray instance."""
        return self.__id

    def __repr__(self):
        size: int = len(self._values)
        if size > 5:
            s_values = (
                ", ".join(f"{val}" for val in self._values[:2]) +
                " ... " +
                f"{self._values[-1]}"
            )
        else:
            s_values = ", ".join(f"{val}" for val in self._values)
        return f"<ValArray #{self.id}: [{s_values}] size={len(self)}>"


class ValSetEval(Iterable[_DST]):
    """
    Represents a set of records, where each key is defined with an iterable.

    A result of evaluation is a list of dictionaries, where each key has
    a single value. The list is built as a cortesian product of all keys
    values, see examples below.

    For instance, given a dictionary {"A": (1, 2, 3), "B": ("hello", "bye")},
    ValSetEval will create 6 records:

    - {"A": 1, "B": "hello"}
    - {"A": 1: "B": "bye"}
    - {"A": 2, "B": "hello"}
    - {"A": 2: "B": "bye"}
    - {"A": 3, "B": "hello"}
    - {"A": 3: "B": "bye"}

    As a special case, if a dictionary doesn't have values at all, the
    result will have a single empty dictionary.

    When created without arguments, will result in an empty set of values.

    ValSetEval implements several methods:

    - all(): return a list of all values (dictionaries with a single value
             assigned for each key.
    - keys(): returns a set of keys from which the ValSetEval was built
    - values(): returns a set of values
    - items(): returns an iterable to iterate over key-value pairs
    - __iter__(): iterate over evaluated records from all()
    - __len__(): return the number of evaluated records in all().

    Examples
    --------
    >>> ve = ValSetEval({"a": (10, 20), "b": ValArray([34, 42])})
    >>> for record in ve:
    >>>     print(record)
    {"a": 10, "b": 34}
    {"a": 10, "b": 42}
    {"a": 20, "b": 34}
    {"a": 20, "b": 42}

    >>> ValSetEval({}).all()
    ({},)

    >>> ValSetEval().all()
    ()
    """

    __next_id = itertools.count(1)

    def __init__(self, data: dict[str, Iterable] = None):
        """
        Initialize a ValSetEval instance.

        Parameters
        ----------
        data : dict with iterable values, optional
            default is None

        Raises
        ------
        TypeError
            raised when data is not a dictionary
        """
        self.__id = next(self.__next_id)

        if data is None:
            self._data: dict[str, _T] = {}
            self._all: tuple[dict[str, _T], ...] = ()
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

    def all(self) -> tuple[_DST, ...]:
        """
        Get a list of all records with atomic key values.
        """
        return self._all

    def __iter__(self):
        for value in self._all:
            yield value

    def __len__(self):
        return len(self._all)

    @property
    def id(self) -> int:  # pylint: disable=C0103
        """Unique ID of this ValSetEval instance."""
        return self.__id

    def keys(self):
        """Set of keys defined in this values set."""
        return self._data.keys()

    def values(self):
        """Set of values defined in this values set."""
        return self._data.values()

    def items(self):
        """Iterator over key-value pairs."""
        return self._data.items()

    def __repr__(self) -> str:
        return f"<ValSetEval #{self.id}: size={len(self)}>"


class ValSetJoin(Iterable[_DST]):
    """
    ValSetJoin concatenates sets of values.

    For example, if two sets are:

    S1 = ({"a": 10, "b": 34}, {"a": 11, "b": 42}, {"a": 100})
    S2 = ({"x": "hello"}, {"x": "bye"}, {"a": 100})

    the join result ValSetJoin(S1, S2) will contain 6 values:

    - {"a": 10, "b": 34}
    - {"a": 11, "b": 42}
    - {"a": 100}
    - {"x": "hello"}
    - {"x": "bye"}
    - {"a": 100}

    To get rid of duplicates, ValSetJoin accepts `unique` flag. If an object
    is created as `ValSetJoin(S1, S2, unique=True)`, then the result
    will contain only five records:

    - {"a": 10, "b": 34}
    - {"a": 11, "b": 42}
    - {"a": 100}
    - {"x": "hello"}
    - {"x": "bye"}

    Each `ValSetJoin` instance receives a unique ID that is incremented
    each time a new object is created.

    The following methods were implemented in `ValSetJoin`:

    - `all()`: get a tuple of all values
    - `__iter__()`: iterate of values
    - `__len__()`: get the number of values

    There are also two properties:

    - `id`: `ValSetJoin` instance ID
    - `args`: a tuple of sets those were used to create a `ValSetJoin`

    Examples
    --------
    >>> ValSetJoin().all()
    ()

    >>> ValSetJoin([{}]).all()
    ({},)

    >>> ValSetJoin([{"a": 10}, {"b": 20}], [{"b": 42, "x": "hello"}]).all()
    ({"a": 10}, {"b": 20}, {"b": 42, "x": "hello"})
    """
    __next_id = itertools.count(1)

    def __init__(self, *args: Iterable[_DST], unique: bool = False):
        """
        Initialize a ValSetJoin instance.

        Parameters
        ----------
        args : tuple of Iterable[dict]
            iterables with dictionary values
        unique : bool, optional
            flag indicating whether to remove duplicated dicts (default: False)

        Raises
        ------
        TypeError
            raised when any set contain any value except a dict
        """
        self._args = args
        items_list = []
        for arg in args:
            for item in arg:
                if not isinstance(item, dict):
                    raise TypeError(f"expected dict, {type(item)} found")
                items_list.append(item)
        if unique:
            items_list = make_unique(items_list)
        self.__id = next(ValSetJoin.__next_id)
        self._args = args
        self._data = tuple(items_list)

    def all(self) -> tuple[dict[str, Any], ...]:
        """
        Get a tuple of all values.

        Returns
        -------
        tuple[dict]
        """
        return self._data

    def __iter__(self):
        for item in self._data:
            yield item

    def __len__(self):
        return len(self._data)

    @property
    def id(self) -> int:  # pylint: disable=C0103
        """Unique ID of the ValSetJoin."""
        return self.__id

    @property
    def args(self) -> tuple[Iterable[_DST], ...]:
        """Get the arguments those were passed when creating an object."""
        return self._args

    def __repr__(self):
        return f"<ValSetJoin #{self.__id}: size={len(self)}>"


class ValSetProd(Iterable[_DST]):
    """
    Build a cartesian product of sets of dict values.

    If given sets of dict-values S1, S2, .., Sn, `ValSetProd` will build a
    cartesian set S1 x S2 x ... x Sn. For example:

    S1 = [{"a": 10}, {"a": 20}]
    S2 = [{"b": 100}]
    S3 = [{"x": "hello"}, {"x": "bye}]

    the result will contain four records:

    - {"a": 10, "b": 100, "x": "hello"}
    - {"a": 10, "b": 100, "x": "bye"}
    - {"a": 20, "b": 100, "x": "hello"}
    - {"a": 20, "b": 100, "x": "bye"}

    Empty sets are treated as zeros in multiplication. That is, if any of
    the sets is empty, then the result will also be empty. However, empty
    dictionaries are kept, see examples below.

    It is important, that dictionaries in distinct sets **MUST** have
    non-intersecting sets of keys, otherwise it will lead to an error.
    Rationale behind this is that dictionaries from these sets are merged, so
    if a key will appear in more than one dictionary, this may lead to
    hardly detectable problems in runtime.

    Any iterable object with `dict` values can be used as an argument.
    For instance, `ValSetEval`, `ValSetProd`, `ValSetZip` or `ValSetJoin`
    satisfy this.

    To get rid of duplicated entries, pass `unique=True` to the constructor.

    Each `ValSetProd` instance receives a unique ID that is incremented each
    time a new object is created.

    `ValSetProd` implements the following methods and properties:

    - `all()`: returns a tuple of all records.
    - `__iter__()`: iterate over values.
    - `__len__()`: get the number of records.
    - `id`: `ValSetProd` instance ID.
    - `args`: a tuple of arguments those were used to create the instance.

    Examples
    --------
    >>> ValSetProd().all()
    ()
    >>> ValSetProd([{}]).all()
    ({},)
    >>> ValSetProd([{"a": 1}, {"a": 2}], [{"b": 10}, {"b": 20}])
    ({"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20})
    """

    __next_id = itertools.count(1)

    def __init__(self, *args: Iterable[_DST], unique: bool = False):
        """
        Initialize a ValSetProd instance.

        Parameters
        ----------
        args : tuple of Iterable[dict]
            iterables with dictionary values
        unique : bool, optional
            flag indicating whether to remove duplicated dicts (default: False)

        Raises
        ------
        KeyError
            raised when sets A1 and A2 from args contains dicts D1 and D2 with
            intersecting key sets
        TypeError
            raised when any set contain any value except a dict
        """
        self._args = args

        # Check that key sets in different args are non-intersecting:
        all_keys = set()
        for arg in args:
            curr_keys = set()
            for dict_ in arg:
                try:
                    for key in dict_.keys():
                        if key in all_keys:
                            raise KeyError(f"key '{key}' in multiple sets")
                        curr_keys.add(key)
                except AttributeError as ex:
                    raise TypeError(
                        f"expected dictionary, but {type(dict_)} found"
                    ) from ex
            for key in curr_keys:
                all_keys.add(key)

        if args:
            items_list = [
                reduce(lambda d1, d2: d1 | d2, item) for item in product(*args)
            ]
        else:
            items_list = []

        # If unique=True, then remove duplicates:
        if unique:
            items_list = make_unique(items_list)

        self.__id = next(ValSetProd.__next_id)
        self._data = tuple(items_list)

    def all(self) -> tuple[_DST, ...]:
        """Get a tuple of all records."""
        return self._data

    def __iter__(self):
        for item in self._data:
            yield item

    def __len__(self):
        return len(self._data)

    @property
    def id(self) -> int:  # pylint: disable=C0103
        """Unique ID of the ValSetProd instance."""
        return self.__id

    @property
    def args(self) -> tuple[Iterable[_DST], ...]:
        """Arguments passed when creating a ValSetProd instance."""
        return self._args

    def __repr__(self):
        return f"<ValSetJoin #{self.id}: size={len(self)}>"


class ValSetZip(Iterable[dict[str, _T]]):
    """
    Build a ZIP from sets of dict values.

    This operation joins dictionaries in Python `zip` way. However, in
    contrast to Python `zip`, all sets **MUST** have the same size, or an
    error will be raised.

    For example:

    S1 = [{"a": 10}, {"a": 20}]
    S2 = [{"b": "hello"}, {"b": "bye"}]

    the result will contain two records:

    - {"a": 10, "b": "hello"}
    - {"a": 20, "b": "bye"}

    It is important, that dictionaries being joined **MUST** have
    non-intersecting sets of keys, otherwise it will lead to an error.
    Rationale behind this is that dictionaries from these sets are merged, so
    if a key will appear in more than one dictionary, this may lead to
    hardly detectable problems in runtime.

    However, in contrast to `ValSetProd`, this requirement applies only to
    dictionaries those will be joined, not the the whole sets.
    For demonstration, consider sets S1, S2, S3:

    - S1 = [{"a": 1, "u": "one"}, {"a": 2}]
    - S2 = [{"x": 34}, {"x": 42, "u": "two"}]
    - S3 = [{"u": "what?"}, {"x": 13}]

    We can create `ValSetProd(S1, S2)` since key sets of dictionaries being
    actually joined are not intersecting. The result will be:

    [{"a": 1, "u": "one", "x": 34},
     {"a": 2, "u": "two", "x": 42}]

    However, we can not create `ValSetProd(S1, S3)`, since then we will
    have to merge dicts {"a": 1, "u": "one"} and {"u": "what?"}, which will
    lead to overwriting of "u" value and the result will depend on the order
    of S1 and S3 arguments. For the same reason (but related to "x" key) we
    can not create `ValSetProd(S2, S3)`.

    Any iterable object with `dict` values can be used as an argument.
    For instance, `ValSetEval`, `ValSetProd`, `ValSetZip` or `ValSetJoin`
    satisfy this.

    To get rid of duplicated entries, pass `unique=True` to the constructor.

    Each `ValSetZip` instance receives a unique ID that is incremented each
    time a new object is created.

    `ValSetZip` implements the following methods and properties:

    - `all()`: returns a tuple of all records.
    - `__iter__()`: iterate over values.
    - `__len__()`: get the number of records.
    - `id`: `ValSetProd` instance ID.
    - `args`: a tuple of arguments those were used to create the instance.

    Examples
    --------
    >>> ValSetZip().all()
    ()
    >>> ValSetZip([{}], [{}]).all()
    ({},)
    >>> ValSetZip([{"a": 1}, {"a": 2}], [{"b": 100}, {"b": 200}])
    ({"a": 1, "b": 100}, {"a": 2, "b": 200})
    """

    __next_id = itertools.count(1)

    def __init__(self, *args: Iterable[_DST], unique: bool = False):
        """
        Initialize a ValSetZip instance.

        Parameters
        ----------
        args : tuple of Iterable[dict]
            iterables with dictionary values
        unique : bool, optional
            flag indicating whether to remove duplicated dicts (default: False)

        Raises
        ------
        KeyError
            raised when dictionaries A1[i] and A2[i] from args are dicts with
            intersecting key sets
        TypeError
            raised when any set contain any value except a dict
        """
        self._args = args
        if args:
            # noinspection PyTypeChecker
            size = len(args[0])
            for arg in args:
                # noinspection PyTypeChecker
                if len(arg) != size:
                    raise IndexError("wrong dimension, all args must have"
                                     "the same number of elements")
                for dict_ in arg:
                    if not isinstance(dict_, dict):
                        raise TypeError(f"expected dictionary, "
                                        f"but {type(dict_)} found")
            # Check that key sets of dictionaries to be zip'd not intersect
            for row in zip(*args):
                keys = set()
                for dict_ in row:
                    for key in dict_:
                        if key in keys:
                            raise KeyError("intersecting key sets")
                        keys.add(key)
        # Build items list
        items_list: list[dict[str, _T], ...] = [
            reduce(lambda u, v: u | v, item) for item in zip(*args)]
        if unique:
            items_list = make_unique(items_list)
        self.__id: int = next(ValSetZip.__next_id)
        self._data = tuple(items_list)

    def all(self) -> tuple[_DST, ...]:
        """Get a tuple of all values."""
        return self._data

    def __iter__(self):
        for item in self._data:
            yield item

    def __len__(self):
        return len(self._data)

    @property
    def args(self) -> tuple[Iterable[_DST], ...]:
        """Args with which the instance was created."""
        return self._args

    @property
    def id(self) -> int:  # pylint: disable=C0103
        """Unique ID of ValSetZip instance."""
        return self.__id

    def __repr__(self):
        return f"<ValSetZip #{self.id}: size={len(self)}>"


def make_unique(dicts: Iterable[dict]) -> list[dict]:
    """
    Remove duplicates from a sequence of dictionaries.

    Parameters
    ----------
    dicts : Iterable[dict]
        a sequence of dictionaries, possibly - with duplicated values

    Returns
    -------
    list[dict]
        list of unique dictionaries.

    """
    # Since dicts are not hashable, we can not convert them to set.
    # TODO: think about more efficient solution
    new_dicts: list[dict] = []
    for dict_ in dicts:
        found: bool = False
        for ex_dict_ in new_dicts:
            if ex_dict_ == dict_:
                found = True
                break
        if not found:
            new_dicts.append(dict_)
    return new_dicts
