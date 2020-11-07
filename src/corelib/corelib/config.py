"""
Generic functions and classes for processing of models configuration.

This module provides difference classes and functions for processing models
configuration. In particular, it provides means for variadic parameters
specification - `ValRange`, `ValArray`, `ValSetProd`, `ValSetEval`,
`ValSetZip` and `ValSetJoin`.

Author: Andrey Larionov <larioandr@gmail.com>
License: MIT
"""

# Disable pylint warnings about unsubscriptable types (E1136) - they don't
# work properly for Python 3.9 for now. Also disable module length check -
# comments are very large (C0302).
# pylint: disable=E1136,C0302

import functools
from functools import reduce
from itertools import product
from typing import TypeVar, Any, Optional, Mapping, Literal, Union
from collections.abc import Iterable

from marshmallow import Schema, fields, post_load, ValidationError, missing
from marshmallow.fields import Field

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
    >>> ValRange(10, 20, 3).values
    [10, 13, 16, 19, 20]
    >>> ValRange(10, 10).values
    [10]
    >>> tuple(ValArray((10, 14)))
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

    @functools.cached_property
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
        values = self.values
        for i in values:
            yield i

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        return self.left == other.left and \
               self.right == other.right and \
               self.step == other.step

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

    def __repr__(self):
        return f"ValRange{{left={self.left}, " \
               f"right={self.right}, step={self.step}}}"


DTYPE = Literal['int', 'float', 'str', 'matrix', 'vector', 'vec2d', 'vec3d']


class ValArray(Iterable[_T]):
    """
    Represents an array of any type (expected to be mono-typed).

    Examples
    --------
    >>> ValArray([10, 20, 30]).values
    [10, 20, 30]
    >>> # noinspection PyUnresolvedReferences
    >>> ValArray((x**2 for x in range(1, 4))).values
    [1, 4, 9]
    >>> for i in ValArray((34, 42)):
    >>>     print(i)
    34
    42
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, data: Iterable[_T] = (), dtype: Optional[DTYPE] = None):
        """
        Initialize a ValArray.

        The array is statically typed. That is, all elements must have the
        same type. Data type can be provided explicitly via `dtype` argument.
        Otherwise, it is determined by the first item in the array. If the
        array is empty and `dtype` is not provided, then by default 'float'
        is assumed.

        Possible data types are:

        - "int"
        - "float"
        - "str"

        Future:

        - "matrix" - numpy.ndarray with 2 dimensions or convertible
        - "vector" - numpy.ndarray with 1 dimension or convertible
        - "vec2d" - numpy.ndarray with 2 elements or convertible
        - "vec3d" - numpy.ndarray with 2 elements or convertible

        Parameters
        ----------
        data : Iterable, optional
            default is empty tuple
        dtype : _DTYPE, optional
            type of the elements, stored in the array.
        """
        values: tuple[_T] = tuple(data)
        if dtype is None:
            if not values:
                dtype = 'float'
            else:
                dtype = guess_dtype(values)
                if dtype is None:
                    raise TypeError("failed to determine data type")
            self._values = values
        elif dtype in ['int', 'float', 'str']:
            fun = {
                'int': int,
                'float': float,
                'str': str,
            }[dtype]
            try:
                self._values = tuple([fun(item) for item in values])
            except ValueError as ex:
                raise TypeError(f"wrong type, {dtype} expected") from ex
        else:
            self._values = values
        self._dtype = dtype

    @property
    def values(self) -> tuple[_T, ...]:
        """
        Returns a list with all array values.
        """
        return self._values

    @property
    def dtype(self):
        """Array data type."""
        return self._dtype

    def __iter__(self):
        for i in self._values:
            yield i

    def __len__(self):
        return len(self._values)

    def __eq__(self, other):
        return self.values == other.values and self.dtype == other.dtype

    def __repr__(self):
        values = [str(item) for item in self.values]
        values_str = ", ".join(values)
        return f"ValArray{{dtype={self._dtype}, values=[{values_str}]}}"


# Disable pylint warning about too few public methods - its a mixin:
# pylint: disable=R0903
class ValSetEqMixin:
    """
    A mixin that compares value sets by their data.

    Two value sets are considered equal, if they provide the same
    datasets ahdn have the same type.
    """
    # noinspection PyTypeChecker
    def __eq__(self, other: Iterable[_DST]):
        return isinstance(self, type(other)) and \
               isinstance(other, type(self)) and \
               list(self) == list(other)


class ValSetEval(Iterable[_DST], ValSetEqMixin):
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
    def data(self):
        """
        A dictionary that was used when creating the object, alias to `args`.
        """
        return self._data

    @property
    def args(self):
        """
        A dictionary that was used when creating the object, alias to `data`.
        """
        return self._data

    def __repr__(self):
        keys = list(self.data.keys())
        max_len = max([0] + [len(key) for key in keys])
        padding = ' '*4
        lines = [
            f'{padding}{key:<{max_len}}: {self._data[key]}'
            for key in keys
        ]
        lines_str = '\n'.join(lines)
        return f"ValSetEval:\n{lines_str}"


class ValSetJoin(Iterable[_DST], ValSetEqMixin):
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
        self._args = args
        self._data = tuple(items_list)
        self._unique = unique

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
    def args(self) -> tuple[Iterable[_DST], ...]:
        """Get the arguments those were passed when creating an object."""
        return self._args

    @property
    def unique(self) -> bool:
        """Flag indicating that all records are unique."""
        return self._unique

    def __repr__(self):
        args_lines = _pad_args_lines(self._args)
        flags_str = _str_flags(unique=self.unique, empty=(len(self) == 0))
        if args_lines:
            return f"ValSetJoin{flags_str}:\n{args_lines}"
        return f"ValSetJoin{flags_str}"


class ValSetProd(Iterable[_DST], ValSetEqMixin):
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
        self._unique = unique
        if unique:
            items_list = make_unique(items_list)

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
    def unique(self) -> bool:
        """Flag indicating that all records are unique."""
        return self._unique

    @property
    def args(self) -> tuple[Iterable[_DST], ...]:
        """Arguments passed when creating a ValSetProd instance."""
        return self._args

    def __repr__(self):
        args_lines = _pad_args_lines(self._args)
        flags_str = _str_flags(unique=self.unique, empty=(len(self) == 0))
        if args_lines:
            return f"ValSetProd{flags_str}:\n{args_lines}"
        return f"ValSetProd{flags_str}"


class ValSetZip(Iterable[dict[str, _T]], ValSetEqMixin):
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
        self._unique = unique
        if unique:
            items_list = make_unique(items_list)
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
    def unique(self) -> bool:
        """Flag indicating that all records are unique."""
        return self._unique

    def __repr__(self):
        args_lines = _pad_args_lines(self._args)
        flags_str = _str_flags(unique=self.unique, empty=(len(self) == 0))
        if args_lines:
            return f"ValSetZip{flags_str}:\n{args_lines}"
        return f"ValSetZip{flags_str}"


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


# noinspection PyUnresolvedReferences
def guess_dtype(values: Iterable[_T]) -> Optional[DTYPE]:
    """Determine the data type of the array."""
    if not values:
        return 'float'
    head = values[0]
    if isinstance(head, int):
        if all(isinstance(item, int) for item in values[1:]):
            return 'int'
    if all(isinstance(item, (int, float)) for item in values):
        return 'float'
    if all(isinstance(item, str) for item in values):
        return 'str'
    return None


def _pad_args_lines(args, padding=' '*4):
    lines = []
    padding = ' ' * 4
    for arg in args:
        arg_lines = str(arg).split('\n')
        arg_lines = [f"{padding}{line}" for line in arg_lines]
        lines.append("\n".join(arg_lines))
    return '\n'.join(lines)


def _str_flags(**kwargs):
    flags = []
    for key, value in kwargs.items():
        if value:
            flags.append(key)
    flags_str = ", ".join([f"{flag}=True" for flag in flags])
    if flags_str:
        flags_str = f"{{{flags_str}}}"
    return flags_str


#
# MARSHMALLOW SCHEMAS
# ===================

class ValRangeSchema(Schema):
    """Marshmallow schema for ValRange objects."""
    op = fields.String(default='range')
    left = fields.Float()
    right = fields.Float()
    step = fields.Float()

    # Disable pylint warnings about unused kwargs (W0613) and that this
    # method can be made a function since it can not be done (R0201)
    # pylint: disable=W0613
    # pylint: disable=R0201
    @post_load
    def create_object(self, data, **kwargs):
        """Create ValRange instance."""
        return ValRange(data['left'], data['right'], data['step'])


class TypedListField(Field):
    """
    This field type looks at 'dtype' field of the object to guess value type.
    """
    def _deserialize(self, value: Any, attr: Optional[str],
                     data: Optional[Mapping[str, Any]], **kwargs):
        """Guess data type by looking at the owner object dtype field."""
        dtype = data['dtype']
        schema = {
            "int": fields.Integer,
            "str": fields.String,
            "float": fields.Float,
        }[dtype]
        self.inner = schema()  # pylint: disable=W0201
        return super()._deserialize(value, attr, data, **kwargs)


class ValArraySchema(Schema):
    """Marshmallow schema for ValArray objects."""
    op = fields.String(default="array")
    dtype = fields.String(default="float")
    values = TypedListField()

    # Disable pylint warnings about unused kwargs (W0613) and that this
    # method can be made a function since it can not be done (R0201)
    # pylint: disable=W0613
    # pylint: disable=R0201
    @post_load
    def make_object(self, data, **kwargs):
        """Create ValArray instance."""
        return ValArray(data['values'], dtype=data['dtype'])


class OperationField(Field):
    """
    Marshmallow field that is serialized to JSON object with 'op' attr.
    See constructor documentation for more information.
    """
    def __init__(self, *args,
                 ops: Iterable[tuple[str, type, Union[type, str]]],
                 **kwargs):
        """
        Create an operation field.

        Field must receive a mapping of operation string code to the
        operation class and serialization schema, e.g.
        `ops = [('plus', Plus, PlusSchema), ('minus', Minus, MinusSchema)]`.

        Schema may be defined either by a class, or by this class name
        (last option is to face forward-declaration problem).

        Parameters
        ----------
        args : see `marshmallow.Field` for reference
        ops : Iterable[tuple[str, str, str]]
            an iterable of triplets (`operation code`, `class`, `schema`)
        kwargs :  see `marshmallow.Field` for reference
        """
        super().__init__(*args, **kwargs)
        self.ops = ops

    def _serialize(self, value: Any, attr: str, obj: Any, **kwargs):
        schema = None
        for _, type_, schema in self.ops:
            if isinstance(value, type_):
                if isinstance(schema, str):
                    schema = globals()[schema]
                return schema().dump(value)
        raise TypeError(f"unexpected value type {type(value)}")

    def _deserialize(self, value: Any, attr: Optional[str],
                     data: Optional[Mapping[str, Any]], **kwargs):
        """Determined type based on the 'op' field value."""
        for code_, _, schema in self.ops:
            if code_ == value['op']:
                if isinstance(schema, str):
                    schema = globals()[schema]
                return schema().load(value)
        raise ValidationError(f'wrong operand type "{value["op"]}"')


class ValSetEvalSchema(Schema):
    """
    Marshmallow schema for ValSetEval.

    Value set serialized JSON looks like this when `unique=False`:

    ```
    {'op': 'eval', 'args': {'key': {'op': 'array_or_range', ...}}}
    ```

    When `unique=True`, adds `'unique': true` to this JSON.

    All arguments in `'args'` value should be serialized `ValArray` or
    `ValRange`.
    """
    op = fields.String(default="eval")
    args = fields.Dict(fields.String(), OperationField(ops=(
        ("array", ValArray, ValArraySchema),
        ("range", ValRange, ValRangeSchema),
    )))

    # Disable pylint warnings about unused kwargs (W0613) and that this
    # method can be made a function since it can not be done (R0201)
    # pylint: disable=W0613
    # pylint: disable=R0201
    @post_load
    def make_object(self, data, **kwargs):
        """Create ValSetEval instance."""
        return ValSetEval(data['args'])


class UniqueField(Field):
    """
    If unique is true, it should add "'unique': true", otherwise - skip.
    """
    def _serialize(self, value: Any, attr: str, obj: Any, **kwargs):
        if value:
            return super()._serialize(value, attr, obj, **kwargs)
        return missing


class ValSetSchemaBase(Schema):
    """
    Base class for value sets operations ValSetProd, ValSetZip, ValSetJoin.

    For non-unique objects, it serializes an operation to JSON like:

    ```
    {
        'op': OPCODE,
        'args': [
            # list of ValSet serialized objects
        ]
    }
    ```

    If `unique=True`, then adds `'unique': true` to this JSON.

    DO define a concrete schema, subclass should specify `opcode` and
    `make_class` fields:

    - `opcode`: OPCODE value that will be put into JSON 'op' attribute.
    - `make_class`: to this class JSON will be deserialized.
    """
    opcode: str = ''
    make_class: Optional[type] = None

    op = fields.String()
    args = fields.List(OperationField(ops=(
        ("eval", ValSetEval, ValSetEvalSchema),
        ('prod', ValSetProd, 'ValSetProdSchema'),
        ('join', ValSetJoin, 'ValSetJoinSchema'),
        ('zip', ValSetZip, 'ValSetZipSchema'),
    )))
    unique = UniqueField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['op'].default = self.opcode

    # Disable pylint warnings about unused kwargs (W0613) and not callable
    # make_class field (E1102)
    # pylint: disable=W0613
    # pylint: disable=E1102
    @post_load
    def make_object(self, data, **kwargs):
        """Create value set instance."""
        return self.make_class(
            *data['args'],
            unique=data.get('unique', False)
        )


class ValSetJoinSchema(ValSetSchemaBase):
    """
    Marshmallow schema for ValSetProd class.

    Format
    ------
    - if unique=False: `{'op': 'join', 'args': [...]}`
    - if unique=True: `{'op': 'join', 'unique': True, 'args': [...]}`
    """
    opcode = 'join'
    make_class = ValSetJoin


class ValSetProdSchema(ValSetSchemaBase):
    """
    Marshmallow schema for ValSetProd class.

    Format
    ------
    - if unique=False: `{'op': 'prod', 'args': [...]}`
    - if unique=True: `{'op': 'prod', 'unique': True, 'args': [...]}`
    """
    opcode = 'prod'
    make_class = ValSetProd


class ValSetZipSchema(ValSetSchemaBase):
    """
    Marshmallow schema for ValSetZip class.

    Format
    ------
    - if unique=False: `{'op': 'zip', 'args': [...]}`
    - if unique=True: `{'op': 'zip', 'unique': True, 'args': [...]}`
    """
    opcode = 'zip'
    make_class = ValSetZip
