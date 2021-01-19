from collections import namedtuple
from typing import Optional, Sequence, TypeVar, Generic, List
import numpy as np

from pyqumo.random import CountableDistribution


TimeValue = namedtuple('TimeValue', ['time', 'value'])
Statistics = namedtuple('Statistics', ['avg', 'var', 'std', 'count'])


def build_distribution_from_time_values(
        time_values: Sequence[TimeValue],
        end_time: Optional[float] = None
) -> CountableDistribution:
    """
    Build a choice distribution from a sequence of time-value pairs.

    Parameters
    ----------
    time_values : sequence of TimeValue pairs.
    end_time : float, optional
        If provided, specifies time when the processes ended. The last value
        from time_values sequence will be assumed to be kept till this time.
        If omitted, time from the last TimeValue pair is assumed to be used.

    Returns
    -------
    distribution : CountableDistribution

    Raises
    ------
    IndexError
        raised when time_values sequence is empty, or when
    ValueError
        raised if time in time_values sequence is not ordered ascending.
    """
    if not time_values:
        raise IndexError('empty time_values sequence is not allowed')

    if end_time is None:
        end_time = time_values[-1].time

    values = np.asarray([tv.value for tv in time_values])

    timestamps = np.asarray([tv.time for tv in time_values] + [end_time])
    intervals = timestamps[1:] - timestamps[:-1]
    if (intervals < -1e-12).any():
        raise ValueError("negative intervals disallowed")

    max_value = values.max(initial=0)
    pmf = np.zeros(max_value + 1)
    for value, interval in zip(values, intervals):
        value = int(value)
        pmf[value] += interval

    duration = timestamps[-1] - timestamps[0]
    pmf = pmf / duration

    return CountableDistribution(
        lambda x: pmf[x] if 0 <= x <= max_value else 0.0,
        precision=1e-12)


def build_statistics(intervals: Sequence[float]) -> Statistics:
    """
    Build Statistics object from the given sequence of intervals.

    Parameters
    ----------
    intervals : 1D array_like

    Returns
    -------
    statistics : Statistics
    """
    avg = np.mean(intervals)
    var = np.var(intervals, ddof=1)  # unbiased estimate
    std = var**0.5
    return Statistics(avg=avg, var=var, std=std, count=len(intervals))


T = TypeVar('T')


class Queue(Generic[T]):
    """
    Queue represents a simple FIFO container. Inside it is a list.

    Queue accepts two methods:

    - `push(value: T) -> bool`
    - `pop() -> [T]`

    If queue is full, then `push()` returns `False` and queue is not updated.
    If queue is empty, then `pop()` returns `None`. No exceptions are raised
    in either case.

    Queue also has several properties to study its size:

    - `capacity: int`
    - `size: int`
    - `empty: bool`
    - `full: bool`
    """

    def __init__(self, capacity: int):
        """
        Create a queue.

        Parameters
        ----------
        capacity : int
            Specifies maximum queue size
        """
        self.__items: List[T] = [None] * capacity
        self.__capacity = capacity
        self.__size = 0
        self.__head = 0
        self.__end = 0

    @property
    def capacity(self) -> int:
        """
        Get queue capacity.
        """
        return self.__capacity

    def __len__(self) -> int:
        """
        Get the number of items in the queue (alias to size property)
        """
        return self.__size

    @property
    def size(self) -> int:
        """
        Get the number of items in the queue (alias to len(queue))
        """
        return self.__size

    @property
    def empty(self) -> bool:
        """
        Returns `True` if the queue has no items.
        """
        return self.__size == 0

    @property
    def full(self) -> bool:
        """
        Returns `True` if the queue size equals to its capacity.
        """
        return self.__size >= self.__capacity

    def push(self, item: T) -> bool:
        """
        Add an item to the queue, if it is not full.

        Parameters
        ----------
        item : T
            an item to add to the queue

        Returns
        -------
        success : bool
            True iff the queue wasn't full and the item was added.
            Otherwise returns False.
        """
        if self.full:
            return False
        self.__items[self.__end] = item
        self.__end = (self.__end + 1) % self.capacity
        self.__size += 1
        return True

    def pop(self) -> Optional[T]:
        """
        Get the first item from the queue. If empty, returns `None`.

        Returns
        -------
        item : T, optional
            The first item, if wasn't empty. Otherwise `None`.
        """
        if self.empty:
            return None
        item = self.__items[self.__head]
        self.__items[self.__head] = None
        self.__head = (self.__head + 1) % self.capacity
        self.__size -= 1
        return item

    def __repr__(self) -> str:
        """
        Get string representation of the queue.
        """
        if self.__head < self.__end:
            items = self.__items[self.__head:self.__end]
        elif self.__head >= self.__end and self.__size > 0:
            items = self.__items[self.__head:] + self.__items[:self.__end]
        else:
            items = []
        items_str = [str(item) for item in items]
        return f"(Queue: q=[{', '.join(items_str)}], capacity={self.capacity})"

