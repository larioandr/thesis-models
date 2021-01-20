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
    Abstract base class for the queues used in simulation models.

    Queues accept two methods:

    - `push(value: T) -> bool`
    - `pop() -> [T]`

    Push operation adds an item to the queue and returns true or false
    depending on whether the item was actually queued.

    Any queue will also implement four properties:

    - `capacity: int`
    - `size: int`
    - `empty: bool`
    - `full: bool`
    """
    @property
    def size(self) -> int:
        """
        Get the number of items in the queue.
        """
        raise NotImplementedError

    @property
    def capacity(self) -> int:
        """
        Get the maximum number of items in the queue.
        """
        raise NotImplementedError

    def push(self, item: T) -> bool:
        """
        Add an item to the queue.

        Parameters
        ----------
        item : T
            An item to add to the queue

        Returns
        -------
        success : bool
            True, if the item was really added.
        """
        raise NotImplementedError

    def pop(self) -> Optional[T]:
        """
        Extract an item from the queue.

        Returns
        -------
        item : T or None
            If queue failed to extract an item, it should return None
        """
        raise NotImplementedError

    def __len__(self):
        """
        Get the number of items in the queue (alias to size property).
        """
        return self.size

    @property
    def empty(self):
        """
        Check whether the queue is empty, i.e. number of items is zero.
        """
        return self.size == 0

    @property
    def full(self):
        """
        Check whether the queue is full, i.e. number of items equals capacity.
        """
        return self.size >= self.capacity


class FiniteFifoQueue(Queue[T]):
    """
    Finite queue representing a simple FIFO container.
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
        return self.__capacity

    @property
    def size(self) -> int:
        return self.__size

    def push(self, item: T) -> bool:
        if self.full:
            return False
        self.__items[self.__end] = item
        self.__end = (self.__end + 1) % self.capacity
        self.__size += 1
        return True

    def pop(self) -> Optional[T]:
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
        return f"(FiniteFifoQueue: q=[{', '.join(items_str)}], " \
               f"capacity={self.capacity}, size={self.size})"


class InfiniteFifoQueue(Queue[T]):
    """
    Infinite queue with FIFO order.
    """
    def __init__(self):
        self.__items = []

    @property
    def capacity(self):
        return np.inf

    @property
    def size(self):
        return len(self.__items)

    def push(self, item: T) -> bool:
        self.__items.append(item)
        return True

    def pop(self) -> Optional[T]:
        item: Optional[T] = None
        if len(self.__items) > 0:
            item = self.__items[0]
            self.__items = self.__items[1:]
        return item

    def __repr__(self):
        items = ', '.join([str(item) for item in self.__items])
        return f"(InfiniteFifoQueue: q=[{items}], size={self.size})"


class Server(Generic[T]):
    """
    Simple server model. Just stores a packet of type T and can be empty.
    """
    def __init__(self):
        self._packet: Optional[T] = None

    @property
    def ready(self) -> bool:
        return self._packet is None

    @property
    def busy(self) -> bool:
        return self._packet is not None

    @property
    def size(self) -> int:
        return 1 if self._packet is not None else 0

    def pop(self) -> T:
        if self._packet is None:
            raise RuntimeError("attempted to pop from an empty server")
        packet = self._packet
        self._packet = None
        return packet

    def serve(self, packet: T) -> None:
        if self._packet is not None:
            raise RuntimeError("attempted to put a packet to a busy server")
        self._packet = packet

    def __str__(self):
        suffix = "" if self._packet is None else f", packet={self._packet}"
        return f"(Server: busy={self.busy}{suffix})"
