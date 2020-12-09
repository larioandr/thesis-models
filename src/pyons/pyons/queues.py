import heapq
import itertools


class Queue(object):
    """
    Priority queue base abstract class.

    Items are expected to be stored in tuples like ``(time, index, event)``, where ``time`` is current time,
    ``index`` is the event index and ``event`` is the event itself. While it is not mandatory to represents
    records as such tuples, each record must contain these three fields.

    The queue allows to:

    - push and pop events

    - get the timestamp of the last event (the current time)

    - remove events by their indexes, by the predicate or by the comparator

    The following protected abstract methods must be implemented (see their docs for reference):

    - ``_enqueue()``

    - ``_dequeue()``

    - ``_get_queue_length()``

    - ``_get_as_list()``

    - ``_remove_by_index()``

    - ``_remove_by_event()``

    - ``_remove_by_predicate()``

    - ``_remove_all()``
    """

    def __init__(self, time_getter=lambda item: item.time,
                 index_getter=lambda item: item.index):
        """
        Args:
            time_getter: a function returning time from the event,
                by default: ``(item) -> item.time``

            index_getter: a function returning index from the event,
                by default: ``(item) -> item.index``
        """
        self.time_getter = time_getter
        self.index_getter = index_getter

    def push(self, item):
        """
        Add an item into the queue.

        Args:
            item: what to push

        Returns:
            an index of the added item
        """
        time = self.time_getter(item)
        index = self.index_getter(item)
        self._enqueue(time, index, item)
        return index

    def pop(self):
        """
        Get the next item from the queue.

        Raises:
            IndexError: queue is empty

        Return:
            the next item
        """
        if self.empty:
            raise IndexError("attempting to pop from empty queue")
        time, index, event = self._dequeue()
        return event

    def remove(self, *, index=None, predicate=None):
        """
        Remove an item from the queue either by index or by a
        predicate ``(item) -> bool``. If no items found, the method
        silently finishes. If both index and predicate are given,
        the call is equal to sequential calls with separate index
        and predicate arguments.

        Keyword Args:

            index: item index to be removed

            predicate: a logical function ``(item) -> bool``.
                All items matching the predicate would be removed
        """
        if index is not None:
            self._remove_by_index(index)

        if predicate is not None:
            self._remove_by_predicate(predicate)

    def clear(self):
        """
        Remove everything from the queue.
        """
        self._remove_all()

    def __len__(self):
        """
        Returns:
            the number of events in the queue.
        """
        return self._get_queue_length()

    @property
    def items(self):
        """
        Returns:
            a sorted list of events.
        """
        return [item for time, index, item in self._get_as_list()]

    @property
    def empty(self):
        """
        Returns:
            ``True``, if the number of events in the queue is zero.
        """
        return len(self) == 0

    #
    # Abstract methods. Must be implemented in derived classes.
    #
    def has_index(self, index):
        """
        An abstract method. Checks whether the given index is found
        in the queue.

        Args:
            index: an index to check for

        Returns: ``True`` if item with the given index exists.
        """
        raise NotImplementedError()

    def _get_queue_length(self):
        """
        An abstract method. Returns the number of items in the queue.

        Returns: the length of the queue.
        """
        raise NotImplementedError()

    def _get_as_list(self):
        """
        An abstract method. Returns the queue records as a list of
        tuples ``(time, index, item)``.

        Returns: a list of tuples ``(time, index, item)``
        """
        raise NotImplementedError()

    def _enqueue(self, time, index, item):
        """
        An abstract method. Add a new record to the queue.

        Args:

            time: item time

            index: item index

            item: the item itself
        """
        raise NotImplementedError()

    def _dequeue(self):
        """
        An abstract method. Must return a tuple (or a list) with
        ``time``, ``index`` and ``item``.
        If the queue is empty, can either return None or raise an error.

        Returns: ``(time, index, index)``.
        """
        raise NotImplementedError()

    def _remove_by_index(self, index):
        """
        An abstract method. Remove an item by its index.

        Args:
            index: item index
        """
        raise NotImplementedError()

    def _remove_by_predicate(self, predicate):
        """
        An abstract method. Removes all items matching the predicate.

        Args:
            predicate: a logic function: ``(item) -> bool``
        """
        raise NotImplementedError()

    def _remove_all(self):
        """
        An abstract method. Removes everything from the queue.
        """
        raise NotImplementedError()


class HeapQueue(Queue):
    """
    The default implementation of the Queue interface using Python native ``heapq`` library. The basic idea
    is taken from the ``heapq`` documentation:


    The implementation builds an envelope for each event with the format ``[fire_time, index, event]``.
    These lists are stored in two collections:

    - a heap: items are pushed and pop'ped from the list using ``heapq.heappush`` and ``heapq.heappop`` methods;
    sorting is performed by `fire_time` and `index`, so if two events have the same fire_time, the first
    one pushed will be pop'ped first also.

    - an index dictionary: stores data in the format ``{index -> [fire_time, index, event]}``

    When adding a new event, its fire time is extracted using ``event_time_getter`` method and the envelope is
    constructed. Each push operation increases an internal ``index`` counter, so each event will get its unique
    index. Then the envelope is enqueued in the heap and added to dictionaries.
    """
    def __init__(self, time_getter=lambda item: item.time,
                 index_getter=lambda item: item.index):
        super().__init__(time_getter, index_getter)
        self.__heap = []
        self.__index_lookup_table = {}

    def has_index(self, index):
        return index in self.__index_lookup_table

    def _get_queue_length(self):
        return len(self.__index_lookup_table)

    def _get_as_list(self):
        return [(t, i, item) for t, i, item in sorted(self.__heap)
                if item is not None]

    def _enqueue(self, time, index, item):
        entry = [time, index, item]
        self.__index_lookup_table[index] = entry
        heapq.heappush(self.__heap, entry)

    def _dequeue(self):
        while self.__index_lookup_table:
            entry = heapq.heappop(self.__heap)
            time, index, item = entry
            if item is not None:
                del self.__index_lookup_table[index]
                return time, index, item
        return None

    def _remove_by_index(self, index):
        entry = self.__index_lookup_table.pop(index, None)
        if entry is not None:
            entry[-1] = None

    def _remove_by_predicate(self, predicate):
        for time, index, item in self.__heap:
            if item is not None and predicate(item):
                self._remove_by_index(index)

    def _remove_all(self):
        self.__index_lookup_table = {}
        self.__heap = []
