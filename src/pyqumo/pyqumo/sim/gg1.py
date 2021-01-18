from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Generic, TypeVar, Tuple, Sequence
import numpy as np
from tabulate import tabulate

from pyqumo.arrivals import RandomProcess
from pyqumo.matrix import str_array
from pyqumo.random import CountableDistribution


@dataclass
class Packet:
    created_at: float
    service_started_at: Optional[float] = None
    departed_at: Optional[float] = None
    dropped: bool = False
    served: bool = False


TimeValue = namedtuple('TimeValue', ['time', 'value'])
Statistics = namedtuple('Statistics', ['avg', 'var', 'std', 'count'])


@dataclass
class Records:
    packets: List[Packet] = field(default_factory=list)
    system_size: List[TimeValue] = field(default_factory=list)
    queue_size: List[TimeValue] = field(default_factory=list)


@dataclass
class Results:
    system_size: CountableDistribution
    queue_size: CountableDistribution
    busy: CountableDistribution
    loss_prob: float
    departures: Statistics
    response_time: Statistics
    wait_time: Statistics

    @property
    def utilization(self) -> float:
        return self.busy.pmf(1)

    def tabulate(self) -> str:
        system_size_pmf = [
            self.system_size.pmf(x)
            for x in range(self.system_size.truncated_at + 1)]
        queue_size_pmf = [
            self.queue_size.pmf(x)
            for x in range(self.queue_size.truncated_at + 1)]

        items = [
            ('System size PMF', str_array(system_size_pmf)),
            ('System size average', str_array(queue_size_pmf)),
            ('System size std.dev.', self.system_size.std),
            ('Queue size PMF', [f'{x:.3g}' for x in queue_size_pmf]),
            ('Queue size average', self.queue_size.mean),
            ('Queue size std.dev.', self.queue_size.std),
            ('Utilization', self.utilization),
            ('Loss probability', self.loss_prob),
            ('Departures, average', self.departures.avg),
            ('Departures, std.dev.', self.departures.std),
            ('Response time, average', self.response_time.avg),
            ('Response time, std.dev.', self.response_time.std),
            ('Wait time, average', self.wait_time.avg),
            ('Wait time, std.dev.', self.wait_time.std),
        ]
        return tabulate(items, headers=('Param', 'Value'))


T = TypeVar('T')


class Queue(Generic[T]):
    def __init__(self, capacity: int = np.inf):
        self._items: List[T] = []
        self._capacity = capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size(self) -> int:
        return len(self._items)
    
    @property
    def empty(self) -> bool:
        return len(self._items) == 0
    
    @property
    def full(self) -> bool:
        return len(self._items) >= self._capacity

    def push(self, item: T) -> bool:
        if len(self._items) < self._capacity:
            self._items.append(item)
            return True
        return False

    def pop(self) -> Optional[T]:
        if not self._items:
            return None
        head = self._items[0]
        self._items = self._items[1:]
        return head


@dataclass
class Params:
    arrival: RandomProcess
    service: RandomProcess
    queue_capacity: int = np.inf
    max_packets: int = 1000000
    max_time: float = np.inf


class Event(Enum):
    STOP = 0
    ARRIVAL = 1
    SERVICE_END = 2


class System:
    def __init__(self, params: Params):
        self.queue: Queue[Packet] = Queue(params.queue_capacity)
        self.time: float = 0.0
        self.service_end: Optional[float] = None
        self.next_arrival: float = 0.0
        self.server: Optional[Packet] = None
        self.stopped: bool = False

    @property
    def empty(self) -> bool:
        return self.server is None

    @property
    def size(self):
        return (1 if self.server is not None else 0) + self.queue.size

    def get_next_event(self) -> Tuple[Event, float]:
        """
        Get next event type and its firing time.

        Returns
        -------
        pair : tuple of event and time
        """
        if self.service_end is None or self.service_end > self.next_arrival:
            return Event.ARRIVAL, self.next_arrival
        return Event.SERVICE_END, self.service_end


def init(system: System, params: Params, records: Records):
    system.time = 0.0
    system.next_arrival = params.arrival()
    records.queue_size.append(TimeValue(0, 0))
    records.system_size.append(TimeValue(0, 0))


def handle_arrival(system: System, params: Params, records: Records):
    num_packets_built = len(records.packets)
    if num_packets_built >= params.max_packets:
        # If too many packets were generated, ask to stop:
        system.stopped = True

    now = system.time
    packet = Packet(created_at=now)
    records.packets.append(packet)

    # If server is ready, start serving. Otherwise, push the packet into
    # the queue. If the queue was full, mark the packet is being dropped
    # for further analysis.
    if system.empty:
        # start serving immediately
        system.server = packet
        system.service_end = now + params.service()
        packet.service_started_at = now
        records.system_size.append(TimeValue(now, system.size))
    elif system.queue.push(packet):
        # packet was queued
        records.system_size.append(TimeValue(now, system.size))
        records.queue_size.append(TimeValue(now, system.queue.size))
    else:
        # mark packet as being dropped
        packet.dropped = True

    # Schedule next arrival:
    interval = params.arrival()
    system.next_arrival = now + interval


def handle_service_end(system: System, params: Params, records: Records):
    now = system.time
    packet = system.server
    assert packet is not None
    system.server = None  # to be sure that packet is not referenced anymore
    packet.served = True

    # Record packet service end:
    packet.departed_at = now

    # Start serving next packet, if exists:
    packet = system.queue.pop()
    if packet is not None:
        system.server = packet
        packet.service_started_at = now
        system.service_end = now + params.service()
        # Record updated queue size:
        records.queue_size.append(TimeValue(now, system.queue.size))
    else:
        system.service_end = None

    # Record new system size:
    records.system_size.append(TimeValue(now, system.size))


def run_main_loop(system: System, params: Params, records: Records):
    max_time = params.max_time
    while not system.stopped:
        # Extract new event:
        event, new_time = system.get_next_event()
        assert (new_time - system.time) > -1e-11

        # Check whether event is scheduled too late, and we need to stop:
        if new_time > max_time:
            system.stopped = True
            continue

        # Process event
        system.time = new_time
        if event == Event.ARRIVAL:
            handle_arrival(system, params, records)
        elif event == Event.SERVICE_END:
            handle_service_end(system, params, records)


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
    avg = np.mean(intervals)
    var = np.var(intervals, ddof=1)  # unbiased estimate
    std = var**0.5
    return Statistics(avg=avg, var=var, std=std, count=len(intervals))


def build_results(system: System, records: Records) -> Results:
    # 1) Extract queue and system size distributions:
    queue_size = build_distribution_from_time_values(records.queue_size)
    system_size = build_distribution_from_time_values(records.system_size)

    # 2) To build utilization, we use already built system_size distribution:
    p0 = system_size.pmf(0)
    busy = CountableDistribution(
        lambda x: p0 if x == 0 else (1 - p0) if x == 1 else 0,
        precision=1e-12)

    # 3) Loss probability is just a ration of dropped packets:
    num_dropped = len([pkt for pkt in records.packets if pkt.dropped])
    loss_prob = num_dropped / len(records.packets)

    # 4) To study departure intervals, response and wait times we need
    #    to inspect packet timestamps. Let's extract the intervals.
    served_packets = [pkt for pkt in records.packets if pkt.served]
    wait_intervals = [
        pkt.service_started_at - pkt.created_at for pkt in served_packets]
    response_intervals = [
        pkt.departed_at - pkt.created_at for pkt in served_packets]

    departure_intervals, last_departure = [], 0.0
    for packet in served_packets:
        departure_intervals.append(packet.departed_at - last_departure)
        last_departure = packet.departed_at

    # 5) Build result:
    return Results(
        system_size=system_size,
        queue_size=queue_size,
        busy=busy,
        loss_prob=loss_prob,
        departures=build_statistics(departure_intervals),
        response_time=build_statistics(response_intervals),
        wait_time=build_statistics(wait_intervals))


def simulate(
        arrival: RandomProcess,
        service: RandomProcess,
        queue_capacity: int,
        max_time: float = np.inf,
        max_packets: int = 1000000):
    """
    Run simulation model of G/G/1/N system.

    Parameters
    ----------
    arrival
    service
    queue_capacity
    max_time
    max_packets

    Returns
    -------

    """
    params = Params(
        arrival=arrival, service=service, queue_capacity=queue_capacity,
        max_packets=max_packets, max_time=max_time)
    system = System(params)
    records = Records()

    init(system, params, records)
    run_main_loop(system, params, records)
    return build_results(system, records)
