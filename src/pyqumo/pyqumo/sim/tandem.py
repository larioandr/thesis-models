from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
from tabulate import tabulate

from pyqumo.arrivals import RandomProcess
from pyqumo.matrix import str_array
from pyqumo.random import CountableDistribution
from pyqumo.sim.helpers import TimeValue, Statistics, \
    build_distribution_from_time_values, build_statistics, Queue


class Packet:
    """
    Packet representation for tandem G/G/1/N model.

    Stores timestamps, indexes and flags:
    - the first station the packet entered: source
    - when the packet arrived: arrived_time_list[n]
    - when the packet started serve: service_started_time_list[n]
    - when the packet finished serving: service_finished_time_list[n]
    - when the packet was dropped (if was): drop_time
    - where the packet was dropped (if was): drop_node
    - whether the packet was dropped: was_dropped
    - whether the packet was delivered: was_delivered
    - where to deliver: target

    In all lists index is the number of station.
    """
    def __init__(self, source: int, target: int, num_stations: int):
        """
        Create the packet.

        Parameters
        ----------
        source : int
            Node index where the packet was created
        target : int
            Node index where the packet should be delivered
        num_stations : int
            Number of stations in the network
        """

        def create_list() -> List[Optional[float]]:
            return [None] * num_stations

        self.source: int = source
        self.target: int = target

        self.arrived_time_list: List[Optional[float]] = create_list()
        self.service_started_time_list: List[Optional[float]] = create_list()
        self.service_finished_time_list: List[Optional[float]] = create_list()
        self.was_dropped: bool = False
        self.drop_time: Optional[float] = None
        self.drop_node: Optional[int] = None
        self.was_delivered: bool = False
        self.delivered_time: Optional[float] = None
        self.was_served: List[bool] = [False] * num_stations


class Records:
    """
    Statistical records.
    """
    def __init__(self, num_stations: int):
        self.packets_list: List[Packet] = []
        self.system_size_list: List[List[TimeValue]] = \
            [list() for _ in range(num_stations)]

    def add_system_size(self, node: int, time: float, size: int):
        self.system_size_list[node].append(TimeValue(time, size))


CountDistList = List[Optional[CountableDistribution]]


class Results:
    """
    Results returned from G/G/1/N model simulation.

    Discrete stochastic properties like system size, queue size and busy
    periods are represented with `CountableDistribution`. Continuous properties
    are not fitted into any kind of distribution, they are represented with
    `Statistics` tuples.

    Utilization coefficient, as well as loss probability, are just floating
    point numbers.

    To pretty print the results one can make use of `tabulate()` method.
    """
    def __init__(self, num_stations: int):

        def create_list():
            return [None] * num_stations

        self.system_size_list: CountDistList = create_list()
        self.queue_size_list: CountDistList = create_list()
        self.busy_list: CountDistList = create_list()
        self.loss_prob: float = 0.0
        self.departures_list: List[Optional[Statistics]] = create_list()
        self.response_time_list: List[Optional[Statistics]] = create_list()
        self.waiting_time_list: List[Optional[Statistics]] = create_list()
        self.end_to_end_delays_list: List[Optional[Statistics]] = create_list()

        self._num_stations = num_stations

    def get_utilization(self, node: int) -> float:
        """
        Get utilization coefficient, that is `Busy = 1` probability.
        """
        return self.busy_list[node].pmf(1)

    def tabulate(self) -> str:
        """
        Build a pretty formatted table with all key properties.
        """
        items = [
            ('Number of stations', self._num_stations),
            ('Loss probability', self.loss_prob),
        ]

        for node in range(self._num_stations):
            items.append((f'[[ STATION #{node} ]]', ''))

            system_size = self.system_size_list[node]
            queue_size = self.queue_size_list[node]

            system_size_pmf = [
                system_size.pmf(x) for x in range(system_size.truncated_at + 1)
            ]

            queue_size_pmf = [
                queue_size.pmf(x) for x in range(queue_size.truncated_at + 1)
            ]

            items.extend([
                ('System size PMF', str_array(system_size_pmf)),
                ('System size average', system_size.mean),
                ('System size std.dev.', system_size.std),
                ('Queue size PMF', str_array(queue_size_pmf)),
                ('Queue size average', queue_size.mean),
                ('Queue size std.dev.', queue_size.std),
                ('Utilization', self.get_utilization(node)),
                ('Departures, average', self.departures_list[node].avg),
                ('Departures, std.dev.', self.departures_list[node].std),
                ('Response time, average', self.response_time_list[node].avg),
                ('Response time, std.dev.', self.response_time_list[node].std),
                ('Wait time, average', self.waiting_time_list[node].avg),
                ('Wait time, std.dev.', self.waiting_time_list[node].std),
            ])
        return tabulate(items, headers=('Param', 'Value'))


@dataclass
class Params:
    """
    Model parameters: arrival and service processes, queue capacity and limits.
    """
    arrival: RandomProcess
    services: List[RandomProcess]
    num_stations: int
    queue_capacity: int
    max_packets: int = 1000000
    max_time: float = np.inf


class Event(Enum):
    STOP = 0
    ARRIVAL = 1
    SERVICE_END = 2


class System:
    """
    System state representation.

    This object takes care of the queue, current time, next arrival and
    service end time, server status and any other kind of dynamic information,
    except for internal state of arrival or service processes.
    """
    def __init__(self, params: Params):
        """
        Constructor.

        Parameters
        ----------
        params : Params
            Model parameters
        """
        n: int = params.num_stations
        self.queues_list: List[Queue[Packet]] = [
            Queue(params.queue_capacity) for _ in range(params.num_stations)]
        self._time: float = 0.0
        self._service_ends_list: List[Optional[float]] = [None] * n
        self._next_arrival: float = 0.0
        self.servers_list: List[Optional[Packet]] = [None] * n
        self._stopped: bool = False

    @property
    def time(self):
        return self._time

    @property
    def stopped(self):
        return self._stopped

    def reset_time(self):
        self._time = 0.0

    def stop(self):
        self._stopped = True

    def is_empty(self, node: int) -> bool:
        """
        Return `True` if server is not serving any packet.
        """
        return self.servers_list[node] is None

    def get_size(self, node: int):
        """
        Get system size, that is queue size plus one (busy) or zero (empty).
        """
        server_size = 0 if self.servers_list[node] is None else 1
        return server_size + self.queues_list[node].size

    def schedule(self, event: Event, interval: float, node: int = 0):
        if event == Event.ARRIVAL:
            self._next_arrival = self._time + interval
        else:
            assert event == Event.SERVICE_END, event
            self._service_ends_list[node] = self._time + interval

    def next_event(self) -> Tuple[Event, int]:
        """
        Get next event type and move time.

        Returns
        -------
        pair : tuple of event and station index
        """
        min_service_end: float = np.inf
        min_service_at: int = len(self._service_ends_list)
        for i, service_end in enumerate(self._service_ends_list):
            if service_end is not None and service_end < min_service_end:
                min_service_end = service_end
                min_service_at = i

        # Check whether service ends at some station before next arrival:
        if min_service_end < self._next_arrival:
            self._time = min_service_end
            self._service_ends_list[min_service_at] = None
            return Event.SERVICE_END, min_service_at

        # No service ends, or any service end is greater then arrival time:
        self._time = self._next_arrival
        self._next_arrival = None
        return Event.ARRIVAL, 0  # always arrive at the first node


def simulate(
        arrival: RandomProcess,
        service: RandomProcess,
        queue_capacity: int,
        num_stations: int,
        max_time: float = np.inf,
        max_packets: int = 1000000
) -> Results:
    """
    Run simulation model of G/G/1/N system.

    Simulation can be stopped in two ways: by reaching maximum simulation time,
    or by reaching the maximum number of generated packets. By default,
    simulation is limited with the maximum number of packets only (1 million).

    Queue is expected to have finite capacity.

    Arrival and service time processes can be of any kind, including Poisson
    or MAP. To use a PH or normal distribution, a GenericIndependentProcess
    model with the corresponding distribution may be used.

    Parameters
    ----------
    arrival : RandomProcess
        Arrival random process.
    service : RandomProcess
        Service time random process.
    queue_capacity : int
        Queue capacity.
    num_stations : int
        Number of stations in the network
    max_time : float, optional
        Maximum simulation time (default: infinity).
    max_packets
        Maximum number of simulated packets (default: 1'000'000)

    Returns
    -------
    results : Results
        Simulation results.
    """
    params = Params(
        arrival=arrival,
        services=[service.copy() for _ in range(num_stations)],
        queue_capacity=queue_capacity,
        num_stations=num_stations,
        max_packets=max_packets,
        max_time=max_time
    )
    system = System(params)
    records = Records(num_stations)

    _init(system, params, records)

    max_time = params.max_time
    while not system.stopped:
        # Extract new event:
        event, node = system.next_event()

        # Check whether event is scheduled too late, and we need to stop:
        if system.time > max_time:
            system.stop()
            continue

        # Process event
        if event == Event.ARRIVAL:
            _handle_arrival(node, system, params, records)
        elif event == Event.SERVICE_END:
            _handle_service_end(node, system, params, records)

    return _build_results(params, system, records)


def _init(system: System, params: Params, records: Records):
    """
    Initialize the model: set time to true, init statistics, schedule arrival.

    Parameters
    ----------
    system : System
    params : Params
    records : Records
    """
    system.reset_time()
    system.schedule(Event.ARRIVAL, params.arrival())
    for node in range(params.num_stations):
        records.system_size_list[node].append(TimeValue(0, 0))


def _process_packet(node: int, packet: Packet, system: System, params: Params,
                    records: Records):
    now = system.time
    packet.arrived_time_list[node] = now

    # If server is ready, start serving. Otherwise, push the packet into
    # the queue. If the queue was full, mark the packet is being dropped
    # for further analysis.
    if system.is_empty(node):
        # start serving immediately
        system.servers_list[node] = packet
        packet.service_started_time_list[node] = now
        system.schedule(Event.SERVICE_END, params.services[node]())
        records.add_system_size(node, now, system.get_size(node))

    elif system.queues_list[node].push(packet):
        # packet was queued
        records.add_system_size(node, now, system.get_size(node))

    else:
        # mark packet as being dropped
        packet.was_dropped = True
        packet.drop_node = node
        packet.drop_time = now


def _handle_arrival(node: int, system: System, params: Params,
                    records: Records):
    """
    Handle new packet arrival event.

    First of all, a new packet is created. Then we check whether the
    system is empty. If it is, this new packet starts serving immediately.
    Otherwise, it is added to the queue.

    If the queue was full, the packet is dropped. To mark this, we set
    `dropped` flag in the packet to `True`.

    In the end we schedule the next arrival. We also check whether the
    we have already generated enough packets. If so, `system.stopped` flag
    is set to `True`, so on the next main loop iteration the simulation
    will be stopped.

    Parameters
    ----------
    node : int
    system : System
    params : Params
    records : Records
    """
    num_packets_built = len(records.packets_list)
    if num_packets_built >= params.max_packets:
        # If too many packets were generated, ask to stop:
        system.stop()

    now = system.time
    packet = Packet(node, params.num_stations - 1, params.num_stations)
    records.packets_list.append(packet)

    # Handle packet:
    _process_packet(node, packet, system, params, records)

    # Schedule next arrival:
    system.schedule(Event.ARRIVAL, params.arrival())


def _handle_service_end(node: int, system: System, params: Params,
                        records: Records):
    """
    Handle end of the packet service.

    If the queue is empty, the server becomes idle. Otherwise, it starts
    serving the next packet from the queue.

    The packet that left the server is marked as `served = True`.

    Parameters
    ----------
    system : System
    params : Params
    records : Records
    """
    now = system.time
    packet = system.servers_list[node]
    system.servers_list[node] = None

    # Record packet service end:
    assert packet is not None
    packet.service_finished_time_list[node] = now
    packet.was_served[node] = True

    # If this node is the target, then packet is delivered.
    # Otherwise, pass it to the next node.
    if packet.target == node:
        packet.was_delivered = True
        packet.delivered_time = now
    else:
        _process_packet(node + 1, packet, system, params, records)

    # Start serving next packet, if exists:
    packet = system.queues_list[node].pop()
    if packet is not None:
        assert isinstance(packet, Packet)
        system.servers_list[node] = packet
        packet.service_started_time_list[node] = now
        system.schedule(Event.SERVICE_END, params.services[node]())

    # Record new system size:
    records.add_system_size(node, now, system.get_size(node))


def _build_results(params: Params, system: System, records: Records) -> Results:
    """
    Build Results instance from the collected records.

    This method is called after the main loop finished.

    Parameters
    ----------
    system : System
    records : Records

    Returns
    -------

    """
    results = Results(params.num_stations)

    num_dropped = len([pkt for pkt in records.packets_list if pkt.was_dropped])
    results.loss_prob = num_dropped / len(records.packets_list)

    for node in range(params.num_stations):
        system_sizes = records.system_size_list[node]
        queue_sizes = \
            [TimeValue(t, (x - 1 if x > 1 else 0)) for (t, x) in system_sizes]

        results.system_size_list[node] = \
            build_distribution_from_time_values(system_sizes)
        results.queue_size_list[node] = \
            build_distribution_from_time_values(queue_sizes)

        p0 = results.system_size_list[node].pmf(0)
        results.busy_list[node] = CountableDistribution(
            lambda x: p0 if x == 0 else (1 - p0) if x == 1 else 0,
            precision=1e-12)

        served_packets = \
            [pkt for pkt in records.packets_list if pkt.was_served[node]]
        wait_intervals = [
            pkt.service_started_time_list[node] - pkt.arrived_time_list[node]
            for pkt in served_packets
        ]
        response_intervals = [
            pkt.service_finished_time_list[node] - pkt.arrived_time_list[node]
            for pkt in served_packets
        ]
        departure_intervals, last_departure = [], 0.0
        for packet in served_packets:
            interval = packet.service_finished_time_list[node] - last_departure
            departure_intervals.append(interval)
            last_departure = packet.service_finished_time_list[node]

        results.waiting_time_list[node] = build_statistics(wait_intervals)
        results.response_time_list[node] = build_statistics(response_intervals)
        results.departures_list[node] = build_statistics(departure_intervals)

    return results
