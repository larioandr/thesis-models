from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
from tabulate import tabulate

from pyqumo.arrivals import RandomProcess
from pyqumo.matrix import str_array
from pyqumo.random import CountableDistribution
from pyqumo.sim.helpers import TimeValue, Statistics, \
    build_distribution_from_time_values, build_statistics, FiniteFifoQueue, \
    InfiniteFifoQueue


@dataclass
class Packet:
    """
    Packet representation for G/G/1/N model.

    Stores timestamps: when the packet arrived (created_at), started
    serving (service_started_at) and finished serving (departed_at).

    Also stores flags, indicating whether the packet was dropped or was
    completely served. Packets arrived in the end of the modeling, may not
    be served, as well as dropped packets.
    """
    created_at: float
    service_started_at: Optional[float] = None
    departed_at: Optional[float] = None
    dropped: bool = False
    served: bool = False


@dataclass
class Records:
    """
    Records for G/G/1/N statistical analysis.
    """
    packets: List[Packet] = field(default_factory=list)
    system_size: List[TimeValue] = field(default_factory=list)
    queue_size: List[TimeValue] = field(default_factory=list)


@dataclass
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
    system_size: CountableDistribution
    queue_size: CountableDistribution
    busy: CountableDistribution
    loss_prob: float
    departures: Statistics
    response_time: Statistics
    wait_time: Statistics

    @property
    def utilization(self) -> float:
        """
        Get utilization coefficient, that is `Busy = 1` probability.
        """
        return self.busy.pmf(1)

    def tabulate(self) -> str:
        """
        Build a pretty formatted table with all key properties.
        """
        system_size_pmf = [
            self.system_size.pmf(x)
            for x in range(self.system_size.truncated_at + 1)]
        queue_size_pmf = [
            self.queue_size.pmf(x)
            for x in range(self.queue_size.truncated_at + 1)]

        items = [
            ('System size PMF', str_array(system_size_pmf)),
            ('System size average', self.system_size.mean),
            ('System size std.dev.', self.system_size.std),
            ('Queue size PMF', str_array(queue_size_pmf)),
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


@dataclass
class Params:
    """
    Model parameters: arrival and service processes, queue capacity and limits.
    """
    arrival: RandomProcess
    service: RandomProcess
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
        if params.queue_capacity < np.inf:
            self.queue = FiniteFifoQueue(params.queue_capacity)
        else:
            self.queue = InfiniteFifoQueue()

        self.time: float = 0.0
        self.service_end: Optional[float] = None
        self.next_arrival: float = 0.0
        self.server: Optional[Packet] = None
        self.stopped: bool = False

    @property
    def empty(self) -> bool:
        """
        Return `True` if server is not serving any packet.
        """
        return self.server is None

    @property
    def size(self):
        """
        Get system size, that is queue size plus one (busy) or zero (empty).
        """
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


def simulate(
        arrival: RandomProcess,
        service: RandomProcess,
        queue_capacity: int = np.inf,
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
        arrival=arrival, service=service, queue_capacity=queue_capacity,
        max_packets=max_packets, max_time=max_time)
    system = System(params)
    records = Records()

    _init(system, params, records)

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
            _handle_arrival(system, params, records)
        elif event == Event.SERVICE_END:
            _handle_service_end(system, params, records)

    return _build_results(system, records)


def _init(system: System, params: Params, records: Records):
    """
    Initialize the model: set time to true, init statistics, schedule arrival.

    Parameters
    ----------
    system : System
    params : Params
    records : Records
    """
    system.time = 0.0
    system.next_arrival = params.arrival()
    records.queue_size.append(TimeValue(0, 0))
    records.system_size.append(TimeValue(0, 0))


def _handle_arrival(system: System, params: Params, records: Records):
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
    system : System
    params : Params
    records : Records
    """
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


def _handle_service_end(system: System, params: Params, records: Records):
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


def _build_results(system: System, records: Records) -> Results:
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
