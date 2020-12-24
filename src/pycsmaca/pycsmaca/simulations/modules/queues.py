from collections import deque

from pydesim import Model, Trace, Intervals, Statistic


class QueuedPacket:
    def __init__(self, packet, arrived_at):
        self.packet = packet
        self.arrived_at = arrived_at
    
    @property
    def size(self):
        return self.packet.size
    
    def __str__(self):
        return ('QPkt('
                f'{self.packet.sender_address}->{self.packet.receiver_address}'
                f'{self.packet.size} bits, arrived_at={self.arrived_at})')


class Queue(Model):
    """`Queue` model stores packets and forwards them to services by requests.

    `Queue` module may have finite or infinite capacity, which is specified
    during the `Queue` instance creation. If the capacity is finite, then
    packets pushed when the queue is full will be dropped. In other case,
    the queue will store all the packets being received.

    `Queue` can support multiple services (e.g., network interfaces). It
    delivers packets when a service calls `get_next(service)` method on
    the queue.

    > IMPORTANT: to use a queue, service MUST be connected using
    bi-directional connection.

    If multiple services request packets, they are delivered in the order of
    `get_next()` was called. Each call leads to a single packet being
    transmitted.

    The packet is transmitted to the service immediately after `get_next()`
    call, if the queue is not empty. If the queue is empty, it stores a request
    and send a packet right after a new packet arrives. In the latter case,
    queue size is not updated, and it is counted as the packet went directly
    to the service.

    Packets are sent to the services via `connection.send()` call, and this
    is the reason why the service is meant to be connected to the queue.

    `Queue` accepts packets via its `handle_message()` call.

    Connections:

    - any service that is going to use a queue as a data source MUST be
        connected (these connections can have any names);

    - data producers should also be connected to use `connection.send()` call
        that causes `handle_message(packet)` being called (these connections
        can also have any names).

    > NOTE: queue can also be accessed with `push()` and `pop()` methods,
    so in some cases producers and services may use this module directly,
    without connections (and without `get_next()` also in this case).
    """
    def __init__(self, sim, capacity=None):
        super().__init__(sim)
        self.__capacity = capacity
        self.__packets = deque()
        self.__data_requests = deque()
        # Statistics:
        self.__num_dropped = 0
        self.__num_arrived = 0
        self.__size_trace = Trace()
        self.__bitsize_trace = Trace()
        self.__size_trace.record(sim.stime, 0)
        self.__bitsize_trace.record(sim.stime, 0)
        self.__arrival_intervals = Intervals()
        self.__arrival_intervals.record(self.sim.stime)
        self.__wait_intervals = Statistic()

    @property
    def capacity(self):
        return self.__capacity

    @property
    def num_dropped(self):
        return self.__num_dropped
    
    @property
    def num_arrived(self):
        return self.__num_arrived
    
    @property
    def drop_ratio(self):
        if self.__num_arrived > 0:
            return self.__num_dropped / self.__num_arrived
        return 0

    @property
    def size_trace(self):
        return self.__size_trace

    @property
    def bitsize_trace(self):
        return self.__bitsize_trace
    
    @property
    def arrival_intervals(self):
        return self.__arrival_intervals
    
    @property
    def wait_intervals(self):
        return self.__wait_intervals

    def empty(self):
        return len(self) == 0

    def full(self):
        return len(self) == self.capacity

    def __len__(self):
        return len(self.__packets)

    def size(self):
        return len(self)

    def bitsize(self):
        return sum(pkt.size for pkt in self.__packets)

    def as_tuple(self):
        return tuple(self.__packets)

    def push(self, packet):
        self.__num_arrived += 1
        self.__arrival_intervals.record(self.sim.stime)
        if self.__data_requests:
            connection = self.__data_requests.popleft()
            connection.send(packet)
            self.__wait_intervals.append(0.0)
        else:
            if self.capacity is None or len(self) < self.capacity:
                qp = QueuedPacket(packet, arrived_at=self.sim.stime)
                self.__packets.append(qp)
                self.__size_trace.record(self.sim.stime, len(self))
                self.__bitsize_trace.record(self.sim.stime, self.bitsize())
            else:
                self.__num_dropped += 1

    def pop(self):
        try:
            qp = self.__packets.popleft()
        except IndexError as err:
            raise ValueError('pop from empty Queue') from err
        else:
            self.__size_trace.record(self.sim.stime, len(self))
            self.__bitsize_trace.record(self.sim.stime, self.bitsize())
            self.__wait_intervals.append(self.sim.stime - qp.arrived_at)
            return qp.packet

    def get_next(self, service):
        connection = self._get_connection_to(service)
        if not self.empty():
            connection.send(self.pop())
        else:
            self.__data_requests.append(connection)

    def handle_message(self, message, connection=None, sender=None):
        self.push(message)

    def _get_connection_to(self, module):
        for conn_name, peer in self.connections.as_dict().items():
            if module == peer:
                return self.connections[conn_name]
        raise ValueError(f'connection to {module} not found')

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent is not None else ''
        return f'{prefix}Queue'


class SaturatedQueue(Queue):
    """Saturated queue, that requests source a new packet when it needs it.

    This queue is bound with `ControlledSource` module. When it is empty, and
    one of its connected services requests a packet with `q.get_next(service)`
    call, this queue calls `source.get_next()` for the new packet generation.
    """
    def __init__(self, sim, source, capacity=None):
        super().__init__(sim, capacity)
        self.source = source

    def get_next(self, service):
        if self.empty():
            self.source.get_next()
        super().get_next(service)
