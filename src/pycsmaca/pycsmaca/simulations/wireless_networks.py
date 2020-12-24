from math import pi, cos, sin

from numpy.random.mtrand import uniform
from pydesim import Model

from pycsmaca.simulations.modules import RandomSource, Queue, Transmitter, \
    Receiver, Radio, ConnectionManager, WirelessInterface, SaturatedQueue
from pycsmaca.simulations.modules.app_layer import ControlledSource
from pycsmaca.simulations.modules.station import Station


class _HalfDuplexNetworkBase(Model):
    def __init__(self, sim):
        super().__init__(sim)

        if sim.params.num_stations < 2:
            raise ValueError('minimum number of stations in network is 2')

        # Building connection manager:
        self.__conn_manager = ConnectionManager(sim)

        self.__stations = []

        conn_radius = sim.params.connection_radius
        for i in range(sim.params.num_stations):
            # Building elementary components:
            source = self.create_source(i)
            max_propagation = conn_radius / sim.params.speed_of_light
            transmitter = Transmitter(sim, max_propagation=max_propagation)
            receiver = Receiver(sim)
            queue = self.create_queue(i, source=source)
            radio = Radio(
                sim, self.__conn_manager,
                connection_radius=conn_radius,
                position=self.get_position(i)
            )

            # Building wireless interfaces:
            iface = WirelessInterface(sim, i + 1, queue, transmitter,
                                      receiver, radio)

            # Building station:
            sta = Station(sim, source=source, interfaces=[iface])
            self.__stations.append(sta)

            # Writing switching table:
            self.write_switch_table(i)

        # Adding stations as children:
        self.children['stations'] = self.__stations

    @property
    def destination_address(self):
        raise NotImplementedError

    def create_source(self, index):
        raise NotImplementedError

    def create_queue(self, index, source=None):
        return Queue(self.sim)

    def get_position(self, index):
        raise NotImplementedError

    def write_switch_table(self, index):
        raise NotImplementedError

    @property
    def stations(self):
        return self.__stations

    @property
    def connection_manager(self):
        return self.__conn_manager

    @property
    def num_stations(self):
        return len(self.stations)

    def get_iface(self, index):
        if index < self.num_stations:
            return self.stations[index].interfaces[-1]
        raise ValueError(f'station index {index} out of bounds')

    @property
    def clients(self):
        raise NotImplementedError

    @property
    def server(self):
        return NotImplementedError

    def __str__(self):
        return 'Network'

    # noinspection PyTypeChecker
    def describe_topology(self):
        def str_sid(c):
            return c.source.source_id if c.source else '<NONE>'

        def str_peers(iface):
            _peers = self.connection_manager.get_peers(iface.radio)
            return ", ".join([str(peer.parent.address) for peer in _peers])

        def str_ifaces(c):
            _prefix = "\n\t\t\t"
            return _prefix + _prefix.join([
                f'[addr:{iface.address}], sends to: {str_peers(iface)}'
                for i, iface in enumerate(c.interfaces)
            ])

        def str_sw_table(c):
            d = c.switch.table.as_dict()
            if not d:
                return 'EMPTY'
            return "\n\t\t\t" + "\n\t\t\t".join([
                f'{key} via {val[1]} (interface "{val[0]}")'
                for key, val in d.items()
            ])

        s1 = 'NETWORK TOPOLOGY'
        s2 = f'- num stations: {self.num_stations}'
        s3 = '- clients:\n\t' + '\n\t'.join([
            f'{i}: SID={str_sid(cli)}\n\t\t- interfaces: {str_ifaces(cli)}'
            f'\n\t\t- switching table:{str_sw_table(cli)}'
            for i, cli in enumerate(self.clients)
        ])
        s4 = f'- server:\n\t\t- interfaces: {str_ifaces(self.server)}'
        return '\n'.join([s1, s2, s3, s4])

    # noinspection PyUnresolvedReferences
    def get_stats(self):
        from collections import namedtuple
        client_fields = [
            'index', 'service_time', 'num_retries', 'queue_size', 'tx_busy',
            'rx_busy', 'arrival_intervals', 'num_packets_sent', 'delay', 'sid',
            'num_rx_collided', 'num_rx_success',
        ]
        server_fields = [
            'arrival_intervals', 'num_rx_collided', 'num_rx_success',
            'num_packets_received',
        ]
        client_class = namedtuple('Client', client_fields)
        server_class = namedtuple('Server', server_fields)

        _client_sources = [cli.source for cli in self.clients]
        _client_ifaces = [cli.interfaces[0] for cli in self.clients]
        _srv = self.server

        clients = [
            client_class(
                index=i,
                service_time=iface.transmitter.service_time.mean(),
                num_retries=iface.transmitter.num_retries_vector.mean(),
                queue_size=iface.queue.size_trace.timeavg(),
                tx_busy=iface.transmitter.busy_trace.timeavg(),
                rx_busy=iface.receiver.busy_trace.timeavg(),
                arrival_intervals=(
                    src.arrival_intervals.statistic().mean() if src else None),
                num_packets_sent=iface.transmitter.num_sent,
                delay=(
                    _srv.sink.source_delays.get(src.source_id).mean()
                    if src else None),
                sid=(src.source_id if src else None),
                num_rx_collided=iface.receiver.num_collisions,
                num_rx_success=iface.receiver.num_received,
            ) for i, (src, iface) in enumerate(
                zip(_client_sources, _client_ifaces))
        ]
        server = server_class(
            arrival_intervals=_srv.sink.arrival_intervals.statistic().mean(),
            num_rx_collided=_srv.interfaces[0].receiver.num_collisions,
            num_rx_success=_srv.interfaces[0].receiver.num_received,
            num_packets_received=_srv.sink.num_packets_received,
        )
        return (client_fields, clients), (server_fields, server)


class WirelessHalfDuplexLineNetwork(_HalfDuplexNetworkBase):
    def __init__(self, sim):
        super().__init__(sim)

    def create_source(self, index):
        if index in self.sim.params.active_sources:
            return RandomSource(
                self.sim,
                self.sim.params.payload_size,
                self.sim.params.source_interval,
                source_id=index,
                dest_addr=self.destination_address
            )
        return None

    @property
    def destination_address(self):
        return self.sim.params.num_stations

    def get_position(self, index):
        return index * self.sim.params.distance, 0

    def write_switch_table(self, index):
        if index < self.sim.params.num_stations - 1:
            sta = self.stations[index]
            iface = sta.interfaces[0]
            switch_conn = sta.get_switch_connection_for(iface)
            sta.switch.table.add(
                self.destination_address,
                switch_conn.name,
                iface.address + 1
            )

    @property
    def clients(self):
        return self.stations[:-1]

    @property
    def server(self):
        return self.stations[-1]


class CollisionDomainNetwork(_HalfDuplexNetworkBase):
    def __init__(self, sim):
        super().__init__(sim)

    @property
    def destination_address(self):
        return 1

    def create_source(self, index):
        if index > 0:
            return RandomSource(
                self.sim, self.sim.params.payload_size,
                self.sim.params.source_interval,
                source_id=index, dest_addr=self.destination_address
            )
        return None

    def get_position(self, index):
        area_radius = self.sim.params.connection_radius / 2.1
        distance, angle = uniform(0.1, 1) * area_radius, uniform(0, 2 * pi)
        position = (distance * cos(angle), distance * sin(angle))
        return position

    def write_switch_table(self, index):
        if index > 0:
            sta = self.stations[index]
            iface = sta.interfaces[0]
            switch_conn = sta.get_switch_connection_for(iface)
            sta.switch.table.add(
                self.destination_address,
                switch_conn.name,
                self.destination_address
            )

    @property
    def clients(self):
        return tuple(self.stations[1:])

    @property
    def server(self):
        return self.stations[0]


class CollisionDomainSaturatedNetwork(CollisionDomainNetwork):
    def __init__(self, sim):
        super().__init__(sim)

    def create_source(self, index):
        if index > 0:
            return ControlledSource(
                self.sim, self.sim.params.payload_size,
                source_id=index, dest_addr=self.destination_address
            )
        return None

    def create_queue(self, index, source=None):
        if index > 0:
            return SaturatedQueue(self.sim, source=source)
        return Queue(self.sim)

