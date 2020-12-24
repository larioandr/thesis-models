from pydesim import Model

from pycsmaca.simulations.modules import RandomSource, WiredTransceiver, Queue, \
    WiredInterface
from pycsmaca.simulations.modules.station import Station


class WiredLineNetwork(Model):
    def __init__(self, sim):
        super().__init__(sim)

        if sim.params.num_stations < 2:
            raise ValueError('minimum number of stations in network is 2')

        # Reading parameters and building stations:
        stations = []
        next_address, n = 1, sim.params.num_stations
        destination_address = 2 + (n - 2) * 2
        for i in range(n):
            if i in sim.params.active_sources:
                source = RandomSource(
                    sim, sim.params.payload_size, sim.params.source_interval,
                    source_id=i, dest_addr=destination_address
                )
            else:
                source = None

            # Building wired interfaces:
            interfaces = []
            for _ in range(2 if 0 < i < n - 1 else 1):
                transceiver = WiredTransceiver(
                    sim, bitrate=sim.params.bitrate,
                    header_size=sim.params.header_size,
                    preamble=sim.params.preamble,
                    ifs=sim.params.ifs,
                )
                queue = Queue(sim)
                iface = WiredInterface(sim, next_address, queue, transceiver)
                next_address += 1
                interfaces.append(iface)

            # Building station:
            sta = Station(sim, source=source, interfaces=interfaces)
            stations.append(sta)

            # Writing routing table for all station except the last one:
            if i < n - 1:
                out_iface = interfaces[-1]
                switch_conn = sta.get_switch_connection_for(out_iface)
                sta.switch.table.add(
                    destination_address,
                    switch_conn.name,
                    out_iface.address + 1
                )

        # Adding stations as children:
        self.children['stations'] = stations

        # Connecting stations interfaces in chain:
        for i in range(n - 1):
            if1, if2 = stations[i].interfaces[-1], stations[i+1].interfaces[0]
            conn = if1.connections.set('wire', if2, rname='wire')
            conn.delay = sim.params.distance / sim.params.speed_of_light

    @property
    def stations(self):
        return self.children['stations']

    @property
    def clients(self):
        return self.stations[:-1]

    @property
    def server(self):
        return self.stations[-1]

    @property
    def num_stations(self):
        return len(self.stations)

    def get_tx_iface(self, sta_index):
        if sta_index < self.num_stations - 1:
            return self.stations[sta_index].interfaces[-1]
        raise ValueError(
            f'TX interface defined for the first {self.num_stations - 1} '
            f'stations only')

    def get_rx_iface(self, sta_index):
        if sta_index > 0:
            return self.stations[sta_index].interfaces[0]
        raise ValueError(
            f'RX interface defined for the last {self.num_stations - 1} '
            f'stations only')

    def __str__(self):
        return 'Network'

    def print_children(self):
        def get_all_leafs(module):
            children = [module]
            for child in module.children.all():
                children.extend(get_all_leafs(child))
            return children

        print('Network components:')
        for m in get_all_leafs(self):
            print(f'+ {m}')

    # noinspection PyTypeChecker
    def describe_topology(self):
        def str_sid(c):
            return c.source.source_id if c.source else '<NONE>'

        def str_ifaces(c):
            _prefix = "\n\t\t\t"
            return _prefix + _prefix.join([
                f'[addr:{iface.address}], '
                f'connected to: {iface.connections["wire"].module.address}'
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
            'index', 'service_time', 'queue_size', 'tx_busy', 'rx_busy',
            'arrival_intervals', 'num_packets_sent', 'delay', 'sid',
        ]
        server_fields = [
            'arrival_intervals', 'num_packets_received',
        ]
        client_class = namedtuple('Client', client_fields)
        server_class = namedtuple('Server', server_fields)

        _client_sources = [cli.source for cli in self.clients]
        _client_ifaces = [(cli.interfaces[0], cli.interfaces[-1])
                          for cli in self.clients]
        _srv = self.server

        clients = [
            client_class(
                index=i,
                service_time=out_if.transceiver.service_time.mean(),
                queue_size=out_if.queue.size_trace.timeavg(),
                tx_busy=out_if.transceiver.tx_busy_trace.timeavg(),
                rx_busy=inp_if.transceiver.rx_busy_trace.timeavg(),
                arrival_intervals=(
                    src.arrival_intervals.statistic().mean() if src else None),
                num_packets_sent=out_if.transceiver.num_transmitted_packets,
                delay=(
                    _srv.sink.source_delays.get(
                        src.source_id).mean() if src else None),
                sid=(src.source_id if src else None),
            ) for i, (src, (inp_if, out_if)) in enumerate(
                zip(_client_sources, _client_ifaces))
        ]
        server = server_class(
            arrival_intervals=_srv.sink.arrival_intervals.statistic().mean(),
            num_packets_received=_srv.sink.num_packets_received,
        )
        return (client_fields, clients), (server_fields, server)
