from collections import namedtuple

from .wireless_networks import CollisionDomainNetwork, \
    CollisionDomainSaturatedNetwork, WirelessHalfDuplexLineNetwork
from .wired_networks import WiredLineNetwork
from pydesim import simulate, Logger


SPEED_OF_LIGHT = 299792458.0


def collision_domain_network(
        num_clients, payload_size, source_interval, ack_size, mac_header_size,
        phy_header_size, preamble, bitrate, difs, sifs, slot, cwmin, cwmax,
        queue_capacity=None, connection_radius=100,
        speed_of_light=SPEED_OF_LIGHT, sim_time_limit=1000,
        log_level=Logger.Level.INFO):
    ret = simulate(
        CollisionDomainNetwork,
        stime_limit=sim_time_limit,
        params=dict(
            num_stations=(num_clients + 1),
            payload_size=payload_size,
            source_interval=source_interval,
            mac_header_size=mac_header_size,
            phy_header_size=phy_header_size,
            ack_size=ack_size,
            preamble=preamble,
            bitrate=bitrate,
            difs=difs,
            sifs=sifs,
            slot=slot,
            cwmin=cwmin,
            cwmax=cwmax,
            connection_radius=connection_radius,
            speed_of_light=speed_of_light,
            queue_capacity=queue_capacity,
        ), loglevel=log_level
    )

    simret_class = namedtuple('SimRet', ['clients', 'server', 'network'])
    client_class = namedtuple('Client', [
        'service_time', 'num_retries', 'queue_size', 'busy',
        'source_intervals', 'num_packets_sent', 'queue_drop_ratio',
        'queue_wait',
    ])
    server_class = namedtuple('Server', [
        'arrival_intervals', 'num_rx_collided', 'num_rx_success',
        'num_packets_received', 'collision_ratio',
    ])

    clients = [
        client_class(
            service_time=cli.interfaces[0].transmitter.service_time,
            num_retries=cli.interfaces[0].transmitter.num_retries_vector,
            queue_size=cli.interfaces[0].queue.size_trace,
            busy=cli.interfaces[0].transmitter.busy_trace,
            source_intervals=cli.source.arrival_intervals.statistic(),
            num_packets_sent=cli.interfaces[0].transmitter.num_sent,
            queue_drop_ratio=cli.interfaces[0].queue.drop_ratio,
            queue_wait=cli.interfaces[0].queue.wait_intervals,
        ) for cli in ret.data.clients
    ]

    srv = ret.data.server
    server = server_class(
        arrival_intervals=srv.sink.arrival_intervals.statistic(),
        num_rx_collided=srv.interfaces[0].receiver.num_collisions,
        num_rx_success=srv.interfaces[0].receiver.num_received,
        num_packets_received=srv.sink.num_packets_received,
        collision_ratio=srv.interfaces[0].receiver.collision_ratio,
    )

    return simret_class(clients=clients, server=server, network=ret.data)


def collision_domain_saturated_network(
        num_clients, payload_size, ack_size, mac_header_size,
        phy_header_size, preamble, bitrate, difs, sifs, slot, cwmin, cwmax,
        queue_capacity=None, connection_radius=100,
        speed_of_light=SPEED_OF_LIGHT, sim_time_limit=1000,
        log_level=Logger.Level.INFO):
    ret = simulate(
        CollisionDomainSaturatedNetwork,
        stime_limit=sim_time_limit,
        params=dict(
            num_stations=(num_clients + 1),
            payload_size=payload_size,
            mac_header_size=mac_header_size,
            phy_header_size=phy_header_size,
            ack_size=ack_size,
            preamble=preamble,
            bitrate=bitrate,
            difs=difs,
            sifs=sifs,
            slot=slot,
            cwmin=cwmin,
            cwmax=cwmax,
            connection_radius=connection_radius,
            speed_of_light=speed_of_light,
            queue_capacity=queue_capacity,
        ), loglevel=log_level
    )

    simret_class = namedtuple('SimRet', ['clients', 'server', 'network'])
    client_class = namedtuple('Client', [
        'service_time', 'num_retries', 'queue_size', 'busy',
        'source_intervals', 'num_packets_sent',
    ])
    server_class = namedtuple('Server', [
        'arrival_intervals', 'num_rx_collided', 'num_rx_success',
        'num_packets_received', 'collision_ratio',
    ])

    clients = [
        client_class(
            service_time=cli.interfaces[0].transmitter.service_time,
            num_retries=cli.interfaces[0].transmitter.num_retries_vector,
            queue_size=cli.interfaces[0].queue.size_trace,
            busy=cli.interfaces[0].transmitter.busy_trace,
            source_intervals=cli.source.arrival_intervals.statistic(),
            num_packets_sent=cli.interfaces[0].transmitter.num_sent,
        ) for cli in ret.data.clients
    ]

    srv = ret.data.server
    server = server_class(
        arrival_intervals=srv.sink.arrival_intervals.statistic(),
        num_rx_collided=srv.interfaces[0].receiver.num_collisions,
        num_rx_success=srv.interfaces[0].receiver.num_received,
        num_packets_received=srv.sink.num_packets_received,
        collision_ratio=srv.interfaces[0].receiver.collision_ratio,
    )

    return simret_class(clients=clients, server=server, network=ret.data)


def wireless_half_duplex_line_network(
        num_clients, payload_size, source_interval, ack_size, mac_header_size,
        phy_header_size, preamble, bitrate, difs, sifs, slot, cwmin, cwmax,
        queue_capacity=None, active_sources=(0,), connection_radius=120,
        distance=100, speed_of_light=SPEED_OF_LIGHT, sim_time_limit=1000,
        log_level=Logger.Level.INFO):
    ret = simulate(
        WirelessHalfDuplexLineNetwork,
        stime_limit=sim_time_limit,
        params=dict(
            num_stations=(num_clients + 1),
            active_sources=active_sources,
            payload_size=payload_size,
            source_interval=source_interval,
            mac_header_size=mac_header_size,
            phy_header_size=phy_header_size,
            ack_size=ack_size,
            preamble=preamble,
            bitrate=bitrate,
            difs=difs,
            sifs=sifs,
            slot=slot,
            cwmin=cwmin,
            cwmax=cwmax,
            connection_radius=connection_radius,
            distance=distance,
            speed_of_light=speed_of_light,
            queue_capacity=queue_capacity,
        ), loglevel=log_level
    )

    simret_class = namedtuple('SimRet', ['clients', 'server', 'network'])
    client_class = namedtuple('Client', [
        'service_time', 'num_retries', 'queue_size', 'tx_busy', 'rx_busy',
        'source_intervals', 'num_packets_sent', 'delay', 'sid',
        'arrival_intervals', 'queue_drop_ratio', 'collision_ratio',
        'queue_wait',
    ])
    server_class = namedtuple('Server', [
        'arrival_intervals', 'num_rx_collided', 'num_rx_success',
        'num_packets_received', 'collision_ratio',
    ])

    # Helper lists and objects:
    _client_sources = [cli.source for cli in ret.data.clients]
    _client_ifaces = [cli.interfaces[0] for cli in ret.data.clients]
    _srv = ret.data.server

    clients = [
        client_class(
            service_time=iface.transmitter.service_time,
            num_retries=iface.transmitter.num_retries_vector,
            queue_size=iface.queue.size_trace,
            tx_busy=iface.transmitter.busy_trace,
            rx_busy=iface.receiver.busy_trace,
            source_intervals=(
                src.arrival_intervals.statistic() if src else None),
            num_packets_sent=iface.transmitter.num_sent,
            delay=(_srv.sink.source_delays.get(src.source_id) if src else None),
            sid=(src.source_id if src else None),
            arrival_intervals=iface.queue.arrival_intervals.statistic(),
            queue_drop_ratio=iface.queue.drop_ratio,
            collision_ratio=iface.receiver.collision_ratio,
            queue_wait=iface.queue.wait_intervals,
        ) for src, iface in zip(_client_sources, _client_ifaces)
    ]
    server = server_class(
        arrival_intervals=_srv.sink.arrival_intervals.statistic(),
        num_rx_collided=_srv.interfaces[0].receiver.num_collisions,
        num_rx_success=_srv.interfaces[0].receiver.num_received,
        num_packets_received=_srv.sink.num_packets_received,
        collision_ratio=_srv.interfaces[0].receiver.collision_ratio,
    )

    return simret_class(clients=clients, server=server, network=ret.data)


def wired_line_network(
        num_clients, payload_size, source_interval, header_size, bitrate,
        preamble=0, ifs=None,  distance=100, queue_capacity=None,
        active_sources=(0,), speed_of_light=SPEED_OF_LIGHT,
        sim_time_limit=1000, log_level=Logger.Level.INFO):

    if ifs is None:
        ifs = 1 / bitrate

    ret = simulate(
        WiredLineNetwork,
        stime_limit=sim_time_limit,
        params=dict(
            num_stations=(num_clients + 1),
            payload_size=payload_size,
            source_interval=source_interval,
            header_size=header_size,
            bitrate=bitrate,
            distance=distance,
            speed_of_light=speed_of_light,
            active_sources=active_sources,
            preamble=preamble,
            ifs=ifs,
            queue_capacity=queue_capacity,
        ),
        loglevel=log_level,
    )

    simret_class = namedtuple('SimRet', ['clients', 'server', 'network'])
    client_class = namedtuple('Client', [
        'service_time', 'queue_size', 'tx_busy', 'rx_busy',
        'source_intervals', 'num_packets_sent', 'delay', 'sid',
        'arrival_intervals', 'queue_drop_ratio', 'queue_wait',
    ])
    server_class = namedtuple('Server', [
        'arrival_intervals', 'num_packets_received',
    ])

    # Helper lists and objects:
    _client_sources = [cli.source for cli in ret.data.clients]
    _client_ifaces = [(cli.interfaces[0], cli.interfaces[-1])
                      for cli in ret.data.clients]
    _srv = ret.data.server

    clients = [
        client_class(
            service_time=out_if.transceiver.service_time,
            queue_size=out_if.queue.size_trace,
            tx_busy=out_if.transceiver.tx_busy_trace,
            rx_busy=inp_if.transceiver.rx_busy_trace,
            source_intervals=(
                src.arrival_intervals.statistic() if src else None),
            num_packets_sent=out_if.transceiver.num_transmitted_packets,
            delay=(_srv.sink.source_delays.get(src.source_id) if src else None),
            sid=(src.source_id if src else None),
            arrival_intervals=out_if.queue.arrival_intervals.statistic(),
            queue_drop_ratio=out_if.queue.drop_ratio,
            queue_wait=out_if.queue.wait_intervals,
        ) for src, (inp_if, out_if) in zip(_client_sources, _client_ifaces)
    ]
    server = server_class(
        arrival_intervals=_srv.sink.arrival_intervals.statistic(),
        num_packets_received=_srv.sink.num_packets_received,
    )

    return simret_class(clients=clients, server=server, network=ret.data)
