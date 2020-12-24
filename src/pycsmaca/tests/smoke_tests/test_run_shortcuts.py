import pytest
from numpy.random.mtrand import randint
from numpy.testing import assert_allclose
from pydesim import Logger
from pyqumo.distributions import Exponential

SIM_TIME_LIMIT = 1000
PAYLOAD_SIZE = Exponential(100.0)      # 100 bits data payload in average
INTERVAL_MEAN = 5.0         # 5 second between packets in average
MAC_HEADER = 50             # bits
PHY_HEADER = 25             # bits
PREAMBLE = 1e-3             # seconds
BITRATE = 10000             # 10 kbps
DIFS = 200e-3               # 200 ms
SIFS = 100e-3               # 100 ms
SLOT = 50e-3                # 50 ms
CWMIN = 2
CWMAX = 8
ACK_SIZE = 100              # 100 bits ACK (without PHY header!)
DISTANCE = 500              # 500 meters between stations
CONNECTION_RADIUS = 750     # 750 meters (all stations in circle are connected)
SPEED_OF_LIGHT = 100000     # 10 kilometers per second speed of light
IFS = 50e-3                 # wired interface preamble
FINITE_QUEUE_CAPACITY = 10


@pytest.mark.repeat(5)
def test_collision_domain_network():
    num_clients = randint(1, 10)
    infinite_queue = randint(0, 2)
    queue_capacity = FINITE_QUEUE_CAPACITY if not infinite_queue else None

    from pycsmaca.simulations.shortcuts import collision_domain_network
    sr = collision_domain_network(
        num_clients=num_clients,
        payload_size=PAYLOAD_SIZE,
        source_interval=Exponential(INTERVAL_MEAN),
        ack_size=ACK_SIZE,
        mac_header_size=MAC_HEADER,
        phy_header_size=PHY_HEADER,
        preamble=PREAMBLE,
        bitrate=BITRATE,
        difs=DIFS,
        sifs=SIFS,
        slot=SLOT,
        cwmin=CWMIN,
        cwmax=CWMAX,
        queue_capacity=queue_capacity,
        connection_radius=CONNECTION_RADIUS,
        speed_of_light=SPEED_OF_LIGHT,
        sim_time_limit=SIM_TIME_LIMIT,
        log_level=Logger.Level.WARNING
    )

    for i in range(num_clients):
        cli = sr.clients[i]

        assert cli.service_time.mean() > 0
        assert cli.num_retries.mean() >= 0
        assert cli.queue_size.timeavg() > 0
        assert cli.busy.timeavg() >= 0
        assert cli.num_packets_sent > 0
        assert_allclose(
            cli.source_intervals.mean(),
            INTERVAL_MEAN,
            rtol=0.25
        )

    srv = sr.server

    assert srv.arrival_intervals.mean() > 0
    if num_clients > 1:
        assert srv.num_rx_collided > 0
    else:
        assert srv.num_rx_collided == 0
    assert srv.num_rx_success > 0
    assert_allclose(
        srv.num_packets_received,
        sum(cli.num_packets_sent for cli in sr.clients),
        rtol=0.25
    )


@pytest.mark.repeat(5)
def test_collision_domain_saturated_network():
    num_clients = randint(1, 10)

    from pycsmaca.simulations.shortcuts import \
        collision_domain_saturated_network
    sr = collision_domain_saturated_network(
        num_clients=num_clients,
        payload_size=PAYLOAD_SIZE,
        ack_size=ACK_SIZE,
        mac_header_size=MAC_HEADER,
        phy_header_size=PHY_HEADER,
        preamble=PREAMBLE,
        bitrate=BITRATE,
        difs=DIFS,
        sifs=SIFS,
        slot=SLOT,
        cwmin=CWMIN,
        cwmax=CWMAX,
        connection_radius=CONNECTION_RADIUS,
        speed_of_light=SPEED_OF_LIGHT,
        sim_time_limit=SIM_TIME_LIMIT,
        log_level=Logger.Level.WARNING
    )

    for i in range(num_clients):
        cli = sr.clients[i]

        assert cli.service_time.mean() > 0
        assert cli.num_retries.mean() >= 0
        assert cli.queue_size.timeavg() == 0  # stations don't relay packets!
        assert cli.busy.timeavg() >= 0
        assert cli.num_packets_sent > 0

    srv = sr.server

    assert srv.arrival_intervals.mean() > 0
    if num_clients > 1:
        assert srv.num_rx_collided > 0
    else:
        assert srv.num_rx_collided == 0
    assert srv.num_rx_success > 0
    assert_allclose(
        srv.num_packets_received,
        sum(cli.num_packets_sent for cli in sr.clients),
        rtol=0.25
    )


@pytest.mark.repeat(5)
def test_wireless_half_duplex_line_network():
    num_clients = randint(1, 10)
    active_sources = [0]
    for i in range(1, num_clients):
        if randint(0, 2):
            active_sources.append(i)
    infinite_queue = randint(0, 2)
    queue_capacity = FINITE_QUEUE_CAPACITY if not infinite_queue else None

    from pycsmaca.simulations.shortcuts import wireless_half_duplex_line_network

    sr = wireless_half_duplex_line_network(
        num_clients=num_clients,
        payload_size=PAYLOAD_SIZE,
        source_interval=Exponential(INTERVAL_MEAN),
        active_sources=active_sources,
        ack_size=ACK_SIZE,
        mac_header_size=MAC_HEADER,
        phy_header_size=PHY_HEADER,
        preamble=PREAMBLE,
        bitrate=BITRATE,
        difs=DIFS,
        sifs=SIFS,
        slot=SLOT,
        cwmin=CWMIN,
        cwmax=CWMAX,
        queue_capacity=queue_capacity,
        connection_radius=CONNECTION_RADIUS,
        distance=DISTANCE,
        speed_of_light=SPEED_OF_LIGHT,
        sim_time_limit=SIM_TIME_LIMIT,
        log_level=Logger.Level.WARNING
    )

    for i in range(num_clients):
        cli = sr.clients[i]

        assert cli.service_time.mean() > 0
        assert cli.num_retries.mean() >= 0
        assert cli.queue_size.timeavg() >= 0
        assert cli.rx_busy.timeavg() >= 0
        assert cli.tx_busy.timeavg() >= 0
        assert cli.num_packets_sent > 0

    for i in active_sources:
        sid = sr.clients[i].sid
        assert sid is not None
        assert sr.clients[i].delay.mean() >= 0
        if i > 0:
            assert (sr.clients[i].arrival_intervals.mean() <
                    sr.clients[i].source_intervals.mean())

    srv = sr.server

    assert srv.arrival_intervals.mean() > 0
    assert srv.num_rx_success > 0
    assert_allclose(
        srv.num_packets_received,
        sr.clients[-1].num_packets_sent,
        rtol=0.25
    )


@pytest.mark.repeat(5)
def test_wired_line_network():
    num_clients = randint(1, 10)
    active_sources = [0]
    for i in range(1, num_clients):
        if randint(0, 2):
            active_sources.append(i)
    infinite_queue = randint(0, 2)
    queue_capacity = FINITE_QUEUE_CAPACITY if not infinite_queue else None

    from pycsmaca.simulations.shortcuts import wired_line_network
    sr = wired_line_network(
        num_clients=num_clients,
        payload_size=PAYLOAD_SIZE,
        source_interval=Exponential(INTERVAL_MEAN),
        header_size=(MAC_HEADER + PHY_HEADER),
        bitrate=BITRATE,
        preamble=PREAMBLE,
        ifs=IFS,
        distance=DISTANCE,
        queue_capacity=queue_capacity,
        active_sources=active_sources,
        speed_of_light=SPEED_OF_LIGHT,
        sim_time_limit=SIM_TIME_LIMIT,
        log_level=Logger.Level.INFO
    )

    for i in range(num_clients):
        cli = sr.clients[i]

        assert cli.service_time.mean() > 0
        assert cli.queue_size.timeavg() >= 0
        assert cli.rx_busy.timeavg() >= 0
        assert cli.tx_busy.timeavg() >= 0
        assert cli.num_packets_sent > 0

    for i in active_sources:
        sid = sr.clients[i].sid
        assert sid is not None
        assert sr.clients[i].delay.mean() >= 0
        assert sr.clients[i].source_intervals.mean() >= 0
        if i > 0:
            assert (sr.clients[i].arrival_intervals.mean() < 
                    sr.clients[i].source_intervals.mean())

    srv = sr.server

    assert srv.arrival_intervals.mean() > 0
    assert_allclose(
        srv.num_packets_received,
        sr.clients[-1].num_packets_sent,
        rtol=0.25
    )
