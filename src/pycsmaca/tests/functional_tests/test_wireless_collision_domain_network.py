from math import floor

import pytest
from numpy.random.mtrand import randint, uniform
from numpy.testing import assert_almost_equal, assert_allclose

from pyqumo.distributions import Constant, Exponential
from pydesim import simulate, Logger
from pycsmaca.simulations.wireless_networks import CollisionDomainNetwork, \
    CollisionDomainSaturatedNetwork


# Here we define the parameters those allow us to easily estimate expected
# values manually:
SIM_TIME_LIMIT = 2000
PAYLOAD_SIZE = Constant(100.0)         # 100 bits data payload
SOURCE_INTERVAL = Exponential(1.0)     # 1 second between packets
MAC_HEADER = 50             # bits
PHY_HEADER = 25             # bits
PREAMBLE = 1e-3             # seconds
BITRATE = 1000              # 1000 bps
DIFS = 200e-3               # 200 ms
SIFS = 100e-3               # 100 ms
SLOT = 50e-3                # 50 ms
CWMIN = 2
CWMAX = 8
ACK_SIZE = 100              # 100 bits ACK (without PHY header!)
DISTANCE = 500              # 500 meters between stations
CONNECTION_RADIUS = 750     # 750 meters (all stations in circle are connected)
SPEED_OF_LIGHT = 100000     # 10 kilometers per second speed of light


def test_network_without_collisions():
    # First, we run the simulation:
    sr = simulate(
        CollisionDomainNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=2,
            payload_size=PAYLOAD_SIZE,
            source_interval=SOURCE_INTERVAL,
            mac_header_size=MAC_HEADER,
            phy_header_size=PHY_HEADER,
            ack_size=ACK_SIZE,
            preamble=PREAMBLE,
            bitrate=BITRATE,
            difs=DIFS,
            sifs=SIFS,
            slot=SLOT,
            cwmin=CWMIN,
            cwmax=CWMAX,
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
            queue_capacity=None,
        ),
        loglevel=Logger.Level.WARNING
    )
    assert sr.stime >= SIM_TIME_LIMIT

    # Since all stations send data to station 0, we assign these stations
    # to more valuable variables:
    access_point = sr.data.stations[0]
    access_point_iface = access_point.interfaces[0]
    client = sr.data.stations[1]
    client_iface = client.interfaces[0]

    # Then we validate that no collisions were registered:
    assert client_iface.receiver.num_collisions == 0
    assert access_point_iface.receiver.num_collisions == 0

    # We also check that average backoff is equal to 0.5 * (CWMIN - 1):
    assert_almost_equal(
        client.interfaces[0].transmitter.backoff_vector.mean(),
        0.5 * (CWMIN - 1),
        decimal=1
    )

    # Now we validate that average service time is equal to:
    #   DIFS + SLOT * E[backoff] + delta + PKT + SIFS + delta + ACK,
    # where PKT = (PAYLOAD + MAC_HEADER + PHY_HEADER) / BITRATE + PREAMBLE,
    #       delta = 2 * RADIUS / SPEED_OF_LIGHT (propagation delay):
    packet_duration = (
            (PAYLOAD_SIZE.mean() + MAC_HEADER + PHY_HEADER) / BITRATE +
            PREAMBLE
    )
    propagation = CONNECTION_RADIUS / SPEED_OF_LIGHT
    mean_backoff = client_iface.transmitter.backoff_vector.mean()
    ack_duration = (ACK_SIZE + PHY_HEADER) / BITRATE + PREAMBLE
    expected_service_time = (
        DIFS + SLOT * mean_backoff + packet_duration + SIFS +
        2 * propagation + ack_duration
    )
    assert_allclose(
        client_iface.transmitter.service_time.mean(),
        expected_service_time,
        rtol=0.1
    )

    # Check that the number of received packets is proportional to arrival rate:
    expected_num_packets = int(floor(sr.stime / SOURCE_INTERVAL.mean()))

    assert (client_iface.transmitter.num_sent * 0.99 <=
            access_point_iface.receiver.num_received <=
            client_iface.transmitter.num_sent)
    assert_allclose(
        client_iface.transmitter.num_sent + client_iface.queue.size(),
        expected_num_packets,
        rtol=0.25)

    # To make sure queue is not overflowing:
    assert 0 < client_iface.queue.size_trace.timeavg() < 20

    # At the end, check that the average packet size is equal to payload:
    assert access_point.sink.data_size_stat.mean() == PAYLOAD_SIZE.mean()
    assert client.source.data_size_stat.mean() == PAYLOAD_SIZE.mean()

    # ... and that interval between generations is as specified by intervals
    # distribution parameter:
    arrival_stats = client.source.arrival_intervals.statistic()
    assert_allclose(
        arrival_stats.mean(),
        sr.params.source_interval.mean(),
        rtol=0.1
    )
    assert_allclose(
        arrival_stats.std(),
        sr.params.source_interval.std(),
        rtol=0.25
    )


@pytest.mark.repeat(5)
def test_large_collision_domain_network__smoke():
    """In this test we validate that all stations are really in a single
    domain and run the model for some time. We actually do not test any
    meaningful properties, except connections and that only server receives
    data.
    """
    num_stations = randint(5, 15)
    source_interval = Exponential(uniform(1.0, 10.0))
    payload_size = Exponential(randint(10, 100))

    sr = simulate(
        CollisionDomainNetwork,
        stime_limit=500,
        params=dict(
            num_stations=num_stations,
            payload_size=payload_size,
            source_interval=source_interval,
            mac_header_size=MAC_HEADER,
            phy_header_size=PHY_HEADER,
            ack_size=ACK_SIZE,
            preamble=PREAMBLE,
            bitrate=BITRATE,
            difs=DIFS,
            sifs=SIFS,
            slot=SLOT,
            cwmin=CWMIN,
            cwmax=CWMAX,
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
            queue_capacity=None,
        ),
        loglevel=Logger.Level.WARNING
    )

    access_point = sr.data.stations[0]
    clients = sr.data.stations[1:]
    conn_man = sr.data.connection_manager

    # Test that connections are established between all stations:
    for i in range(num_stations):
        radio = sr.data.get_iface(i).radio
        peers = set(conn_man.get_peers(radio))
        assert len(peers) == num_stations - 1 and radio not in peers

    # Test that the number of packets received by any client sink is 0:
    for client in clients:
        assert client.sink.num_packets_received == 0

    # Test that the number of packets generated by the sources - (queue sizes
    # + number of packets in transceivers) at the end of simulation is
    # almost equal to the number of received packets by the access point sink:
    num_packets_sent = [
        (cli.source.num_packets_sent -
         cli.interfaces[0].queue.size() -
         (1 if cli.interfaces[0].transmitter.state else 0))
        for cli in clients
    ]
    num_packets_received = access_point.sink.num_packets_received
    assert_allclose(sum(num_packets_sent), num_packets_received, rtol=0.05)


def test_saturated_network_without_collisions():
    # First, we run the simulation:
    sr = simulate(
        CollisionDomainSaturatedNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=2,
            payload_size=PAYLOAD_SIZE,
            mac_header_size=MAC_HEADER,
            phy_header_size=PHY_HEADER,
            ack_size=ACK_SIZE,
            preamble=PREAMBLE,
            bitrate=BITRATE,
            difs=DIFS,
            sifs=SIFS,
            slot=SLOT,
            cwmin=CWMIN,
            cwmax=CWMAX,
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
            queue_capacity=None,
        ),
        loglevel=Logger.Level.WARNING
    )
    assert sr.stime >= SIM_TIME_LIMIT

    # Since all stations send data to station 0, we assign these stations
    # to more valuable variables:
    access_point = sr.data.stations[0]
    access_point_iface = access_point.interfaces[0]
    client = sr.data.stations[1]
    client_iface = client.interfaces[0]

    # Then we validate that no collisions were registered:
    assert client_iface.receiver.num_collisions == 0
    assert access_point_iface.receiver.num_collisions == 0

    # We also check that average backoff is equal to 0.5 * (CWMIN - 1):
    assert_almost_equal(
        client.interfaces[0].transmitter.backoff_vector.mean(),
        0.5 * (CWMIN - 1),
        decimal=1
    )

    # Now we validate that average service time is equal to:
    #   DIFS + SLOT * E[backoff] + delta + PKT + SIFS + delta + ACK,
    # where PKT = (PAYLOAD + MAC_HEADER + PHY_HEADER) / BITRATE + PREAMBLE,
    #       delta = 2 * RADIUS / SPEED_OF_LIGHT (propagation delay):
    packet_duration = (
            (PAYLOAD_SIZE.mean() + MAC_HEADER + PHY_HEADER) / BITRATE +
            PREAMBLE
    )
    propagation = CONNECTION_RADIUS / SPEED_OF_LIGHT
    mean_backoff = client_iface.transmitter.backoff_vector.mean()
    ack_duration = (ACK_SIZE + PHY_HEADER) / BITRATE + PREAMBLE
    expected_service_time = (
        DIFS + SLOT * mean_backoff + packet_duration + SIFS +
        2 * propagation + ack_duration
    )
    assert_allclose(
        client_iface.transmitter.service_time.mean(),
        expected_service_time,
        rtol=0.1
    )

    # Check that the number of received packets is proportional to arrival rate:
    expected_num_packets = int(floor(sr.stime / expected_service_time))

    assert (client_iface.transmitter.num_sent * 0.99 <=
            access_point_iface.receiver.num_received <=
            client_iface.transmitter.num_sent)
    assert_allclose(
        client_iface.transmitter.num_sent + client_iface.queue.size(),
        expected_num_packets,
        rtol=0.25)

    # At the end, check that the average packet size is equal to payload:
    assert access_point.sink.data_size_stat.mean() == PAYLOAD_SIZE.mean()
    assert client.source.data_size_stat.mean() == PAYLOAD_SIZE.mean()

    # ... and that interval between generations is like service time:
    arrival_stats = client.source.arrival_intervals.statistic()
    assert_allclose(
        arrival_stats.mean(),
        expected_service_time,
        rtol=0.1
    )


def test_saturated_network_with_three_stations():
    sr = simulate(
        CollisionDomainSaturatedNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=3,
            payload_size=PAYLOAD_SIZE,
            mac_header_size=MAC_HEADER,
            phy_header_size=PHY_HEADER,
            ack_size=ACK_SIZE,
            preamble=PREAMBLE,
            bitrate=BITRATE,
            difs=DIFS,
            sifs=SIFS,
            slot=SLOT,
            cwmin=CWMIN,
            cwmax=CWMIN,  # the same here, no increase to calculate p_collision
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
            queue_capacity=None,
        ),
        loglevel=Logger.Level.WARNING
    )

    # Since all stations send data to station 0, we assign these stations
    # to more valuable variables:
    access_point = sr.data.stations[0]
    access_point_iface = access_point.interfaces[0]

    # We validate the collisions probabilities:
    p_collision = access_point_iface.receiver.num_collisions / (
            access_point_iface.receiver.num_collisions +
            access_point_iface.receiver.num_received
    )
    assert_allclose(p_collision, 1 / CWMIN, rtol=0.1)
