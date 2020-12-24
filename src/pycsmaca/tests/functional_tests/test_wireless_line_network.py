from math import floor

import pytest
from numpy.testing import assert_allclose
from pydesim import simulate, Logger
from pyqumo.distributions import Constant, Exponential

from pycsmaca.simulations import WirelessHalfDuplexLineNetwork


SIM_TIME_LIMIT = 10000
PAYLOAD_SIZE = Constant(100.0)      # 100 bits data payload
SOURCE_INTERVAL = Constant(6.0)     # 1 second between packets
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
SPEED_OF_LIGHT = 10000      # 10 kilometers per second speed of light


def test_two_stations_half_duplex_network():
    sr = simulate(
        WirelessHalfDuplexLineNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=2,
            active_sources=[0],
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
            distance=DISTANCE,
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
        ),
        loglevel=Logger.Level.INFO
    )

    client = sr.data.stations[0]
    server = sr.data.stations[1]

    expected_number_of_packets = floor(SIM_TIME_LIMIT / SOURCE_INTERVAL.mean())

    assert client.source.num_packets_sent == expected_number_of_packets
    assert (expected_number_of_packets - 1 <=
            server.sink.num_packets_received <= expected_number_of_packets)

    mean_payload = PAYLOAD_SIZE.mean()
    expected_service_time = (
            DIFS + CWMIN/2 * SLOT
            + PREAMBLE + (mean_payload + MAC_HEADER + PHY_HEADER) / BITRATE
            + SIFS + PREAMBLE + (PHY_HEADER + ACK_SIZE) / BITRATE
            + 2 * DISTANCE / SPEED_OF_LIGHT

    )

    source_id = client.source.source_id
    assert_allclose(
        server.sink.source_delays[source_id].mean(),
        expected_service_time,
        rtol=0.2
    )

    client_if = client.get_interface_to(server)
    assert client_if.queue.size_trace.timeavg() == 0

    assert_allclose(
        client_if.transmitter.busy_trace.timeavg(),
        expected_service_time / SOURCE_INTERVAL.mean(),
        rtol=0.2
    )


@pytest.mark.parametrize('num_stations', [3, 4])
def test_dcf_line_network_with_single_source(num_stations):
    sr = simulate(
        WirelessHalfDuplexLineNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=num_stations,
            active_sources=[0],
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
            distance=DISTANCE,
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
        ),
        loglevel=Logger.Level.INFO
    )

    client = sr.data.stations[0]
    server = sr.data.stations[-1]
    source_id = client.source.source_id

    expected_number_of_packets = floor(SIM_TIME_LIMIT / SOURCE_INTERVAL.mean())

    assert client.source.num_packets_sent == expected_number_of_packets
    assert (expected_number_of_packets - 1 <=
            server.sink.num_packets_received <= expected_number_of_packets)

    mean_payload = PAYLOAD_SIZE.mean()
    expected_service_time = (
            DIFS + CWMIN/2 * SLOT
            + PREAMBLE + (mean_payload + MAC_HEADER + PHY_HEADER) / BITRATE
            + SIFS + PREAMBLE + (PHY_HEADER + ACK_SIZE) / BITRATE
            + 2 * DISTANCE / SPEED_OF_LIGHT

    )
    expected_end_to_end_delay = expected_service_time * (num_stations - 1)

    assert_allclose(
        server.sink.source_delays[source_id].mean(),
        expected_end_to_end_delay,
        rtol=0.2
    )

    client_if = client.get_interface_to(server)
    assert client_if.queue.size_trace.timeavg() == 0

    assert_allclose(
        client_if.transmitter.busy_trace.timeavg(),
        expected_service_time / SOURCE_INTERVAL.mean(),
        rtol=0.2
    )


@pytest.mark.parametrize('num_stations', [3, 4])
def test_wireless_half_duplex_line_network_with_cross_traffic(num_stations):
    sr = simulate(
        WirelessHalfDuplexLineNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=num_stations,
            active_sources=range(num_stations - 1),
            payload_size=PAYLOAD_SIZE,
            source_interval=Exponential(SOURCE_INTERVAL.mean()),
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
            distance=DISTANCE,
            connection_radius=CONNECTION_RADIUS,
            speed_of_light=SPEED_OF_LIGHT,
        ),
        loglevel=Logger.Level.ERROR
    )

    client = sr.data.stations[0]
    server = sr.data.stations[-1]
    source_id = client.source.source_id

    expected_interval_avg = SOURCE_INTERVAL.mean()
    expected_number_of_packets = floor(SIM_TIME_LIMIT / expected_interval_avg)

    assert_allclose(
        client.source.num_packets_sent,
        expected_number_of_packets,
        rtol=0.25
    )

    assert_allclose(
        server.sink.num_packets_received,
        (num_stations - 1) * expected_number_of_packets,
        rtol=0.2
    )

    mean_payload = PAYLOAD_SIZE.mean()
    expected_service_time = (
            DIFS + CWMIN/2 * SLOT
            + PREAMBLE + (mean_payload + MAC_HEADER + PHY_HEADER) / BITRATE
            + SIFS + PREAMBLE + (PHY_HEADER + ACK_SIZE) / BITRATE
            + 2 * DISTANCE / SPEED_OF_LIGHT

    )
    delay_low_bound = expected_service_time * (num_stations - 1) * 0.9999
    assert server.sink.source_delays[source_id].mean() >= delay_low_bound

    expected_busy_ratio = expected_service_time / SOURCE_INTERVAL.mean()
    client_iface = sr.data.get_iface(0)
    assert client_iface.transmitter.busy_trace.timeavg() >= expected_busy_ratio

    # Here we make sure that out interfaces for all middle stations
    # have non-empty queues since they also generate traffic at almost the same
    # time as they receive packets from connected stations:
    for i in range(0, num_stations - 2):
        prev_if = sr.data.get_iface(i)
        next_if = sr.data.get_iface(i + 1)
        assert next_if.queue.size_trace.timeavg() > 0
        if i > 0:
            next_busy_rate = next_if.transmitter.busy_trace.timeavg()
            prev_busy_rate = prev_if.transmitter.busy_trace.timeavg()
            assert_allclose(
                next_busy_rate, prev_busy_rate + expected_busy_ratio, rtol=0.35
            )
