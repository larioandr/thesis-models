from math import floor

import pytest
from numpy.testing import assert_allclose
from pydesim import simulate, Logger
from pyqumo.distributions import Constant, Exponential

from pycsmaca.simulations import WiredLineNetwork


SIM_TIME_LIMIT = 1000
PAYLOAD_SIZE = Constant(100.0)      # 100 bits data payload
SOURCE_INTERVAL = Constant(1.0)     # 1 second between packets
HEADER_SIZE = 10                    # 10 bits header
BITRATE = 500                       # 500 bps
DISTANCE = 500                      # 500 meters between stations
SPEED_OF_LIGHT = 10000              # 10 kilometers per second speed of light
IFS = 0.00001
PREAMBLE = 0


def test_two_wire_connected_stations():
    sr = simulate(
        WiredLineNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=2,
            payload_size=PAYLOAD_SIZE,
            source_interval=SOURCE_INTERVAL,
            header_size=HEADER_SIZE,
            bitrate=BITRATE,
            distance=DISTANCE,
            speed_of_light=SPEED_OF_LIGHT,
            active_sources=[0],
            preamble=PREAMBLE,
            ifs=IFS,
        ),
        loglevel=Logger.Level.INFO
    )

    client = sr.data.stations[0]
    server = sr.data.stations[1]

    expected_interval_avg = SOURCE_INTERVAL.mean()
    expected_number_of_packets = floor(SIM_TIME_LIMIT / expected_interval_avg)

    assert client.source.num_packets_sent == expected_number_of_packets
    assert (expected_number_of_packets - 1 <=
            server.sink.num_packets_received <= expected_number_of_packets)

    expected_transmission_delay = (PAYLOAD_SIZE.mean() + HEADER_SIZE) / BITRATE
    expected_delay = DISTANCE / SPEED_OF_LIGHT + expected_transmission_delay

    source_id = client.source.source_id
    assert_allclose(
        server.sink.source_delays[source_id].mean(), expected_delay, rtol=0.1)

    client_if = client.get_interface_to(server)
    assert client_if.queue.size_trace.timeavg() == 0

    expected_busy_ratio = expected_transmission_delay / expected_interval_avg
    assert_allclose(client_if.transceiver.tx_busy_trace.timeavg(),
                    expected_busy_ratio, rtol=0.1)


@pytest.mark.parametrize('num_stations', [3, 4])
def test_wired_line_network_with_single_source(num_stations):
    sr = simulate(
        WiredLineNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=num_stations,
            payload_size=PAYLOAD_SIZE,
            source_interval=SOURCE_INTERVAL,
            header_size=HEADER_SIZE,
            bitrate=BITRATE,
            distance=DISTANCE,
            speed_of_light=SPEED_OF_LIGHT,
            active_sources=[0],
            preamble=PREAMBLE,
            ifs=IFS,
        ),
        loglevel=Logger.Level.ERROR
    )

    client = sr.data.stations[0]
    server = sr.data.stations[-1]
    source_id = client.source.source_id

    expected_interval_avg = SOURCE_INTERVAL.mean()
    expected_number_of_packets = floor(SIM_TIME_LIMIT / expected_interval_avg)

    assert client.source.num_packets_sent == expected_number_of_packets
    assert (expected_number_of_packets - 1 <=
            server.sink.num_packets_received <= expected_number_of_packets)

    expected_transmission_delay = (
            (PAYLOAD_SIZE.mean() + HEADER_SIZE) / BITRATE + PREAMBLE + IFS
    )
    expected_delay = (
            (DISTANCE / SPEED_OF_LIGHT + expected_transmission_delay) *
            (num_stations - 1)
    )

    assert_allclose(
        server.sink.source_delays[source_id].mean(), expected_delay, rtol=0.1)

    client_if = client.get_interface_to(server)
    assert client_if.queue.size_trace.timeavg() == 0

    expected_busy_ratio = expected_transmission_delay / expected_interval_avg
    assert_allclose(client_if.transceiver.tx_busy_trace.timeavg(),
                    expected_busy_ratio, rtol=0.1)


@pytest.mark.parametrize('num_stations', [3, 4])
def test_wired_line_network_with_cross_traffic(num_stations):
    sr = simulate(
        WiredLineNetwork,
        stime_limit=SIM_TIME_LIMIT,
        params=dict(
            num_stations=num_stations,
            payload_size=PAYLOAD_SIZE,
            source_interval=Exponential(SOURCE_INTERVAL.mean()),
            header_size=HEADER_SIZE,
            bitrate=BITRATE,
            distance=DISTANCE,
            speed_of_light=SPEED_OF_LIGHT,
            active_sources=range(num_stations - 1),  # all except last station
            preamble=PREAMBLE,
            ifs=IFS,
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
        rtol=0.1
    )
    assert_allclose(
        server.sink.num_packets_received,
        (num_stations - 1) * expected_number_of_packets,
        rtol=0.1
    )

    expected_transmission_delay = (
            (PAYLOAD_SIZE.mean() + HEADER_SIZE) / BITRATE + PREAMBLE + IFS
    )
    delay_low_bound = (
            (DISTANCE / SPEED_OF_LIGHT + expected_transmission_delay) *
            (num_stations - 1)
    ) * 0.9999

    assert server.sink.source_delays[source_id].mean() >= delay_low_bound

    expected_busy_ratio = expected_transmission_delay / expected_interval_avg
    client_iface = sr.data.get_tx_iface(0)
    assert_allclose(client_iface.transceiver.tx_busy_trace.timeavg(),
                    expected_busy_ratio, rtol=0.1)

    # Here we make sure that out interfaces for all middle stations
    # have non-empty queues since they also generate traffic at almost the same
    # time as they receive packets from connected stations:
    for i in range(1, num_stations - 1):
        sta = sr.data.stations[i]
        next_sta = sr.data.stations[i + 1]
        sta_if = sta.get_interface_to(next_sta)
        assert sta_if.queue.size_trace.timeavg() > 0
        if i > 0:
            sta_inp_if = sta.get_interface_to(sr.data.stations[i - 1])
            tx_busy_rate = sta_if.transceiver.tx_busy_trace.timeavg()
            rx_busy_rate = sta_inp_if.transceiver.rx_busy_trace.timeavg()
            assert_allclose(
                tx_busy_rate, rx_busy_rate + expected_busy_ratio, rtol=0.1
            )
