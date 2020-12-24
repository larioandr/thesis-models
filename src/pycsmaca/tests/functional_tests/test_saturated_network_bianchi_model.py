import pytest
from numpy.testing import assert_almost_equal

from pycsmaca.analytic import bianchi_time
from pyqumo.distributions import Constant

# Here we define the parameters those allow us to easily estimate expected
# values manually:
PAYLOAD = 1000  # bits
ACK = 100  # bits
MAC_HEADER = 50  # bits
PHY_HEADER = 25  # bits
PREAMBLE = 1e-3  # seconds
BITRATE = 1e3  # bits per second
DIFS = 200e-3  # 200 ms
SIFS = 100e-3  # 100 ms
SLOT = 50e-3  # 50 ms
CWMIN = 2
CWMAX = 8
SPEED_OF_LIGHT = 1e5   # meters per second


def test_bianchi_time_without_collisions():
    radius = 100

    # First, we run the simulation:
    ret = bianchi_time(
        num_clients=1, payload_size=Constant(PAYLOAD), ack_size=ACK,
        mac_header_size=MAC_HEADER, phy_header_size=PHY_HEADER,
        preamble=PREAMBLE, bitrate=BITRATE, difs=DIFS, sifs=SIFS,
        slot=SLOT, cwmin=CWMIN, cwmax=CWMAX, distance=radius,
        c=SPEED_OF_LIGHT
    )

    # Now we validate that average service time is equal to:
    #   DIFS + SLOT * E[backoff] + delta + PKT + SIFS + delta + ACK,
    # where PKT = (PAYLOAD + MAC_HEADER + PHY_HEADER) / BITRATE + PREAMBLE,
    #       delta = 2 * RADIUS / SPEED_OF_LIGHT (propagation delay):
    packet_duration = (PAYLOAD + MAC_HEADER + PHY_HEADER) / BITRATE + PREAMBLE
    propagation = 2 * radius / SPEED_OF_LIGHT
    mean_backoff = (CWMIN - 1) * 0.5
    ack_duration = (ACK + PHY_HEADER) / BITRATE + PREAMBLE
    expected_service_time = (
        DIFS + SLOT * mean_backoff + packet_duration + SIFS +
        2 * propagation + ack_duration,
    )

    assert_almost_equal(ret.mean, expected_service_time, decimal=1)
