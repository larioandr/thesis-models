from math import pi

from numpy.testing import assert_allclose

import pytest

from rfidsim.parameters import ModelDescriptor
from rfidsim.factory import Factory
from rfidsim.pyradise import isotropic_rp


@pytest.mark.parametrize(
    'num_lanes, lane, side, angle, width, offset, height, forward, pos', [
        (1, 0, 'front', pi/3, 7, 2, 13, (-0.866, 0, -0.5), (-2, 3.5, 13)),
        (1, 0, 'back', pi/3, 7, 2, 13, (0.866, 0, -0.5), (2, 3.5, 13)),
        (2, 0, 'front', pi/6, 10, 3, 19, (-0.5, 0, -0.866), (-3, -5, 19)),
        (2, 0, 'back', pi/6, 10, 3, 19, (0.5, 0, -0.866), (3, -5, 19)),
        (2, 1, 'front', pi/4, 10, 3, 19, (-0.707, 0, -0.707), (-3, 5, 19)),
        (2, 1, 'back', pi/4, 10, 3, 19, (0.707, 0, -0.707), (3, 5, 19)),
    ]
)
def test_build_reader_antenna__geometry(
        num_lanes, lane, side, angle, width, offset, height, forward, pos):
    """
    Validate that forward direction, right direction and position of the
    reader antenna are computed and set properly.
    """
    # Define constant values for expected right_dir values:
    right = {
        'front': (0, -1, 0),
        'back': (0, 1, 0),
    }[side]
    md = ModelDescriptor()
    md.reader_antenna_angle = angle
    md.lanes_number = num_lanes
    md.lane_width = width
    md.reader_antenna_offset = offset
    md.reader_antenna_height = height

    # Create ReaderAntenna:
    factory = Factory()
    factory.params = md
    ant = factory.build_reader_antenna(lane, side)

    # Validate geometry properties:
    assert ant.lane == lane
    assert ant.side == side
    assert_allclose(ant.dir_forward, forward, rtol=0.01)
    assert_allclose(ant.dir_right, right, rtol=0.01)
    assert_allclose(ant.position, pos, rtol=0.01)


def test_build_reader_antenna__attributes():
    """Validate that settings from the model descriptor come into antenna.
    """
    md = ModelDescriptor()
    md.reader_antenna_rp = isotropic_rp
    md.reader_antenna_gain = 13.0
    md.reader_antenna_cable_loss = -2.0
    md.reader_antenna_polarization = 1.0

    factory = Factory()
    factory.params = md
    ant = factory.build_reader_antenna(0, 'front')

    assert ant.rp == md.reader_antenna_rp
    assert ant.gain == md.reader_antenna_gain
    assert ant.cable_loss == md.reader_antenna_cable_loss
    assert ant.polarization == md.reader_antenna_polarization
