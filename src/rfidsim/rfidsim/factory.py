import itertools
import numpy as np

import pyons
from parameters import ModelDescriptor
import phy
import reader
import tag
import vehicles


class Factory(object, metaclass=pyons.Singleton):
    def __init__(self, model_descriptor=None):
        assert model_descriptor is None or \
               isinstance(model_descriptor, ModelDescriptor)
        self._params = model_descriptor
        self.tag_id = itertools.count(1)
        self.vehicle_id = itertools.count(1)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        assert value is None or isinstance(value, ModelDescriptor)
        self._params = value

    def build_reader_antenna(self, lane, side):
        if self.params.lanes_number <= 0 or self.params.lanes_number > 2:
            raise RuntimeError("only 1 or 2 lanes supported, {} found".format(
                self.params.lanes_number))

        if lane < 0 or lane >= self.params.lanes_number:
            raise RuntimeError("undefined lane={}".format(lane))

        if side not in ['front', 'back']:
            raise RuntimeError("undefined side={}".format(side))

        x = self.params.reader_antenna_offset
        y = self.params.lane_width / 2
        z = self.params.reader_antenna_height
        a = self.params.reader_antenna_angle

        position = None
        forward_dir = None
        right_dir = None
        if side == 'front':
            forward_dir = (-np.sin(a), 0, -np.cos(a))
            right_dir = (0, -1, 0)
            if lane == 0 and self.params.lanes_number == 2:
                position = (-x, -y, z)
            else:
                position = (-x, y, z)
        elif side == 'back':
            forward_dir = (np.sin(a), 0, -np.cos(a))
            right_dir = (0, 1, 0)
            if lane == 0 and self.params.lanes_number == 2:
                position = (x, -y, z)
            else:
                position = (x, y, z)
        antenna = phy.ReaderAntenna()
        antenna.position = position
        antenna.dir_forward = forward_dir
        antenna.dir_right = right_dir
        antenna.rp = self.params.reader_antenna_rp
        antenna.gain = self.params.reader_antenna_gain
        antenna.cable_loss = self.params.reader_antenna_cable_loss
        antenna.polarization = self.params.reader_antenna_polarization
        antenna.node = None
        antenna.lane = lane
        antenna.side = side
        return antenna

    def build_reader(self, channel):
        rd = reader.ReaderDescriptor()
        rd.antennas = []
        for side in self.params.reader_antennas_sides:
            for lane in range(0, self.params.lanes_number):
                rd.antennas.append(self.build_reader_antenna(lane, side))
        rd.rounds_per_antenna = self.params.reader_rounds_per_antenna
        rd.tari = self.params.tari
        rd.rtcal = None
        rd.trcal = None
        rd.data0_multiplier = self.params.data0_multiplier
        rd.rtcal_multiplier = self.params.rtcal_multiplier
        rd.tx_power = self.params.reader_tx_power
        rd.circulator_noise = self.params.reader_circulator_noise
        rd.switch_power = self.params.reader_switch_power
        rd.power_on_interval = self.params.reader_power_on_interval
        rd.power_off_interval = self.params.reader_power_off_interval
        rd.sl = self.params.sl
        rd.session = self.params.session
        rd.session_strategy = self.params.reader_session_strategy
        rd.rounds_per_inventory_flag = \
            self.params.reader_rounds_per_inventory_flag
        rd.frequency = self.params.reader_frequency
        rd.modulation_loss = 0.0
        rd.tag_encoding = self.params.tag_encoding
        rd.dr = self.params.dr
        rd.trext = self.params.trext
        rd.q = self.params.q
        rd.channel = channel
        rd.ber_model = self.params.reader_ber_model
        return reader.Reader(rd)

    def build_channel(self):
        cd = phy.ChannelDescriptor()
        cd.thermal_noise = self.params.thermal_noise
        cd.ground_conductivity = self.params.conductivity
        cd.ground_permittivity = self.params.permittivity
        cd.use_doppler = self.params.use_doppler
        return phy.Channel(cd)

    def build_tag(self, vehicle_index, location, lane, channel):
        if self.params.lanes_number <= 0 or self.params.lanes_number > 2:
            raise RuntimeError("only 1 or 2 lanes supported, {} found".format(
                self.params.lanes_number))

        if lane < 0 or lane >= self.params.lanes_number:
            raise RuntimeError("undefined lane={}".format(lane))

        if location not in ['front', 'back']:
            raise RuntimeError("undefined side={}".format(location))

        x = -self.params.tag_start_offset
        y = self.params.lane_width / 2
        z = self.params.tag_height

        if lane < self.params.lanes_number - 1:
            y *= -1
        if location == 'front':
            x += self.params.vehicle_length
            orientation = self.params.vehicle_direction
        else:
            orientation = -1.0 * np.array(self.params.vehicle_direction)

        tag_id = next(self.tag_id)
        epc = self.params.epc_base_8b + "{:08X}".format(tag_id)
        tid = self.params.tid_base_4b + "{:08X}".format(tag_id)

        td = tag.TagDescriptor()
        td.identifier = "Tag{}{}".format(vehicle_index, location[0].upper())
        td.channel = channel
        td.position = (x, y, z)
        td.direction = self.params.vehicle_direction
        td.speed = self.params.vehicle_speed
        td.orientation = orientation
        td.up_direction = (0, 0, 1)
        td.gain = self.params.tag_antenna_gain
        td.rp = self.params.tag_antenna_rp
        td.polarization = self.params.tag_antenna_polarization
        td.sensitivity = self.params.tag_sensitivity
        td.modulation_loss = self.params.tag_modulation_loss
        td.s1_persistence = self.params.tag_s1_persistence
        td.s2_persistence = self.params.tag_s2_persistence
        td.s3_persistence = self.params.tag_s3_persistence
        td.sl_persistence = self.params.tag_sl_persistence
        td.epc = epc
        td.tid = tid
        td.location = location
        return tag.Tag(td)

    def build_vehicle(self, lane, channel):
        vehicle_index = next(self.vehicle_id)
        vehicle = vehicles.Vehicle()
        vehicle.vehicle_id = vehicle_index
        vehicle.lifetime = self.params.vehicle_lifetime
        vehicle.speed = self.params.vehicle_speed
        vehicle.length = self.params.vehicle_length
        vehicle.lane = lane
        if 'front' in self.params.vehicle_tag_locations:
            vehicle.front_tag = self.build_tag(
                vehicle_index, 'front', lane, channel)
            vehicle.front_tag.vehicle = vehicle
        if 'back' in self.params.vehicle_tag_locations:
            vehicle.back_tag = self.build_tag(
                vehicle_index, 'back', lane, channel)
            vehicle.back_tag.vehicle = vehicle
        return vehicle
