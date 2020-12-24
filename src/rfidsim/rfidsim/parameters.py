import numpy as np

from . import phy
from . import reader
from . import protocol as gen2
from . import pyradise


class ModelDescriptor:
    def __init__(self):
        self.lanes_number = 2
        self.reader_antennas_sides = ['front', 'back']
        self.vehicle_tag_locations = ['front', 'back']
        self.use_doppler = True
        self.thermal_noise = phy.THERMAL_NOISE
        self.permittivity = 15.0
        self.conductivity = 3e-2
        self.reader_antenna_angle = np.pi / 4
        self.lane_width = 3.5
        self.reader_antenna_offset = 1.0
        self.reader_antenna_height = 5.0
        self.reader_antenna_rp = pyradise.dipole_rp
        self.reader_antenna_gain = 8.0
        self.reader_antenna_cable_loss = -1.0
        self.reader_antenna_polarization = 0.5
        self.reader_rounds_per_antenna = 1
        self.reader_frequency = 860e6
        self.reader_tx_power = 31.5
        self.reader_circulator_noise = -80.0
        self.reader_switch_power = True
        self.reader_power_on_interval = 2000e-3
        self.reader_power_off_interval = 100e-3
        self.reader_rounds_per_inventory_flag = 1
        self.reader_session_strategy = reader.Reader.SessionStrategy.ONLY_A
        self.reader_ber_model = pyradise.ber_over_rayleigh
        self.tari = 6.25e-6
        self.data0_multiplier = 2.0
        self.rtcal_multiplier = 2.0
        self.sl = gen2.Sel.SL_ALL
        self.session = gen2.Session.S0
        self.tag_encoding = gen2.TagEncoding.FM0
        self.dr = gen2.DR.DR_8
        self.trext = False
        self.q = 4
        self.vehicle_length = 4
        self.vehicle_speed = 20.0
        self.vehicle_position_update_interval = 1e-2
        self.tag_start_offset = 10.0
        self.tag_height = 0.5
        self.vehicle_direction = (1, 0, 0)
        self.tag_antenna_gain = 2.0
        self.tag_antenna_rp = pyradise.dipole_rp
        self.tag_antenna_polarization = 1.0
        self.tag_modulation_loss = -10.0
        self.tag_sensitivity = -18.0
        self.tag_s1_persistence = 0.5
        self.tag_s2_persistence = 2.0
        self.tag_s3_persistence = 2.0
        self.tag_sl_persistence = 2.0
        self.epc_base_8b = '0000000000000000'
        self.tid_base_4b = 'E00F0000'
        self.vehicle_lifetime = 2.0
        self.vehicle_generation_interval = lambda: np.random.uniform(0.9, 1.1)
        self.max_vehicles_num = 1.0

    def validate(self) -> bool:
        """
        Validate parameters, raises `ValueError` if errors found.

        Returns: True, if no errors found.
        """
        if self.lanes_number <= 0 or self.lanes_number > 2:
            raise ValueError(
                f"expected 1 or 2 lanes, {self.lanes_number} found")
        return True
