import pandas as pd

import pyons
from pyons.base import Singleton
import pyradise


class VehicleInfoRecord(object):
    def __init__(self):
        self.vehicle_id = None
        self.created_at = None
        self.destroyed_at = None
        self.direction = None
        self.lane = None
        self.front_tag_epc = None
        self.front_tag_tid = None
        self.back_tag_epc = None
        self.back_tag_tid = None
        self.speed = None
        self.n_read = 0

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            vehicle_id=self.vehicle_id,
            created_at=self.created_at,
            destroyed_at=self.destroyed_at,
            dir_x=get_vector_item(self.direction, 0),
            dir_y=get_vector_item(self.direction, 1),
            dir_z=get_vector_item(self.direction, 2),
            lane=self.lane,
            front_tag_epc=self.front_tag_epc,
            front_tag_tid=self.front_tag_tid,
            back_tag_epc=self.back_tag_epc,
            back_tag_tid=self.back_tag_tid,
            speed=self.speed,
            n_read=self.n_read)


class TagInfoRecord(object):
    def __init__(self):
        self.epc = None
        self.tid = None
        self.vehicle_id = None
        self.location = None
        self.created_at = None
        self.destroyed_at = None
        self.antenna_forward_dir = None
        self.antenna_right_dir = None
        self.antenna_gain = None
        self.antenna_polarization = None
        self.antenna_rp = None
        self.modulation_loss = None
        self.sensitivity = None
        self.lane = None
        self.n_epc_read = 0
        self.n_tid_read = 0
        self.n_rounds = 0       # the number of attempts to read the tag

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            epc=self.epc,
            tid=self.tid,
            vehicle_id=self.vehicle_id,
            location=self.location,
            created_at=self.created_at,
            destroyed_at=self.destroyed_at,
            fwd_x=get_vector_item(self.antenna_forward_dir, 0),
            fwd_y=get_vector_item(self.antenna_forward_dir, 1),
            fwd_z=get_vector_item(self.antenna_forward_dir, 2),
            right_x=get_vector_item(self.antenna_right_dir, 0),
            right_y=get_vector_item(self.antenna_right_dir, 1),
            right_z=get_vector_item(self.antenna_right_dir, 2),
            gain=self.antenna_gain,
            polarization=self.antenna_polarization,
            rp=get_rp_name(self.antenna_rp),
            modulation_loss=self.modulation_loss,
            lane=self.lane,
            n_epc_read=self.n_epc_read,
            n_tid_read=self.n_tid_read,
            n_rounds=self.n_rounds
        )


class ReaderAntennaInfoRecord(object):
    def __init__(self):
        self.index = None
        self.side = None
        self.lane = None
        self.position = None
        self.forward_dir = None
        self.right_dir = None
        self.gain = None
        self.rp = None
        self.polarization = None
        self.cable_loss = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            side=self.side,
            lane=self.lane,
            x=get_vector_item(self.position, 0),
            y=get_vector_item(self.position, 1),
            z=get_vector_item(self.position, 2),
            fwd_x=get_vector_item(self.forward_dir, 0),
            fwd_y=get_vector_item(self.forward_dir, 1),
            fwd_z=get_vector_item(self.forward_dir, 2),
            right_x=get_vector_item(self.right_dir, 0),
            right_y=get_vector_item(self.right_dir, 1),
            right_z=get_vector_item(self.right_dir, 2),
            gain=self.gain,
            rp=get_rp_name(self.rp),
            polarization=self.polarization,
            cable_loss=self.cable_loss
        )


class ReaderInfoRecord(object):
    def __init__(self):
        self.n_antennas = None
        self.tx_power = None
        self.circulator_noise = None
        self.power_switch_enabled = None
        self.session_switch_enabled = None
        self.power_on_duration = None
        self.power_off_duration = None
        self.n_rounds_per_antenna = None
        self.n_rounds_per_session = None
        self.n_slots = None
        self.q = None
        self.m = None
        self.trext = None
        self.dr = None
        self.blf = None
        self.data0 = None
        self.data1 = None
        self.rtcal = None
        self.trcal = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            n_antennas=self.n_antennas,
            tx_power=self.tx_power,
            circulator_noise=self.circulator_noise,
            power_switch_enabled=self.power_switch_enabled,
            session_switch_enabled=self.session_switch_enabled,
            power_on_duration=self.power_on_duration,
            power_off_duration=self.power_off_duration,
            n_rounds_per_antenna=self.n_rounds_per_antenna,
            n_rounds_per_session=self.n_rounds_per_session,
            n_slots=self.n_slots,
            q=self.q, m=get_enum_name(self.m), trext=self.trext,
            dr=get_enum_name(self.dr), blf=self.blf,
            data0=self.data0, data1=self.data1, rtcal=self.rtcal,
            trcal=self.trcal)


class ChannelInfoRecord(object):
    def __init__(self):
        super().__init__()
        self.thermal_noise = None
        self.permittivity = None
        self.conductivity = None
        self.use_doppler = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            thermal_noise=self.thermal_noise,
            permittivity=self.permittivity,
            conductivity=self.conductivity,
            use_doppler=self.use_doppler)


class ChannelStateRecord(object):
    def __init__(self):
        self.timestamp = None
        self.reader_position = None
        self.reader_side = None
        self.reader_lane = None
        self.tag_position = None
        self.tag_location = None
        self.tag_lane = None
        self.tag_speed = None
        self.channel_lifetime = None
        self.tag_rx_power = None
        self.reader_rx_power = None
        self.reader_snr = None
        self.reader_ber = None
        self.rt_path_loss = None
        self.tr_path_loss = None
        self.vehicle_id = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(timestamp=self.timestamp,
                    reader_x=get_vector_item(self.reader_position, 0),
                    reader_y=get_vector_item(self.reader_position, 1),
                    reader_z=get_vector_item(self.reader_position, 2),
                    reader_side=self.reader_side,
                    reader_lane=self.reader_lane,
                    channel_time=self.channel_lifetime,
                    tag_x=get_vector_item(self.tag_position, 0),
                    tag_y=get_vector_item(self.tag_position, 1),
                    tag_z=get_vector_item(self.tag_position, 2),
                    tag_lane=self.tag_lane,
                    tag_loc=self.tag_location,
                    tag_speed=self.tag_speed,
                    tag_rx_power=self.tag_rx_power,
                    reader_rx_power=self.reader_rx_power,
                    reader_snr=self.reader_snr,
                    reader_ber=self.reader_ber,
                    rt_path_loss=self.rt_path_loss,
                    tr_path_loss=self.tr_path_loss,
                    vehicle_id=self.vehicle_id)


class TagReadRecord(object):
    def __init__(self):
        self.round_index = None
        self.slot_index = None
        self.slot_timestamp = None
        self.epc = None
        self.tid = None
        self.vehicle_lane = None
        self.location = None
        self.reader_antenna_lane = None
        self.reader_antenna_side = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            round_index=self.round_index,
            slot_index=self.slot_index,
            epc=self.epc,
            tid=self.tid,
            vehicle_lane=self.vehicle_lane,
            location=self.location,
            reader_antenna_lane=self.reader_antenna_lane,
            reader_antenna_side=self.reader_antenna_side
        )


class InventoryRoundRecord(object):
    def __init__(self):
        self.index = None
        self.n_collisions = None
        self.n_errors = None
        self.n_epc_only_reads = None
        self.n_tid_reads = None
        self.n_empty_slots = None
        self.n_tags = None
        self.n_vehicles_registered = None
        self.antenna_side = None
        self.antenna_lane = None
        self.antenna_index = None
        self.session = None
        self.duration = None
        self.min_slot_duration = None
        self.max_slot_duration = None
        self.avg_slot_duration = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            index=self.index,
            n_collisions=self.n_collisions,
            n_errors=self.n_errors,
            n_epc_only_reads=self.n_epc_only_reads,
            n_tid_reads=self.n_tid_reads,
            n_empty_slots=self.n_empty_slots,
            n_tags=self.n_tags,
            antenna_side=self.antenna_side,
            antenna_lane=self.antenna_lane,
            antenna_index=self.antenna_index,
            session=get_enum_name(self.session),
            duration=self.duration,
            min_slot_duration=self.min_slot_duration,
            max_slot_duration=self.max_slot_duration,
            avg_slot_duration=self.avg_slot_duration,
            n_vehicles_registered=self.n_vehicles_registered)


class FrameBERRecord(object):
    def __init__(self):
        self.reader_lane = None
        self.reader_side = None
        self.reader_position = None
        self.tag_lane = None
        self.tag_side = None
        self.tag_position = None
        self.frame_bitlen = None
        self.ber = None
        self.probability = None
        self.result = None

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(
            reader_lane=self.reader_lane,
            reader_side=self.reader_side,
            reader_x=get_vector_item(self.reader_position, 0),
            reader_y=get_vector_item(self.reader_position, 1),
            reader_z=get_vector_item(self.reader_position, 2),
            tag_lane=self.tag_lane,
            tag_side=self.tag_side,
            tag_x=get_vector_item(self.tag_position, 0),
            tag_y=get_vector_item(self.tag_position, 1),
            tag_z=get_vector_item(self.tag_position, 2),
            frame_bitlen=self.frame_bitlen,
            ber=self.ber,
            probability=self.probability,
            result=self.result
        )


class Journal(metaclass=Singleton):
    def __init__(self):
        super().__init__()

        self.vehicle_info = {}         # vehicle_id -> VehicleInfoRecord
        self.tag_info = {}             # epc -> TagInfoRecord
        self.reader_antenna_info = {}
        self.reader_info = None
        self.channel_info = None
        self.channel_state_journal = []  # list of PowerRecord
        self.tag_read_journal = []
        self.inventory_round_journal = []
        self.frame_ber_journal = []
        self.n_skip_vehicles = 3

        self.channel_state_logging_enabled = True
        self.tag_read_logging_enabled = True
        self.inventory_round_logging_enabled = True
        self.frame_ber_logging_enabled = True

    def clear(self):
        self.vehicle_info.clear()
        self.tag_info.clear()
        self.reader_antenna_info.clear()
        self.reader_info.clear()
        self.channel_journal.clear()
        self.tag_read_journal.clear()
        self.inventory_round_journal.clear()

    def write_vehicle_created(self, vehicle_info_record):
        assert isinstance(vehicle_info_record, VehicleInfoRecord)
        assert vehicle_info_record.vehicle_id not in self.vehicle_info
        self.vehicle_info[vehicle_info_record.vehicle_id] = vehicle_info_record

    def write_vehicle_destroyed(self, vehicle_id, timestamp):
        assert vehicle_id in self.vehicle_info
        self.vehicle_info[vehicle_id].destroyed_at = timestamp

    def write_tag_created(self, tag_info_record):
        assert isinstance(tag_info_record, TagInfoRecord)
        assert tag_info_record.epc not in self.tag_info
        self.tag_info[tag_info_record.epc] = tag_info_record

    def write_tag_destroyed(self, tag_epc, timestamp):
        assert tag_epc in self.tag_info
        self.tag_info[tag_epc].destroyed_at = timestamp

    def write_reader_antenna_info(self, reader_antenna_info_record):
        assert isinstance(reader_antenna_info_record, ReaderAntennaInfoRecord)
        index = reader_antenna_info_record.index
        assert index not in self.reader_antenna_info
        self.reader_antenna_info[index] = reader_antenna_info_record

    def write_reader_info(self, reader_info):
        assert isinstance(reader_info, ReaderInfoRecord)
        self.reader_info = reader_info

    def write_channel_info(self, channel_info):
        assert isinstance(channel_info, ChannelInfoRecord)
        self.channel_info = channel_info

    def write_channel_state(self, channel_state):
        assert isinstance(channel_state, ChannelStateRecord)
        if self.channel_state_logging_enabled:
            self.channel_state_journal.append(channel_state)

    def write_tag_read(self, tag_read_record):
        assert isinstance(tag_read_record, TagReadRecord)
        if tag_read_record.epc is not None:
            assert tag_read_record.epc in self.tag_info
            tag_info_record = self.tag_info[tag_read_record.epc]
            assert tag_info_record is not None
            tag_info_record.n_epc_read += 1
            if tag_read_record.tid is not None:
                tag_info_record.n_tid_read += 1
                # !!!!! READING VEHICLE BY TAG ONLY !!!!!
                vehicle_id = tag_info_record.vehicle_id
                if vehicle_id is not None:
                    assert vehicle_id in self.vehicle_info, \
                        f"missing vehicle {vehicle_id} in vehicle_info = " \
                        f"{self.vehicle_info}, journal = {id(self)}"
                    vehicle_info_record = self.vehicle_info[vehicle_id]
                    assert vehicle_info_record is not None
                    vehicle_info_record.n_read += 1
        if self.tag_read_logging_enabled:
            self.tag_read_journal.append(tag_read_record)

    def write_inventory_round(self, inventory_round):
        assert isinstance(inventory_round, InventoryRoundRecord)
        if self.inventory_round_logging_enabled:
            self.inventory_round_journal.append(inventory_round)

    def write_ber_journal(self, frame_ber_record):
        assert isinstance(frame_ber_record, FrameBERRecord)
        if self.frame_ber_logging_enabled:
            self.frame_ber_journal.append(frame_ber_record)

    def get_vehicle_read_rate(self):
        n_vehicles = len(self.vehicle_info)
        if n_vehicles <= self.n_skip_vehicles:
            return None
        n_vehicles -= self.n_skip_vehicles
        accum = 0
        for vid, record in self.vehicle_info.items():
            if vid > n_vehicles:
                continue
            print("vid={}, n_read={}".format(vid, record.n_read))
            if record.n_read > 0:
                accum += 1
        print("n_vehicles={}, accum={}".format(n_vehicles, accum))
        return accum / n_vehicles

    def get_tag_read_rate(self):
        n_vehicles = len(self.vehicle_info)
        if n_vehicles <= self.n_skip_vehicles:
            return None, None
        n_vehicles -= self.n_skip_vehicles
        acc_epc = 0
        acc_tid = 0
        n_tags = 0
        for epc, record in self.tag_info.items():
            assert isinstance(record, TagInfoRecord)
            if record.vehicle_id > n_vehicles:
                continue
            n_tags += 1
            acc_epc += 1 if record.n_epc_read > 0 else 0
            acc_tid += 1 if record.n_tid_read > 0 else 0
        pr_epc = acc_epc / n_tags
        pr_tid = acc_tid / n_tags
        return pr_epc, pr_tid

    def get_avg_vehicles_and_tags_num_per_round(self):
        n_vehicles = 0
        n_tags = 0
        n_records = len(self.inventory_round_journal)
        n_tags_in_used_rounds = 0
        n_used_rounds = 0
        if n_records == 0:
            return None, None, None
        for round_rec in self.inventory_round_journal:
            assert isinstance(round_rec, InventoryRoundRecord)
            n_vehicles += round_rec.n_vehicles_registered
            n_tags += round_rec.n_tags
            if round_rec.n_tags > 0:
                n_tags_in_used_rounds += round_rec.n_tags
                n_used_rounds += 1
        return (n_vehicles / n_records, n_tags / n_records,
                n_tags_in_used_rounds / n_used_rounds if n_used_rounds > 0
                else None)

    def get_avg_round_duration(self):
        n_rounds = len(self.inventory_round_journal)
        duration_ac = 0.0
        for record in self.inventory_round_journal:
            assert isinstance(record, InventoryRoundRecord)
            duration_ac += record.duration
        return None if n_rounds == 0 else duration_ac / n_rounds

    def get_avg_rounds_per_tag(self):
        tags = list(self.tag_info.values())
        n_tags = 0
        n_rounds_ac = 0
        n_vehicles = len(self.vehicle_info) - self.n_skip_vehicles
        for tag in tags:
            assert isinstance(tag, TagInfoRecord)
            if tag.vehicle_id > n_vehicles:
                continue
            n_tags += 1
            n_rounds_ac += tag.n_rounds
        return None if n_tags == 0 else n_rounds_ac / n_tags

    def get_avg_antenna_interval(self):
        n_switches = 0
        duration_ac = 0.0
        cur_index = None
        for record in self.inventory_round_journal:
            assert isinstance(record, InventoryRoundRecord)
            if cur_index != record.antenna_index:
                n_switches += 1
                cur_index = record.antenna_index
            duration_ac += record.duration
        return duration_ac / n_switches if n_switches > 0 else None

    def print_all(self, print_tag_read_data=None, print_inventory_rounds=None,
                  print_channel_state=None, print_frame_ber=None):
        def print_header(header, top_symbol='-', bottom_symbol='-'):
            print("+" + str(top_symbol) * 78 + "+")
            print("| {:<77s}|".format(header))
            print("+" + str(bottom_symbol) * 78 + "+")

        print_header("VEHICLE INFO")
        print(dict_to_df(self.vehicle_info).to_string())

        print_header("TAGS INFO")
        print(dict_to_df(self.tag_info).to_string())

        print_header("READER INFO")
        print(record_to_df(self.reader_info).to_string())

        print_header("READER ANTENNAS INFO")
        print(dict_to_df(self.reader_antenna_info).to_string())

        print_header("CHANNEL INFO")
        print(record_to_df(self.channel_info).to_string())

        if print_tag_read_data is None:
            print_tag_read_data = self.tag_read_logging_enabled
        if print_channel_state is None:
            print_channel_state = self.channel_state_logging_enabled
        if print_inventory_rounds is None:
            print_inventory_rounds = self.inventory_round_logging_enabled
        if print_frame_ber is None:
            print_frame_ber = self.frame_ber_logging_enabled

        if print_tag_read_data:
            print_header("TAG READ JOURNAL")
            print(list_to_df(self.tag_read_journal).to_string())

        if print_inventory_rounds:
            print_header("INVENTORY ROUND JOURNAL")
            print(list_to_df(self.inventory_round_journal).to_string())

        if print_channel_state:
            print_header("CHANNEL STATE JOURNAL")
            print(list_to_df(self.channel_state_journal).to_string())

        if print_frame_ber:
            print_header("BER PER FRAME")
            print(list_to_df(self.frame_ber_journal).to_string())


def get_vector_item(v, i, default=None):
    return v[i] if v is not None and len(v) > i else default


def get_rp_name(rp):
    if rp == pyradise.isotropic_rp:
        return 'isotropic'
    elif rp == pyradise.dipole_rp:
        return 'dipole'
    elif rp == pyradise.array_dipole_rp:
        return 'dipole-array'
    elif rp == pyradise.helix_rp:
        return 'helix'
    elif rp == pyradise.patch_rp:
        return 'patch'
    else:
        raise RuntimeError("unrecognized radiation patter: {}".format(rp))


def get_enum_name(v, default=None):
    return v.name if default is not None else default


def list_to_df(journal):
    return pd.DataFrame((x.to_dict() for x in journal))


def dict_to_df(journal):
    return pd.DataFrame((x.to_dict() for x in journal.values()))


def record_to_df(record):
    return list_to_df([record])
