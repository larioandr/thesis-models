import numpy as np
import matplotlib.pyplot as plt
import time as systime

import pyons

from . import journal
from . factory import Factory
from .model import Model
from .parameters import ModelDescriptor
from .generator import Generator
from . import protocol as gen2
from .reader import Reader
from . import pyradise

# @pyons.stop_condition()
# def check_sim_time():
#     return pyons.time() > pyons.get_model().params.vehicle_lifetime + 0.1

def plot_path_loss_contour(df, lane, reader_side, tag_location):
    df = df[(df.reader_lane == df.tag_lane) & (df.reader_lane == lane) &
            (df.reader_side == reader_side) & (df.tag_loc == tag_location) &
            ((reader_side == 'front') & (df.reader_x >= df.tag_x) |
             (reader_side == 'back') & (df.reader_x <= df.tag_x))]
    time = df.channel_time.as_matrix()
    distance = np.abs((df.reader_x - df.tag_x).as_matrix())
    rt_pl = df.rt_path_loss.as_matrix()
    tr_pl = df.tr_path_loss.as_matrix()
    ber = df.reader_ber

    tv, dv = np.meshgrid(distance, time)
    fig, ax_rt_pl = plt.subplots(1, 1)
    ax_rt_pl.contourf(tv, dv, rt_pl)
    plt.show()


def plot_path_loss_line(chan_df, lane, reader_side, tag_location, vehicle_id):
    df = chan_df
    df = df[(df.reader_lane == df.tag_lane) & (df.reader_lane == lane) &
            (df.reader_side == reader_side) & (df.tag_loc == tag_location) &
            ((reader_side == 'front') & (df.reader_x >= df.tag_x) |
             (reader_side == 'back') & (df.reader_x <= df.tag_x)) &
            (df.vehicle_id == vehicle_id)]

    # print(chan_df.head(10).to_string())
    # print("\n============ PL+BER for lane={}, side={}, location={}, vid={}"
    #       "".format(lane, reader_side, tag_location, vehicle_id))
    # print(df.to_string())
    # print("="*80)

    distance = np.abs((df.reader_x - df.tag_x).as_matrix())
    rt_pl = df.rt_path_loss.as_matrix()
    tr_pl = df.tr_path_loss.as_matrix()
    reader_rx = df.reader_rx_power
    tag_rx = df.tag_rx_power
    ber = df.reader_ber

    fig, (ax_power, ax_ber) = plt.subplots(1, 2)
    ax_power.set_title("Power")
    ax_power.set_ylim([-120, 0])
    ax_power.grid()
    ax_power.plot(distance, rt_pl, label='R=>T PL')
    ax_power.plot(distance, tr_pl, label='T=>R PL')
    ax_power.plot(distance, reader_rx, label='Reader RX Power')
    ax_power.plot(distance, tag_rx, label='Tag RX Power')
    ax_power.legend()

    ax_ber.set_title("BER")
    ax_ber.plot(distance, ber)
    ax_ber.set_ylim([0.0, 0.01])
    ax_ber.set_yticks(np.arange(0.0, 0.01, 0.001))
    ax_ber.grid()

    plt.show()
    # print("HO-HO-HO\n" * 10)
    # print(df.to_string())

def main():
    md = ModelDescriptor()

    md.lanes_number = 2
    md.vehicle_tag_locations = ['front', 'back']
    md.reader_antennas_sides = ['front', 'back']
    md.reader_rounds_per_antenna = 1
    md.reader_session_strategy = Reader.SessionStrategy.ONLY_A
    md.vehicle_length = 4.0
    md.tag_start_offset = 31.0
    md.vehicle_speed = 20.0
    md.vehicle_lifetime = 2 * md.tag_start_offset / md.vehicle_speed
    md.use_doppler = True
    md.tag_modulation_loss = -12.0
    md.reader_antenna_cable_loss = -2.0
    md.vehicle_position_update_interval = 1e-3
    md.reader_antenna_polarization = 0.5
    md.tag_antenna_polarization = 1.0
    md.vehicle_generation_interval = lambda: np.random.uniform(0.4, 0.6)
    md.tag_sensitivity = -18
    md.max_vehicles_num = 100
    md.reader_ber_model = pyradise.ber_over_rayleigh
    md.tag_encoding = gen2.TagEncoding.M8
    md.dr = gen2.DR.DR_8
    md.tari = 6.25e-6
    md.reader_circulator_noise = -80.0
    md.reader_antenna_gain = 6
    md.tag_antenna_gain = 2.0
    # md.reader_antenna_rp = pyradise.
    # md.conductivity = 0.15
    # md.permittivity = 30.0
    md.trext = False
    md.q = 2

    print("M={}, trext={}, DR={}, tari={:.2f}us, Q={}, "
          "c={:.3f}, p={:.3f}, rounds_per_antenna={}, session_strategy={}"
          "".format(
            md.tag_encoding.name, md.trext, md.dr, md.tari * 1e6, md.q,
            md.conductivity, md.permittivity,
            md.reader_rounds_per_antenna, md.reader_session_strategy.name))

    factory = Factory(md)

    model = Model(md)
    pyons.set_model(model)

    model.channel = factory.build_channel()
    model.reader = factory.build_reader(model.channel)
    model.generator = Generator(md)

    print("SYMBOL DURATION: {}us".format(1./model.reader.blf*1e6))
    frame = gen2.TagFrame(m=md.tag_encoding, trext=md.trext,
                          blf=model.reader.blf)
    # print("PREAMBLE DURATION: {}us".format(frame.preamble_duration * 1e6))

    # pyons.Dispatcher().log_entities = True
    # pyons.Dispatcher().log_queue = True
    # pyons.Dispatcher().logging_interval = 1e4
    pyons.setup_env(log_level=pyons.LogLevel.WARNING)

    journal.Journal().channel_state_logging_enabled = False
    journal.Journal().inventory_round_logging_enabled = True
    journal.Journal().frame_ber_logging_enabled = False

    t_start = systime.time()
    pyons.run()
    print("+ elapsed time: {:.1f}s".format(systime.time() - t_start))

    journal.Journal().n_skip_vehicles = 6

    vehicle_reade_rate = journal.Journal().get_vehicle_read_rate()
    epc_rate, tid_rate = journal.Journal().get_tag_read_rate()
    avg_antenna_interval = journal.Journal().get_avg_antenna_interval()
    avg_rounds_per_tag = journal.Journal().get_avg_rounds_per_tag()
    avg_round_duration = journal.Journal().get_avg_round_duration()
    avg_n_vehicles, avg_n_tags_in_round, avg_n_tags_in_busy_round = \
        journal.Journal().get_avg_vehicles_and_tags_num_per_round()

    journal.Journal().print_all(print_inventory_rounds=False,
                                print_tag_read_data=False,
                                print_channel_state=False,
                                print_frame_ber=False)

    print("VEHICLE READ RATE:      {:.4f}".format(vehicle_reade_rate))
    print("TAG EPC READ RATE:      {:.4f}".format(epc_rate))
    print("TAG TID READ RATE:      {:.4f}".format(tid_rate))
    print("-" * 20)
    print("AVG VEHICLES NUM :      {:.4f}".format(avg_n_vehicles))
    print("AVG TAGS IN ROUND:      {:.4f}".format(avg_n_tags_in_round))
    print("AVG TAGS IN BUSY ROUND: {:.4f}".format(avg_n_tags_in_busy_round))
    print("-" * 20)
    print("AVG ROUNDS PER TAG    : {:.4f}".format(avg_rounds_per_tag))
    print("AVG ANTENNA INTERVAL  : {:.4f}us".format(avg_antenna_interval*1e6))
    print("AVG ROUND DURATION    : {:.4f}us".format(avg_round_duration*1e6))

    # chan_df = journal.list_to_df(journal.Journal().channel_state_journal)

    # plot_path_loss_line(chan_df, 0, reader_side='front', tag_location='front',
    #                     vehicle_id=1)
    # plot_path_loss_line(chan_df, 0, reader_side='front', tag_location='front',
    #                     vehicle_id=2)
    # plot_path_loss_line(chan_df, 0, reader_side='front', tag_location='front',
    #                     vehicle_id=3)

    # plot_path_loss_contour(chan_df, 0,
    #                        reader_side='front', tag_location='front')

    # channel_df = journal.list_to_df(journal.Journal().channel_state_journal)
    # fmt_power = '{:.2f}'.format
    # fmt_time = '{:.6f}'.format
    # fmt_pos = '{:.2f}'.format
    # formatters = dict(reader_rx_power=fmt_power, tag_rx_power=fmt_power,
    #                   rt_path_loss=fmt_power, tr_path_loss=fmt_power,
    #                   channel_time=fmt_time, timestamp=fmt_time,
    #                   reader_x=fmt_pos, reader_y=fmt_pos, reader_z=fmt_pos,
    #                   tag_x=fmt_pos, tag_y=fmt_pos, tag_z=fmt_pos)
    # print(channel_df.to_string(formatters=formatters))

if __name__ == '__main__':
    main()
