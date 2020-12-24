from unittest.mock import Mock, patch, ANY

import pytest
from numpy import asarray, cumsum

from pycsmaca.simulations.modules.app_layer import AppData
from pycsmaca.simulations.modules.network_layer import NetworkPacket
from pycsmaca.simulations.modules.wired_interface import (
    WiredTransceiver, WireFrame, WiredInterface,
)

WIRE_FRAME_CLASS = 'pycsmaca.simulations.modules.wired_interface.WireFrame'


#############################################################################
# TEST WireFrame
#############################################################################
def test_wire_frame_init_and_properties():
    pkt_1 = NetworkPacket(data=AppData(100))
    pkt_2 = NetworkPacket(data=AppData(200))

    frame_1 = WireFrame(pkt_1, header_size=10, preamble=0.2, duration=1.5)
    assert frame_1.packet == pkt_1
    assert frame_1.duration == 1.5
    assert frame_1.header_size == 10
    assert frame_1.preamble == 0.2
    assert frame_1.size == 10 + pkt_1.size

    frame_2 = WireFrame(packet=pkt_2)
    assert frame_2.packet == pkt_2
    assert frame_2.duration == 0
    assert frame_2.header_size == 0
    assert frame_2.preamble == 0
    assert frame_2.size == 0


def test_wire_frame_implements_str():
    pkt_1 = NetworkPacket(data=AppData(100))
    pkt_2 = NetworkPacket(data=AppData(200))

    frame_1 = WireFrame(pkt_1, header_size=10, preamble=1, duration=2)
    assert str(frame_1) == f'WireFrame[D=2,HDR=10,PR=1 | {pkt_1}]'

    frame_2 = WireFrame(pkt_2)
    assert str(frame_2) == f'WireFrame[D=0,HDR=0,PR=0 | {pkt_2}]'


#############################################################################
# TEST WiredTransceiver MODEL
#############################################################################

# noinspection PyPropertyAccess
@pytest.mark.parametrize('bitrate, header_size, preamble, ifs', (
        (100, 10, 0.2, 0.05),
        (512, 22, 0.08, 0.1),
))
def test_wired_transceiver_properties(bitrate, header_size, preamble, ifs):
    sim = Mock()
    sim.stime = 13
    iface = WiredTransceiver(
        sim, bitrate=bitrate, header_size=header_size, preamble=preamble,
        ifs=ifs,
    )

    # Check that transceiver has bitrate, header size, preamble and ifs attrs:
    assert iface.bitrate == bitrate
    assert iface.header_size == header_size
    assert iface.preamble == preamble
    assert iface.ifs == ifs

    # We also check that transceiver is in ready state, but not started:
    assert not iface.started
    assert iface.tx_ready
    assert not iface.tx_busy
    assert iface.rx_ready
    assert not iface.rx_busy

    # Check that statuses are read-only from outside:
    with pytest.raises(AttributeError):
        iface.started = False
    with pytest.raises(AttributeError):
        iface.tx_ready = True
    with pytest.raises(AttributeError):
        iface.tx_busy = False
    with pytest.raises(AttributeError):
        iface.rx_ready = True
    with pytest.raises(AttributeError):
        iface.rx_busy = False

    # Finally, we assert that `WiredTransceiver` scheduled `start()` call:
    sim.schedule.assert_called_once_with(sim.stime, iface.start)


@pytest.mark.parametrize('bitrate, header_size, preamble, ifs', (
        (100, 10, 0.2, 0.05),
        (512, 22, 0.08, 0.1),
))
def test_wired_transceiver_packet_from_queue_transmission(
        bitrate, header_size, preamble, ifs):
    sim = Mock()
    iface = WiredTransceiver(
        sim, bitrate=bitrate, header_size=header_size, preamble=preamble,
        ifs=ifs,
    )

    # Now we connect the transceiver with a queue and start it. Make sure
    # that the queue is connected via 'queue' link, and after start `get_next()`
    # is called.
    queue = Mock()
    queue_rev_conn = Mock()
    queue.connections.set = Mock(return_value=queue_rev_conn)

    queue_conn = iface.connections.set('queue', queue, rname='iface')
    queue.get_next.assert_not_called()

    iface.start()  # start of the transceiver causes `get_next()` call

    queue.get_next.assert_called_once_with(iface)
    queue.get_next.reset_mock()
    assert iface.started and iface.tx_ready and not iface.tx_busy

    #
    # After being started, transceiver expects a `NetworkPacket` in its
    # handle_message() call. We connect another mock to the transceiver via
    # 'peer' connection and make sure that after the call that `send()` was
    # called on that peer connection.
    #
    # Since `WireFrame` objects are expected to be used in connections
    # between peers, we patch them.
    #
    peer = Mock()
    peer_rev_conn = Mock()
    peer.connections.set = Mock(return_value=peer_rev_conn)

    iface.connections.set('peer', peer, rname='peer')
    packet = NetworkPacket(data=AppData(size=500))
    duration = (packet.size + header_size) / bitrate + preamble

    with patch(WIRE_FRAME_CLASS) as WireFrameMock:
        frame_kwargs = {
            'packet': packet,
            'header_size': header_size,
            'duration': duration,
            'preamble': preamble,
        }
        frame_instance = Mock()
        frame_instance.duration = duration
        frame_instance.size = header_size + packet.size
        WireFrameMock.return_value = frame_instance

        sim.stime = 0
        iface.handle_message(packet, sender=queue, connection=queue_conn)
        sim.schedule.assert_any_call(
            0, peer.handle_message, args=(frame_instance,), kwargs={
                'connection': peer_rev_conn, 'sender': iface,
            }
        )
        WireFrameMock.assert_called_once_with(**frame_kwargs)

        # Also check that wired transceiver scheduled a timeout:
        sim.schedule.assert_any_call(duration, iface.handle_tx_end)

        # .. and that now transceiver is busy:
        assert iface.started and not iface.tx_ready and iface.tx_busy
        sim.schedule.reset_mock()

    # Now we imitate `handle_tx_end()` call, make sure that after that the
    # transceiver is not yet ready, but schedules `handle_ifs_end()`:
    sim.stime = duration
    iface.handle_tx_end()
    sim.schedule.assert_called_once_with(ifs, iface.handle_ifs_end)
    assert iface.started and not iface.tx_ready and iface.tx_busy

    # After the IFS waiting finished, transceiver is expected to call
    # `queue.get_next(iface)` and be ready for new packets:
    sim.stime += ifs
    iface.handle_ifs_end()
    queue.get_next.assert_called_once_with(iface)
    assert iface.started and iface.tx_ready and not iface.tx_busy


def test_wired_transceiver_raises_error_if_requested_tx_during_another_tx():
    sim, peer, queue = Mock(), Mock(), Mock()
    iface = WiredTransceiver(sim, bitrate=100)
    queue_conn = iface.connections.set('queue', queue, rname='iface')
    iface.connections.set('peer', peer, rname='peer')

    pkt_1 = NetworkPacket(data=AppData(size=10))
    pkt_2 = NetworkPacket(data=AppData(size=20))

    sim.stime = 0
    iface.start()
    iface.handle_message(pkt_1, sender=queue, connection=queue_conn)

    with pytest.raises(RuntimeError):
        iface.handle_message(pkt_2, sender=queue, connection=queue_conn)


def test_wired_transceiver_sends_data_up_when_rx_completed():
    sim, sender, switch = Mock(), Mock(), Mock()
    sim.stime = 0
    iface = WiredTransceiver(sim)
    sim.schedule.reset_mock()  # clear sim.schedule(0, iface.start) call

    pkt = NetworkPacket(data=AppData(size=100))
    frame = WireFrame(pkt, duration=0.5, header_size=20, preamble=0.01)

    switch_rev_conn = Mock()
    switch.connections.set = Mock(return_value=switch_rev_conn)
    iface.connections.set('up', switch, rname='iface')
    sender_conn = iface.connections.set('peer', sender, rname='peer')

    assert iface.rx_ready and not iface.rx_busy

    iface.handle_message(frame, sender=sender, connection=sender_conn)
    assert not iface.rx_ready and iface.rx_busy
    sim.schedule.assert_called_once_with(
        frame.duration, iface.handle_rx_end, args=(frame,),
    )
    sim.schedule.reset_mock()

    sim.stime += frame.duration
    iface.handle_rx_end(frame)
    sim.schedule.assert_called_once_with(
        0, switch.handle_message, args=(pkt,), kwargs={
            'sender': iface, 'connection': switch_rev_conn,
        }
    )
    assert iface.rx_ready and not iface.rx_busy


@pytest.mark.parametrize('bitrate, header_size, preamble, size', (
        (1000, 10, 0.2, 1540),
        (2000, 12, 0.3, 800),
))
def test_wired_transceiver_is_full_duplex(bitrate, header_size, preamble, size):
    sim, peer, queue, switch = Mock(), Mock(), Mock(), Mock()
    sim.stime = 0

    eth = WiredTransceiver(
        sim, header_size=header_size, bitrate=bitrate, preamble=preamble, ifs=0)

    peer_conn = eth.connections.set('peer', peer, reverse=False)
    queue_conn = eth.connections.set('queue', queue, reverse=False)
    eth.connections.set('up', switch, reverse=False)

    inp_pkt = NetworkPacket(data=AppData(size=size))
    out_pkt_1 = NetworkPacket(data=AppData(size=size))
    out_pkt_2 = NetworkPacket(data=AppData(size=size))
    duration = (header_size + size) / bitrate + preamble
    frame = WireFrame(inp_pkt, duration=duration, header_size=header_size,
                      preamble=preamble)

    # 1) Transceiver starts transmitting `out_pkt_1`:
    sim.stime = 0
    eth.start()
    eth.handle_message(out_pkt_1, queue_conn, queue)
    assert eth.tx_busy
    assert eth.rx_ready
    sim.schedule.assert_any_call(duration, eth.handle_tx_end)
    sim.schedule.assert_any_call(0, peer.handle_message, args=ANY, kwargs=ANY)
    sim.schedule.reset_mock()

    # 2) Then, after 2/3 of the packet was transmitted, a packet arrives:
    sim.stime = 2 * duration / 3
    eth.handle_message(frame, peer_conn, peer)
    assert eth.tx_busy
    assert eth.rx_busy
    sim.schedule.assert_called_with(duration, eth.handle_rx_end, args=(frame,))
    sim.schedule.reset_mock()

    # 3) After duration, call handle_tx_end and handle_ifs_end:
    sim.stime = duration
    eth.handle_tx_end()
    eth.handle_ifs_end()
    assert eth.tx_ready
    assert eth.rx_busy
    sim.schedule.reset_mock()

    # 4) After another 1/3 duration start new TX (during RX this time):
    sim.stime = 4 / 3 * duration
    eth.handle_message(out_pkt_2, queue_conn, queue)
    assert eth.tx_busy
    assert eth.rx_busy
    sim.schedule.assert_any_call(duration, eth.handle_tx_end)
    sim.schedule.assert_any_call(0, peer.handle_message, args=ANY, kwargs=ANY)
    sim.schedule.reset_mock()

    # 5) After 5/3 duration, RX ends, but TX still goes on:
    sim.stime = 5 / 3 * duration
    eth.handle_rx_end(frame)
    assert eth.tx_busy
    assert eth.rx_ready
    sim.schedule.assert_called_with(0, switch.handle_message, args=ANY,
                                    kwargs=ANY)


def test_wired_transceiver_ignores_frames_not_from_peer():
    sim, sender, switch = Mock(), Mock(), Mock()
    sim.stime = 0
    iface = WiredTransceiver(sim)
    sim.schedule.reset_mock()  # clear sim.schedule(0, iface.start) call

    pkt = NetworkPacket(data=AppData(size=100))
    frame = WireFrame(pkt, duration=0.5, header_size=20, preamble=0.01)

    iface.connections.set('up', switch, reverse=False)
    sender_conn = iface.connections.set('wrong_name', sender, reverse=False)

    iface.handle_message(frame, sender=sender, connection=sender_conn)
    sim.schedule.assert_not_called()
    assert iface.rx_ready


def test_wired_transceiver_drops_received_message_if_not_connected_to_switch():
    sim, sender = Mock(), Mock()
    sim.stime = 0

    iface = WiredTransceiver(sim)
    sender_conn = iface.connections.set('peer', sender, rname='peer')

    pkt = NetworkPacket(data=AppData(size=100))
    frame = WireFrame(pkt, duration=0.5, header_size=20, preamble=0.01)

    iface.handle_message(frame, sender=sender, connection=sender_conn)
    sim.stime += frame.duration

    sim.schedule.reset_mock()
    iface.handle_rx_end(frame)
    sim.schedule.assert_not_called()


@pytest.mark.parametrize(
    'bitrate, data_sizes, header_size, preamble, intervals', [
        (1000, [100, 150, 220, 329], 12, 0.05, [1.1, 2.3, 0, 0.5]),
        (1500, [90, 132, 85, 412], 20, 0.01, [0.05, 0, 0, 1.2]),
    ]
)
def test_wired_transceiver_records_rx_statistics(
        bitrate, data_sizes, header_size, preamble, intervals):
    sim, sender = Mock(), Mock()
    sim.stime = 0

    iface = WiredTransceiver(sim, bitrate, header_size, preamble)
    sender_conn = iface.connections.set('peer', sender, rname='peer')

    packets = [NetworkPacket(data=AppData(size=sz)) for sz in data_sizes]
    durations = [(sz + header_size) / bitrate + preamble for sz in data_sizes]
    frames = [
        WireFrame(pkt, dt, header_size, preamble)
        for pkt, dt in zip(packets, durations)
    ]
    t, timestamps = 0, []
    for interval, duration in zip(intervals, durations):
        t_arrival = t + interval
        t_departure = t_arrival + duration
        t = t_departure
        timestamps.append((t_arrival, t_departure))

    # Simulating receive sequence
    for (t_arrival, t_departure), frame in zip(timestamps, frames):
        sim.stime = t_arrival
        iface.handle_message(frame, sender=sender, connection=sender_conn)

        sim.stime = t_departure
        iface.handle_rx_end(frame)

    # Check RX statistics:
    expected_busy_trace = [(0, 0)]
    for t_arrival, t_departure in timestamps:
        expected_busy_trace.append((t_arrival, 1))
        expected_busy_trace.append((t_departure, 0))

    assert iface.num_received_frames == len(frames)
    assert iface.num_received_bits == sum(frame.size for frame in frames)
    assert iface.rx_busy_trace.as_tuple() == tuple(expected_busy_trace)


@pytest.mark.parametrize(
    'bitrate, data_sizes, header_size, preamble, intervals, ifs', [
        (1000, [100, 150, 220, 329], 12, 0.05, [1.1, 2.3, 0, 0.5], 0.05),
        (1500, [90, 132, 85, 412], 20, 0.01, [0.05, 0, 0, 1.2], 0.13),
    ]
)
def test_wired_transceiver_records_tx_statistics(
        bitrate, data_sizes, header_size, preamble, intervals, ifs):
    sim, receiver, queue = Mock(), Mock(), Mock()
    sim.stime = 0

    iface = WiredTransceiver(sim, bitrate, header_size, preamble, ifs)
    iface.connections.set('peer', receiver, rname='peer')
    queue_conn = iface.connections.set('queue', queue, reverse=False)

    packets = [NetworkPacket(data=AppData(size=sz)) for sz in data_sizes]
    frame_sizes = [sz + header_size for sz in data_sizes]
    durations = [(sz + header_size) / bitrate + preamble for sz in data_sizes]
    t, timestamps = 0, []
    for interval, duration in zip(intervals, durations):
        t_arrival = t + interval
        t_departure = t_arrival + duration + ifs
        timestamps.append((t_arrival, t_departure))
        t = t_departure

    # Simulating transmit sequence
    for (t_arrival, t_departure), packet in zip(timestamps, packets):
        sim.stime = t_arrival
        iface.handle_message(packet, sender=queue, connection=queue_conn)

        sim.stime = t_departure - ifs
        iface.handle_tx_end()

        sim.stime = t_departure
        iface.handle_ifs_end()

    # Check TX statistics:
    expected_busy_trace = [(0, 0)]
    for t_arrival, t_departure in timestamps:
        expected_busy_trace.append((t_arrival, 1))
        expected_busy_trace.append((t_departure, 0))

    assert iface.num_transmitted_packets == len(packets)
    assert iface.num_transmitted_bits == sum(sz for sz in frame_sizes)
    assert iface.tx_busy_trace.as_tuple() == tuple(expected_busy_trace)


#############################################################################
# TEST WiredInterface MODEL
#############################################################################
# noinspection PyProtectedMember,PyPropertyAccess
def test_wired_interface_creation_and_properties():
    sim, queue, transceiver = Mock(), Mock(), Mock()

    transceiver_rev_conn = Mock()
    transceiver.connections.set = Mock(return_value=transceiver_rev_conn)

    iface = WiredInterface(sim, 13, queue, transceiver)

    # 1) Check components and properties:
    assert iface.address == 13
    with pytest.raises(AttributeError):
        iface.address = 5

    assert iface.queue == queue
    with pytest.raises(AttributeError):
        iface.queue = queue

    assert iface.transceiver == transceiver
    with pytest.raises(AttributeError):
        iface.transceiver = transceiver

    # 2) Make sure that queue and transceiver are children of the interface:
    queue._set_parent.assert_called_once_with(iface)
    transceiver._set_parent.assert_called_once_with(iface)

    # 3) Check that internal connections are established:
    # 3.1) '_queue': iface --> queue:
    assert '_queue' in iface.connections
    assert iface.connections['_queue'].module == queue

    # 3.2) '_receiver': transceiver ---{'up'}--------> iface
    #                   transceiver <--{'_receiver'}-- iface
    assert '_receiver' in iface.connections
    assert iface.connections['_receiver'].module == transceiver
    assert iface.connections['_receiver'].reverse == transceiver_rev_conn
    transceiver.connections.set.assert_any_call('up', iface, reverse=False)

    # 3.3) '_peer': transceiver ---{'peer'}--> iface
    #               transceiver <--{'_peer'}-- iface
    assert '_peer' in iface.connections
    assert iface.connections['_peer'].module == transceiver
    assert iface.connections['_peer'].reverse == transceiver_rev_conn
    transceiver.connections.set.assert_any_call('peer', iface, reverse=False)


def test_wired_interface_forwards_packets_from_user_to_queue():
    sim, queue, transceiver, user = Mock(), Mock(), Mock(), Mock()
    iface = WiredInterface(sim, 13, queue, transceiver)

    user_conn = iface.connections.set('user', user, rname='iface')
    pkt = NetworkPacket(data=AppData(size=100))
    iface.handle_message(pkt, connection=user_conn, sender=user)

    int_queue_conn = iface.connections['_queue']

    sim.schedule.assert_called_once_with(
        0, queue.handle_message, args=(pkt,), kwargs={
            'connection': int_queue_conn.reverse, 'sender': iface,
        })


def test_wired_interface_forwards_frames_from_wire_to_transceiver():
    sim, queue, transceiver, peer = Mock(), Mock(), Mock(), Mock()
    iface = WiredInterface(sim, 13, queue, transceiver)

    peer_conn = iface.connections.set('wire', peer, rname='wire')
    frame = WireFrame(NetworkPacket(data=AppData(size=100)))
    iface.handle_message(frame, connection=peer_conn, sender=peer)

    int_peer_conn = iface.connections['_peer']

    sim.schedule.assert_called_once_with(
        0, transceiver.handle_message, args=(frame,), kwargs={
            'connection': int_peer_conn.reverse, 'sender': iface
        })


def test_wired_interface_forwards_packets_after_rx_end_to_user():
    sim, queue, transceiver, user = Mock(), Mock(), Mock(), Mock()
    iface = WiredInterface(sim, 13, queue, transceiver)

    user_conn = iface.connections.set('user', user, rname='iface')

    int_receiver_conn = iface.connections['_receiver']
    pkt = NetworkPacket(data=AppData(size=100))

    iface.handle_message(pkt, connection=int_receiver_conn, sender=transceiver)

    sim.schedule.assert_called_once_with(
        0, user.handle_message, args=(pkt,), kwargs={
            'connection': user_conn.reverse, 'sender': iface,
        })


def test_wired_interface_integration_serves_user_packet():
    sim, user, peer = Mock(), Mock(), Mock()
    sim.stime = 10

    from pycsmaca.simulations.modules.queues import Queue
    queue = Queue(sim)
    transceiver = WiredTransceiver(sim, 1000, 22, 0.03, 0.05)
    iface = WiredInterface(sim, 1, queue=queue, transceiver=transceiver)

    user_conn = iface.connections.set('user', user, rname='iface')

    wire_rev_conn = Mock()
    peer.connections.set = Mock(return_value=wire_rev_conn)
    wire_conn = iface.connections.set('wire', peer, rname='wire')
    wire_conn.delay = 0.01

    user_pkt = NetworkPacket(data=AppData(size=100))

    transceiver_queue_conn = transceiver.connections['queue']
    queue_iface_conn = iface.connections['_queue'].reverse
    iface_transceiver_conn = iface.connections['_peer']

    # First of all, we need to force transceiver start, since no actual
    # simulation execution is performed:
    transceiver.start()

    # 1) Simulate like a new packet arrived from user, make sure that queue
    #    delivery was scheduled:
    iface.handle_message(user_pkt, connection=user_conn, sender=user)
    sim.schedule.assert_called_with(
        0, queue.handle_message, args=(user_pkt,), kwargs={
            'connection': queue_iface_conn, 'sender': iface,
        })
    sim.schedule.reset_mock()

    # 2) Force execution of queue packet delivery, make sure the packet arrives
    #    at the transceiver:
    queue.handle_message(user_pkt, queue_iface_conn, iface)
    sim.schedule.assert_called_with(
        0, transceiver.handle_message, args=(user_pkt,), kwargs={
            'connection': transceiver_queue_conn, 'sender': queue,
        })
    sim.schedule.reset_mock()

    # 3) Force packet handling at the transceiver and make sure it schedules
    #    packet delivery at its peer (iface itself):
    transceiver.handle_message(
        user_pkt, connection=transceiver_queue_conn, sender=queue)
    frame = transceiver.tx_frame
    sim.schedule.assert_any_call(
        0, iface.handle_message, args=(frame,), kwargs={
            'connection': iface_transceiver_conn, 'sender': transceiver,
        }
    )
    assert frame.packet == user_pkt
    sim.schedule.reset_mock()

    # 4) Finally, force frame arrival at the interface and make sure it
    #    schedules frame reception at its peer after the channel delay:
    iface.handle_message(frame, iface_transceiver_conn, transceiver)
    sim.schedule.assert_called_with(
        wire_conn.delay, peer.handle_message, args=(frame,), kwargs={
            'connection': wire_rev_conn, 'sender': iface,
        })


def test_wired_interface_integration_receives_frame():
    sim, user, peer = Mock(), Mock(), Mock()
    sim.stime = 10

    from pycsmaca.simulations.modules.queues import Queue
    queue = Queue(sim)
    transceiver = WiredTransceiver(sim, 1000, 22, 0.1, 0.05)
    iface = WiredInterface(sim, 0, queue=queue, transceiver=transceiver)

    user_rev_conn = Mock()
    user.connections.set = Mock(return_value=user_rev_conn)
    iface.connections.set('user', user, rname='iface')

    wire_rev_conn = Mock()
    peer.connections.set = Mock(return_value=wire_rev_conn)
    wire_conn = iface.connections.set('wire', peer, rname='wire')

    packet = NetworkPacket(data=AppData(size=242))
    duration = (packet.size + transceiver.header_size
                ) / transceiver.bitrate + transceiver.preamble
    frame = WireFrame(
        packet, duration, transceiver.header_size, transceiver.preamble)

    transceiver_peer_conn = iface.connections['_peer'].reverse
    _receiver_conn = iface.connections['_receiver']

    # 1) Simulate like a frame came from the peer:
    iface.handle_message(frame, connection=wire_conn, sender=peer)
    sim.schedule.assert_called_with(
        0, transceiver.handle_message, args=(frame,), kwargs={
            'connection': transceiver_peer_conn, 'sender': iface,
        })
    sim.schedule.reset_mock()

    # 2) Execute transceiver frame reception start, update time and
    #    execute transceiver frame reception end. Then make sure that
    #    packet was scheduled for sending up to the interface via 'up':
    transceiver.handle_message(frame, transceiver_peer_conn, iface)
    sim.schedule.assert_called_with(
        duration, transceiver.handle_rx_end, args=(frame,))
    sim.schedule.reset_mock()
    sim.stime += duration
    transceiver.handle_rx_end(frame)
    sim.schedule.assert_called_with(
        0, iface.handle_message, args=(packet,), kwargs={
            'connection': _receiver_conn, 'sender': transceiver,
        })
    sim.schedule.reset_mock()

    # 3) Execute interface packet reception, make sure it is delivered to user:
    iface.handle_message(packet, _receiver_conn, transceiver)
    sim.schedule.assert_called_with(
        0, user.handle_message, args=(packet,), kwargs={
            'connection': user_rev_conn, 'sender': iface,
        })
    sim.schedule.reset_mock()
