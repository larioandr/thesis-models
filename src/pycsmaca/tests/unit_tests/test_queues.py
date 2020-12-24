from unittest.mock import Mock

import pytest
from numpy import cumsum
from numpy.random.mtrand import uniform

from pycsmaca.simulations.modules.app_layer import AppData
from pycsmaca.simulations.modules.network_layer import NetworkPacket
from pycsmaca.simulations.modules.queues import Queue, SaturatedQueue


#############################################################################
# TEST Queue
#############################################################################
def test_new_finite_queue_is_empty():
    sim = Mock()
    sim.stime = 0

    queue = Queue(sim, capacity=2)

    assert queue.empty()
    assert not queue.full()
    assert len(queue) == 0
    assert queue.size() == 0
    assert queue.bitsize() == 0
    assert queue.as_tuple() == ()


def test_push_to_empty_queue_without_service_correctly_updates_content():
    sim = Mock()
    sim.stime = 0

    queue = Queue(sim, capacity=2)
    data_size = 123
    packet = NetworkPacket(data=AppData(0, data_size, 0, 0))

    queue.push(packet)
    assert not queue.empty()
    assert not queue.full()
    assert len(queue) == 1
    assert queue.size() == 1
    assert queue.bitsize() == data_size
    assert tuple(qp.packet for qp in queue.as_tuple()) == (packet,)


def test_push_up_to_full_queue_without_service_correctly_updates_content():
    sim = Mock()
    sim.stime = 0
    data_size = [123, 412]
    packets = [NetworkPacket(data=AppData(0, sz, 0, 0)) for sz in data_size]

    queue = Queue(sim, capacity=2)
    queue.push(packets[0])
    queue.push(packets[1])

    assert not queue.empty()
    assert queue.full()
    assert len(queue) == 2
    assert queue.size() == 2
    assert queue.bitsize() == sum(data_size)
    assert tuple(qp.packet for qp in queue.as_tuple()) == tuple(packets)


def test_push_to_full_queue_without_service_drops_last_packet():
    sim = Mock()
    sim.stime = 0
    data_size = [123, 412]
    packets = [NetworkPacket(data=AppData(0, sz, 0, 0)) for sz in data_size]

    queue = Queue(sim, capacity=1)
    queue.push(packets[0])

    # Check that num_dropped counter is 0 before overflow:
    assert queue.num_dropped == 0

    # Pushing a packet that will overflow the queue:
    queue.push(packets[1])
    assert queue.num_dropped == 1

    # Now check that only first packet is in the queue:
    assert not queue.empty()
    assert queue.full()
    assert len(queue) == 1
    assert queue.size() == 1
    assert queue.bitsize() == data_size[0]
    assert tuple(qp.packet for qp in queue.as_tuple()) == (packets[0],)


def test_pop_from_empty_queue_raises_error():
    sim = Mock()
    sim.stime = 0

    queue = Queue(sim, capacity=2)
    with pytest.raises(ValueError):
        queue.pop()


def test_pop_extracts_packets_in_correct_order():
    sim = Mock()
    sim.stime = 0
    data_size = [123, 412]
    packets = [NetworkPacket(data=AppData(0, sz, 0, 0)) for sz in data_size]

    queue = Queue(sim, capacity=2)
    queue.push(packets[0])
    queue.push(packets[1])

    assert queue.pop() == packets[0]
    assert not queue.empty()
    assert not queue.full()
    assert len(queue) == 1
    assert queue.size() == 1
    assert queue.bitsize() == data_size[1]
    assert tuple(qp.packet for qp in queue.as_tuple()) == (packets[1],)

    assert queue.pop() == packets[1]
    assert queue.empty()
    assert not queue.full()
    assert len(queue) == 0
    assert queue.size() == 0
    assert queue.bitsize() == 0
    assert queue.as_tuple() == ()


def test_finite_queue_without_service_writes_statistics():
    sim = Mock()
    size = [123, 412, 230, 312]
    t0, t1, t2, t3, t4, t5 = 2, 7, 8, 10, 14, 19
    packets = [NetworkPacket(data=AppData(0, sz, 0, 0)) for sz in size]

    sim.stime = t0
    q = Queue(sim, capacity=2)

    # Run a sequence of operations:
    sim.stime = t1
    q.push(packets[0])  # stored after: packet[0]
    sim.stime = t2
    q.push(packets[1])  # stored after: packet[0], packet[1]
    sim.stime = t3
    q.push(packets[2])  # dropped due to overflow, stored: packet[0], packet[1]
    sim.stime = t4
    q.pop()             # stored after: packet[1]
    sim.stime = t5
    q.push(packets[3])  # stored after: packet[1], packet[3]

    assert tuple(qp.packet for qp in q.as_tuple()) == (packets[1], packets[3])
    assert q.size_trace.as_tuple() == (
        (t0, 0), (t1, 1), (t2, 2), (t4, 1), (t5, 2)
    )
    assert q.bitsize_trace.as_tuple() == (
        (t0, 0), (t1, size[0]), (t2, size[0] + size[1]), (t4, size[1]),
        (t5, size[1] + size[3]),
    )
    assert q.num_dropped == 1


def test_infinite_queue_stores_many_enough_packets():
    n = 50
    packets = [
        NetworkPacket(data=AppData(0, uniform(0, 1000), 0, 0)) for _ in range(n)
    ]
    times = list(cumsum(uniform(0, 20, n)))

    sim = Mock()
    sim.stime = 0

    queue = Queue(sim)

    for pkt, t in zip(packets, times):
        sim.stime = t
        queue.push(pkt)

    assert queue.size() == n
    assert len(queue.size_trace) == n + 1
    assert queue.num_dropped == 0


def test_queue_with_service_passes_new_packet_directly_after_get_next_call():
    sim, service = Mock(), Mock()
    sim.stime = 0

    service_rev_conn = Mock()
    service.connections.set = Mock(return_value=service_rev_conn)

    queue = Queue(sim=sim)
    queue.connections.set('service', service, rname='queue')
    queue.get_next(service=service)

    packet = NetworkPacket(data=AppData(size=100))
    sim.stime = 13
    queue.push(packet)

    # Check that the message was delivered to the service:
    sim.schedule.assert_called_once_with(
        0, service.handle_message, args=(packet,), kwargs={
            'connection': service_rev_conn, 'sender': queue,
        }
    )

    # Check that queue is still empty:
    assert queue.as_tuple() == ()

    # Also make sure that size updates were not written:
    assert queue.size_trace.as_tuple() == ((0, 0),)
    assert queue.bitsize_trace.as_tuple() == ((0, 0),)
    assert queue.num_dropped == 0


def test_queue_with_service_passes_single_stored_packet_after_get_next_call():
    t0, t1, t2, t3, t4 = 0, 13, 19, 22, 29
    size = [100, 200, 300]
    sim, service = Mock(), Mock()
    sim.stime = t0

    service_rev_conn = Mock()
    service.connections.set = Mock(return_value=service_rev_conn)

    queue = Queue(sim=sim)
    queue.connections.set('service', service, rname='queue')

    packets = [NetworkPacket(data=AppData(size=sz)) for sz in size]
    sim.stime = t1
    queue.push(packets[0])
    sim.stime = t2
    queue.push(packets[1])

    # Check that queue is updated, since no `get_next()` call was performed:
    assert tuple(qp.packet for qp in queue.as_tuple()) == tuple(packets[0:2])
    sim.schedule.assert_not_called()

    # Check that after `get_next()` request the message is passed:
    sim.stime = t3
    queue.get_next(service=service)
    assert tuple(qp.packet for qp in queue.as_tuple()) == (packets[1],)
    sim.schedule.assert_called_once_with(
        0, service.handle_message, args=(packets[0],), kwargs={
            'connection': service_rev_conn, 'sender': queue,
        }
    )

    sim.stime = t4
    queue.push(packets[2])
    assert tuple(qp.packet for qp in queue.as_tuple()) == tuple(packets[1:3])

    # Also make sure that size updates were written:
    assert queue.size_trace.as_tuple() == (
        (t0, 0), (t1, 1), (t2, 2), (t3, 1), (t4, 2),
    )
    assert queue.bitsize_trace.as_tuple() == (
        (t0, 0), (t1, size[0]), (t2, size[0] + size[1]), (t3, size[1]),
        (t4, size[1] + size[2]),
    )
    assert queue.num_dropped == 0


def test_queue_with_several_services_finds_right_connections():
    sim, blue, red, green = Mock(), Mock(), Mock(), Mock()
    sim.stime = 0

    blue_rev_conn = Mock()
    blue.connections.set = Mock(return_value=blue_rev_conn)
    red_rev_conn = Mock()
    red.connections.set = Mock(return_value=red_rev_conn)
    green_rev_conn = Mock()
    green.connections.set = Mock(return_value=green_rev_conn)

    queue = Queue(sim=sim)
    queue.connections.set('blue', blue, rname='queue')
    queue.connections.set('red', red, rname='queue')
    queue.connections.set('green', green, rname='queue')

    # First, we fill the queue:
    pkt_1 = NetworkPacket(data=AppData(size=100))
    pkt_2 = NetworkPacket(data=AppData(size=200))
    pkt_3 = NetworkPacket(data=AppData(size=300))

    # Now, while queue is empty, two services request data:
    sim.stime = 0.5
    queue.get_next(green)
    sim.stime = 1.0
    queue.get_next(blue)

    # At some moment, a packet arrives. It should be passed to the module that
    # first requested the packet:
    sim.stime = 2
    queue.push(pkt_1)
    sim.schedule.assert_called_once_with(
        0, green.handle_message, args=(pkt_1,), kwargs={
            'connection': green_rev_conn, 'sender': queue,
        }
    )
    assert queue.as_tuple() == ()
    sim.schedule.reset_mock()

    # At the next moment, another packet arrives and is being passed to the
    # module that requested data after the first one:
    sim.stime = 5
    queue.push(pkt_2)
    sim.schedule.assert_called_once_with(
        0, blue.handle_message, args=(pkt_2,), kwargs={
            'connection': blue_rev_conn, 'sender': queue,
        }
    )
    assert queue.as_tuple() == ()
    sim.schedule.reset_mock()

    # Now another packet arrives, and it should be stored since both requests
    # were fulfilled previously:
    sim.stime = 10
    queue.push(pkt_3)
    assert tuple(qp.packet for qp in queue.as_tuple()) == (pkt_3,)
    sim.schedule.assert_not_called()

    # Finally, the another module requests a packet, and it is immediately
    # delivered to it:
    sim.stime = 19
    queue.get_next(red)
    sim.schedule.assert_called_once_with(
        0, red.handle_message, args=(pkt_3,), kwargs={
            'connection': red_rev_conn, 'sender': queue,
        }
    )


def test_data_request_to_queue_from_module_not_connected_raises_error():
    sim, service = Mock(), Mock()
    sim.stime = 0

    queue = Queue(sim=sim)

    with pytest.raises(ValueError):
        queue.get_next(service=service)


def test_queue_accepts_packets_on_handle_message_call():
    sim, producer = Mock(), Mock()
    sim.stime = 0

    queue = Queue(sim=sim)
    conn = queue.connections.set('input', producer, reverse=False)

    pkt = NetworkPacket(data=AppData(size=123))

    queue.handle_message(pkt, sender=producer, connection=conn)
    assert tuple(qp.packet for qp in queue.as_tuple()) == (pkt,)


#############################################################################
# TEST SaturatedQueue
#############################################################################
def test_saturated_queue_requests_source_packet_when_empty_after_get_next():
    sim, source, switch, service = Mock(), Mock(), Mock(), Mock()
    sim.stime = 0

    service_rev_conn = Mock()
    service.connections.set = Mock(return_value=service_rev_conn)

    queue = SaturatedQueue(sim=sim, source=source)
    queue.connections.set('output', service, rname='queue')
    queue.connections.set('input', switch, reverse=False)

    # Requesting the next packet from the queue:
    queue.get_next(service)

    source.get_next.assert_called_once()

    # Assume that packet was generated by source, went through all layers
    # and arrived from the switch. Make sure that this packet will be
    # delivered to the service:
    pkt = NetworkPacket(data=AppData(size=100))
    queue.handle_message(pkt)
    sim.schedule.assert_called_once_with(
        0, service.handle_message, args=(pkt,), kwargs={
            'connection': service_rev_conn, 'sender': queue,
        })


def test_saturated_queue_not_requests_source_when_not_empty_after_get_next():
    sim, source, switch, service = Mock(), Mock(), Mock(), Mock()
    sim.stime = 0

    service_rev_conn = Mock()
    service.connections.set = Mock(return_value=service_rev_conn)

    queue = SaturatedQueue(sim=sim, source=source)
    queue.connections.set('output', service, rname='queue')
    queue.connections.set('input', switch, reverse=False)

    pkt = NetworkPacket(data=AppData(size=100))
    queue.push(pkt)

    assert queue.size() == 1
    queue.get_next(service)

    # Validate that packet was passed to the service, but source.get_next()
    # not called:
    sim.schedule.assert_called_once_with(
        0, service.handle_message, args=(pkt,), kwargs={
            'connection': service_rev_conn, 'sender': queue,
        })
    source.get_next.assert_not_called()

    # However, after the queue became empty, it will call source:
    queue.get_next(service)
    source.get_next.assert_called_once()
