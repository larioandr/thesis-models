from unittest.mock import Mock, patch, MagicMock

import pytest
from pydesim import Model

from pycsmaca.simulations.modules.network_layer import NetworkService, \
    NetworkPacket, SwitchTable, NetworkSwitch

NET_PACKET_CLASS = 'pycsmaca.simulations.modules.network_layer.NetworkPacket'


class DummyModel(Model):
    """We use this `DummyModel` when we need a full-functioning model.
    """
    def __init__(self, sim, name):
        super().__init__(sim)
        self.name = name

    def __str__(self):
        return self.name


#############################################################################
# TEST NetworkService
#############################################################################
def test_network_service_accepts_packets_from_app():
    sim, app, net = Mock(), Mock(), Mock()
    ns = NetworkService(sim)
    app_conn = ns.connections.set('source', app, reverse=False)

    net_conn = Mock()
    net.connections.set = Mock(return_value=net_conn)

    ns.connections.set('network', net, rname='user')
    net.connections.set.assert_called_once_with('user', ns, reverse=False)

    # Now we simulate packet arrival from APP:
    app_data = Mock()
    app_data.destination_address = 13

    with patch(NET_PACKET_CLASS) as NetworkPacketMock:
        pkt_spec = dict(destination_address=13, data=app_data)
        packet_instance_mock = Mock()
        NetworkPacketMock.return_value = packet_instance_mock

        # Calling `handle_message()` as it to be called upon receiving new
        # `AppData` from 'app' connection:
        ns.handle_message(app_data, connection=app_conn, sender=app)

        # Check that a packet was properly created and also that
        # Network.handle_message() was called:
        NetworkPacketMock.assert_called_once_with(**pkt_spec)
        sim.schedule.assert_called_with(
            0, net.handle_message, args=(packet_instance_mock,), kwargs={
                'connection': net_conn, 'sender': ns,
            }
        )


def test_network_service_fills_data_and_dst_addr_for_packet_from_app():
    sim, app, net = Mock(), Mock(), Mock()
    ns = NetworkService(sim)

    # noinspection PyUnusedLocal
    def schedule_mock(delay, method, args, kwargs):
        packet = args[0]
        connection, sender = kwargs['connection'], kwargs['sender']
        assert isinstance(packet, NetworkPacket)
        assert connection == net_rev_conn
        assert sender == ns
        assert packet.destination_address == 13
        assert packet.originator_address is None
        assert packet.sender_address is None
        assert packet.receiver_address is None

    net_rev_conn = Mock()
    net.connections.set = Mock(return_value=net_rev_conn)

    ns.connections.set('network', net, rname='user')
    app_conn = ns.connections.set('source', app, reverse=False)

    # Now we simulate packet arrival from APP:
    app_data = Mock()
    app_data.destination_address = 13

    sim.schedule = Mock(side_effect=schedule_mock)
    ns.handle_message(app_data, connection=app_conn, sender=app)
    sim.schedule.assert_called_once()


def test_network_service_ignores_app_data_via_other_connections():
    sim, app = Mock(), Mock()
    ns = NetworkService(sim)
    wrong_app_conn = ns.connections.set('wrong_name', app, reverse=False)

    # Now we simulate packet arrival from APP via unsupported connection:
    app_data = Mock()
    app_data.destination_address = 1
    with patch(NET_PACKET_CLASS) as NetworkPacketMock:
        # Imitate packet AppData arrival via wrong connections and make
        # sure it doesn't cause NetworkPacket instantiation:
        ns.handle_message(app_data, connection=wrong_app_conn, sender=app)
        NetworkPacketMock.assert_not_called()


def test_network_service_accept_packets_from_network():
    sim, network, sink = Mock(), Mock(), Mock()
    ns = NetworkService(sim)
    net_conn = ns.connections.set('network', network, reverse=False)

    sink_conn = Mock()
    sink.connections.set = Mock(return_value=sink_conn)
    ns.connections.set('sink', sink, rname='network')

    # Now we are going to simulate `NetworkPacket` arrival and make sure
    # `AppData` is extracted and passed up via the "sink" connection.
    # First, we define app_data and network_packet:
    app_data = Mock()
    network_packet = Mock()
    network_packet.data = app_data

    # Calling `handle_message()` as it to be called upon receiving new
    # `NetworkPacket` from 'network' connection:
    ns.handle_message(network_packet, connection=net_conn, sender=network)

    # Check that `sink.handle_message` call is scheduled:
    sim.schedule.assert_called_with(
        0, sink.handle_message, args=(app_data,), kwargs={
            'connection': sink_conn, 'sender': ns,
        }
    )


def test_network_service_ignores_net_packets_received_via_other_connections():
    sim, network = Mock(), Mock()
    ns = NetworkService(sim)
    wrong_conn = ns.connections.set('wrong_name', network, reverse=False)

    # Imitate `NetworkPacket` arrival via the wrong connection and make sure
    # nothing is being scheduled:
    network_packet = Mock()
    ns.handle_message(network_packet, connection=wrong_conn, sender=network)
    sim.schedule.assert_not_called()


def test_str_uses_parent_if_specified():
    sim = Mock()
    parent = DummyModel(sim, 'DummyParent')
    ns1 = NetworkService(sim)
    ns2 = NetworkService(sim)
    parent.children['ns'] = ns2

    assert str(ns1) == "NetworkService"
    assert str(ns2) == "DummyParent.NetworkService"


#############################################################################
# TEST NetworkPacket
#############################################################################
def test_network_packet_creation():
    data = Mock()
    packet = NetworkPacket(
        destination_address=10, originator_address=2, sender_address=5,
        receiver_address=6, osn=32, data=data)
    assert packet.destination_address == 10
    assert packet.originator_address == 2
    assert packet.sender_address == 5
    assert packet.receiver_address == 6
    assert packet.osn == 32
    assert packet.data == data


def test_network_packet_implements_str():
    data = MagicMock()
    data.__str__.return_value = 'AppData{sid=13}'
    pkt1 = NetworkPacket(
        destination_address=10, originator_address=2, sender_address=5,
        receiver_address=6, osn=4, data=data)
    pkt2 = NetworkPacket(destination_address=5, data=data)
    pkt3 = NetworkPacket(destination_address=8)
    assert str(pkt1) == f'NetPkt{{DST=10,ORIGIN=2,SND=5,RCV=6,OSN=4 | {data}}}'
    assert str(pkt2) == f'NetPkt{{DST=5 | {data}}}'
    assert str(pkt3) == f'NetPkt{{DST=8}}'


def test_network_packet_size():
    data = MagicMock()
    data.size = 100

    pkt1 = NetworkPacket(data=data)
    pkt2 = NetworkPacket()

    assert pkt1.size == 100
    assert pkt2.size == 0


#############################################################################
# TEST SwitchTable
#############################################################################
def test_switch_table_add_and_as_tuple_methods():
    table = SwitchTable()
    table.add(10, connection='eth0', next_hop=4)
    table.add(22, connection='eth1', next_hop=3)
    assert table.as_dict() == {10: ('eth0', 4), 22: ('eth1', 3)}


def test_switch_table_as_dict_returns_read_only_dict():
    table = SwitchTable()
    with pytest.raises(TypeError):
        table.as_dict()[13] = ('illegal', 66)


def test_switch_table_getitem_method():
    table = SwitchTable()
    table.add(10, connection='eth0', next_hop=4)

    link1 = table[10]
    assert link1.connection == 'eth0'
    assert link1.next_hop == 4

    with pytest.raises(KeyError):
        print(table[22])


def test_switch_table_record_can_be_updated():
    table = SwitchTable()
    table.add(13, connection='wifi', next_hop=5)

    link = table[13]
    link.connection = 'ge'
    link.next_hop = 24

    assert table.as_dict() == {13: ('ge', 24)}


def test_switch_table_provides_get_method():
    table = SwitchTable()
    table.add(13, connection='wifi', next_hop=5)

    assert table.get(13).connection == 'wifi'
    assert table.get(22) is None


def test_switch_table_implements_contains_magic_method():
    table = SwitchTable()
    table.add(13, connection='wifi', next_hop=5)

    assert 13 in table
    assert 14 not in table


def test_switch_table_add_replaces_existing_record():
    table = SwitchTable()
    table.add(10, connection='eth0', next_hop=4)
    assert table.as_dict() == {10: ('eth0', 4)}
    table.add(10, connection='wifi', next_hop=9)
    assert table.as_dict() == {10: ('wifi', 9)}


def test_switch_table_implements_str():
    table = SwitchTable()
    table.add(10, connection='eth0', next_hop=4)
    table.add(22, connection='eth1', next_hop=3)
    assert str(table) in {'SwitchTable{10: (eth0, 4), 22: (eth1, 3)}',
                          'SwitchTable{22: (eth1, 3), 10: (eth0, 4)}'}


#############################################################################
# TEST NetworkSwitch
#############################################################################
def test_network_switch_provides_table_read_only_property():
    sim = Mock()
    switch = NetworkSwitch(sim)
    assert isinstance(switch.table, SwitchTable)
    with pytest.raises(AttributeError):
        # noinspection PyPropertyAccess
        switch.table = SwitchTable()


def test_network_switch_routes_packets_from_user_to_remote_destinations():
    """Validate packets from source to known destination are properly served.

    In this test we define a model with `NetworkService` (mock'ed),
    `NetworkSwitch` (being tested) and two network interfaces - eth and wifi
    (both mock'ed). `NetworkSwitch` defines two routes via these interfaces.

    Then we generate a couple of packets for the destination known to the
    switch and imitate they being received by the switch in `handle_message`.
    We make sure that these packets are being filled with source, sender
    and received addresses (taken from the switching table), SSNs are assigned
    and the packets are transmitted to the proper network interfaces.
    """
    sim, ns, eth, wifi = Mock(), Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    eth.address = 4
    eth_conn = Mock()
    eth.connections.set = Mock(return_value=eth_conn)
    wifi.address = 8
    wifi_conn = Mock()
    wifi.connections.set = Mock(return_value=wifi_conn)

    user_conn = switch.connections.set('user', ns, reverse=False)
    switch.connections.set('eth', eth, rname='network')
    switch.connections.set('wifi', wifi, rname='network')

    switch.table.add(10, connection='eth', next_hop=5)
    switch.table.add(20, connection='wifi', next_hop=13)

    pkt_1 = NetworkPacket(destination_address=10)
    pkt_2 = NetworkPacket(destination_address=20)

    switch.handle_message(pkt_1, connection=user_conn, sender=ns)
    sim.schedule.assert_called_with(
        0, eth.handle_message, args=(pkt_1,), kwargs={
            'connection': eth_conn, 'sender': switch,
        }
    )
    assert pkt_1.receiver_address == 5  # = table[10].next_hop
    assert pkt_1.sender_address == 4  # = eth.address
    assert pkt_1.originator_address == 4  # = eth.address
    assert pkt_1.osn >= 0       # any value, but not None

    switch.handle_message(pkt_2, connection=user_conn, sender=ns)
    sim.schedule.assert_called_with(
        0, wifi.handle_message, args=(pkt_2,), kwargs={
            'connection': wifi_conn, 'sender': switch,
        }
    )
    assert pkt_2.receiver_address == 13   # = table[20].next_hop
    assert pkt_2.sender_address == 8    # = wifi.address
    assert pkt_2.originator_address == 8    # = wifi.address
    assert pkt_2.osn >= 0         # = any value, but not None


def test_network_switch_increments_ssn_for_successive_packets_from_same_src():
    """Validate when two packets come from 'user' to same dest, SSN increments.
    """
    sim, ns, eth = Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    eth.address = 1
    eth_conn = Mock()
    eth.connections.set = Mock(return_value=eth_conn)

    user_conn = switch.connections.set('user', ns, reverse=False)
    switch.connections.set('eth', eth, rname='network')

    switch.table.add(5, connection='eth', next_hop=2)
    switch.table.add(10, connection='eth', next_hop=3)

    pkt_1 = NetworkPacket(destination_address=5)
    pkt_2 = NetworkPacket(destination_address=10)

    switch.handle_message(pkt_1, connection=user_conn, sender=ns)
    switch.handle_message(pkt_2, connection=user_conn, sender=ns)

    assert pkt_2.osn > pkt_1.osn


def test_network_switch_ignores_packets_to_unknown_destinations():
    """Validate `NetworkSwitch` ignores messages without source not from 'user'.
    """
    sim, ns, eth = Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    eth.address = 1
    eth_conn = Mock()
    eth.connections.set = Mock(return_value=eth_conn)

    user_conn = switch.connections.set('invalid', ns, reverse=False)
    switch.connections.set('eth', eth, rname='network')

    switch.table.add(10, connection='eth', next_hop=2)

    pkt = NetworkPacket(destination_address=13)

    switch.handle_message(pkt, connection=user_conn, sender=ns)
    sim.schedule.assert_not_called()


def test_network_switch_sends_packets_with_its_interface_address_to_user():
    """Validate packet with destination matching one interface is routed up.

    We send three packets: one from 'eth', one from 'wifi' and one from 'user'.
    Make sure that in any case the packet is routed to user.
    """
    sim, ns, wifi, eth = Mock(), Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    ns_rev_conn = Mock()
    ns.connections.set = Mock(return_value=ns_rev_conn)

    eth.address = 1
    wifi.address = 2

    ns_conn = switch.connections.set('user', ns, rname='network')
    eth_conn = switch.connections.set('eth', eth, rname='network')
    wifi_conn = switch.connections.set('wifi', wifi, rname='network')

    pkt_1 = NetworkPacket(destination_address=2)
    pkt_2 = NetworkPacket(destination_address=2)
    pkt_3 = NetworkPacket(destination_address=2)

    # Sending the first packet from Ethernet interface:
    switch.handle_message(pkt_1, connection=eth_conn, sender=eth)
    sim.schedule.assert_called_once_with(
        0, ns.handle_message, args=(pkt_1,), kwargs={
            'connection': ns_rev_conn, 'sender': switch,
        }
    )
    sim.schedule.reset_mock()

    # Sending another packet like from WiFi interface:
    switch.handle_message(pkt_2, connection=wifi_conn, sender=wifi)
    sim.schedule.assert_called_once_with(
        0, ns.handle_message, args=(pkt_2,), kwargs={
            'connection': ns_rev_conn, 'sender': switch,
        }
    )
    sim.schedule.reset_mock()

    # Finally, send a packet from NetworkService (loopback-like behaviour):
    switch.handle_message(pkt_3, connection=ns_conn, sender=ns)
    sim.schedule.assert_called_once_with(
        0, ns.handle_message, args=(pkt_3,), kwargs={
            'connection': ns_rev_conn, 'sender': switch,
        }
    )
    sim.schedule.reset_mock()


def test_network_switch_forwards_packets_received_from_network_interfaces():
    """Validate packet with destination matching one interface is routed up.

    We send three packets: one from 'eth', one from 'wifi' and one from 'user'.
    Make sure that in any case the packet is routed to user.
    """
    sim, ns, wifi, eth = Mock(), Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    eth.address = 1
    eth_rev_conn = Mock()
    eth.connections.set = Mock(return_value=eth_rev_conn)

    wifi.address = 20
    wifi_rev_conn = Mock()
    wifi.connections.set = Mock(return_value=wifi_rev_conn)

    switch.connections.set('user', ns, reverse=False)
    switch.connections.set('eth', eth, rname='network')
    wifi_conn = switch.connections.set('wifi', wifi, rname='network')

    switch.table.add(10, connection='eth', next_hop=2)
    switch.table.add(30, connection='wifi', next_hop=23)

    pkt_1 = NetworkPacket(destination_address=10, originator_address=5, osn=8)
    pkt_2 = NetworkPacket(destination_address=30, originator_address=17, osn=4)

    switch.handle_message(pkt_1, connection=wifi, sender=wifi_conn)
    sim.schedule.assert_called_once_with(
        0, eth.handle_message, args=(pkt_1,), kwargs={
            'connection': eth_rev_conn, 'sender': switch,
        }
    )
    sim.schedule.reset_mock()

    switch.handle_message(pkt_2, connection=wifi, sender=wifi_conn)
    sim.schedule.assert_called_once_with(
        0, wifi.handle_message, args=(pkt_2,), kwargs={
            'connection': wifi_rev_conn, 'sender': switch,
        }
    )


def test_network_switch_ignores_old_messages():
    """Validate `NetworkSwitch` ignores messages with old SSN.
    """
    sim, ns, iface = Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    iface.address = 1
    iface_rev_conn = Mock()
    iface.connections.set = Mock(return_value=iface_rev_conn)

    ns_rev_conn = Mock()
    ns.connections.set = Mock(return_value=ns_rev_conn)

    switch.connections.set('user', ns, rname='network')
    iface_conn = switch.connections.set('iface', iface, rname='network')

    switch.table.add(10, connection='iface', next_hop=2)

    pkt_1 = NetworkPacket(destination_address=10, originator_address=13, osn=8)
    pkt_2 = NetworkPacket(destination_address=10, originator_address=13, osn=8)  # the same SSN
    pkt_3 = NetworkPacket(destination_address=1, originator_address=13, osn=5)   # older SSN, to sink
    pkt_4 = NetworkPacket(destination_address=1, originator_address=13, osn=9)   # New one!
    pkt_5 = NetworkPacket(destination_address=10, originator_address=13, osn=8)  # again old one

    switch.handle_message(pkt_1, connection=iface_conn, sender=iface)
    sim.schedule.assert_called_once_with(
        0, iface.handle_message, args=(pkt_1,), kwargs={
            'connection': iface_rev_conn, 'sender': switch,
        }
    )
    sim.schedule.reset_mock()

    switch.handle_message(pkt_2, connection=iface_conn, sender=iface)
    sim.schedule.assert_not_called()

    switch.handle_message(pkt_3, connection=iface_conn, sender=iface)
    sim.schedule.assert_not_called()

    switch.handle_message(pkt_4, connection=iface_conn, sender=iface)
    sim.schedule.assert_called_once_with(
        0, ns.handle_message, args=(pkt_4,), kwargs={
            'connection': ns_rev_conn, 'sender': switch,
        }
    )
    sim.schedule.reset_mock()

    switch.handle_message(pkt_5, connection=iface_conn, sender=iface)
    sim.schedule.assert_not_called()


def test_network_switch_updates_addresses_when_forwarding_packet():
    """Validate sender and receiver addresses are upon forwarding.
    """
    sim, ns, wifi, eth = Mock(), Mock(), Mock(), Mock()
    switch = NetworkSwitch(sim)

    eth.address = 7
    wifi.address = 199
    wifi_rev_conn = Mock()
    wifi.connections.set = Mock(return_value=wifi_rev_conn)

    switch.connections.set('user', ns, reverse=False)
    switch.connections.set('wifi', wifi, rname='network')
    eth_conn = switch.connections.set('eth', eth, reverse=False)

    switch.table.add(230, 'wifi', 205)

    pkt = NetworkPacket(destination_address=230, originator_address=5, sender_address=6, receiver_address=7, osn=8)

    switch.handle_message(pkt, connection=eth_conn, sender=eth)
    sim.schedule.assert_called_once_with(
        0, wifi.handle_message, args=(pkt,), kwargs={
            'connection': wifi_rev_conn, 'sender': switch,
        }
    )

    # These fields are expected to be updated
    assert pkt.sender_address == 199
    assert pkt.receiver_address == 205

    # These fields should be kept:
    assert pkt.osn == 8
    assert pkt.originator_address == 5
    assert pkt.destination_address == 230
