from pydesim import Model

from pycsmaca.utilities import ReadOnlyDict


class NetworkPacket:
    """NetworkPacket is a message that is being used on the network layer.

    It introduces four addresses:

    - destination address: address of the interface the packet is destined to
        (taken from e.g. `AppData`);

    - originator address - address of the interface that originated the packet,
        i.e. sent it for the first time;

    - sender address - address of the interface that last sent the packet;

    - receiver address - address of the interface that is expected to receive
        the packet in the latest transmission.

    Besides these addresses, `NetworkPacket` stores Originator Sequence Number
    (OSN) that is used to filter old packets. OSNs are recorded per
    `originator_address` of the received or originated packet.
    If `NetworkSwitch` receives a packet with the same or smaller OSN,
    it ignores the message (see `NetworkSwitch` for details).

    `NetworkPacket` can also handle a payload (`data`), which is expected
    to be `AppData`.
    """
    def __init__(
            self, destination_address=None, originator_address=None,
            receiver_address=None, sender_address=None, osn=None, data=None):
        self.destination_address = destination_address
        self.originator_address = originator_address
        self.sender_address = sender_address
        self.receiver_address = receiver_address
        self.osn = osn
        self.data = data

    @property
    def size(self):
        return self.data.size if self.data else 0

    def __str__(self):
        fields = []
        for field, value in [
            ('DST', self.destination_address),
            ('ORIGIN', self.originator_address),
            ('SND', self.sender_address),
            ('RCV', self.receiver_address),
            ('OSN', self.osn)
        ]:
            if value is not None:
                fields.append(f'{field}={value}')
        header = ','.join(fields)
        body = f' | {self.data}' if self.data is not None else ''
        return f'NetPkt{{{header}{body}}}'


class NetworkService(Model):
    """Represents an interface between applications and `NetworkSwitch`.

    This module is aimed at encapsulation and decapsulation of `NetworkPacket`
    and `AppData` messages. During handling the message, it inspects the
    connection the message was received within.

    If the message was received from the user (via `'source'` connection),
    `NetworkService` creates a new `NetworkPacket` and fills its `dst_addr`
    and `data` fields.

    If the message was received from the network (via `'network'` connection),
    it decapsulates the message and send `pkt.data` (which is expected to be
    `AppData` instance) to the application layer via `'sink'` connection.

    Connections:
    - `'network'`: (mandatory) - connects to `NetworkSwitch` module (net layer);
    - `'source'`: (mandatory) - connects to `Source` module (app layer);
    - `'sink'`: (mandatory) - connects to `Sink` module (app layer).

    Connection `'sink'` MAY be unidirectional (from `NetworkService` to `Sink`).
    Other connections MUST be bidirectional.
    """
    def __init__(self, sim):
        super().__init__(sim)

    def handle_message(self, message, connection=None, sender=None):
        if connection == self.connections.get('source'):
            packet = NetworkPacket(
                destination_address=message.destination_address, data=message
            )
            self.connections['network'].send(packet)
        elif connection == self.connections.get('network'):
            self.connections['sink'].send(message.data)

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}NetworkService'


class SwitchTable:
    """Represents network layer routing table.

    Stores routes in the form `destination_address -> Link`, where
    `SwitchTable.Link` has `connection` field and `next_hop` field.

    > IMPORTANT: `connection` is the connections name, not he connection itself.

    Links are added using `add()` method. They later can be grabbed with
    square brackets (like in dictionary).

    Records MAY be updated later during the simulation.
    """
    class Link:
        def __init__(self, connection, next_hop):
            self.connection = connection
            self.next_hop = next_hop

        def as_tuple(self):
            return self.connection, self.next_hop

        def __str__(self):
            return f'conn={self.connection}, next_hop={self.next_hop}'

    def __init__(self):
        self.__records = {}

    def add(self, dst, connection, next_hop):
        self.__records[dst] = SwitchTable.Link(connection, next_hop)

    def as_dict(self):
        return ReadOnlyDict({
            dst: link.as_tuple() for dst, link in self.__records.items()
        })

    def __getitem__(self, dst):
        return self.__records[dst]

    def get(self, dst, default=None):
        return self.__records.get(dst, default)

    def __contains__(self, dst):
        return dst in self.__records

    def __str__(self):
        records = (
            f'{dst}: ({link.connection}, {link.next_hop})'
            for dst, link in self.__records.items()
        )
        return f'SwitchTable{{{", ".join(records)}}}'


class NetworkSwitch(Model):
    """Model of the network switch (router). Right now supports static routes.

    This module performs packet forwarding between connected interfaces,
    user-generated packets forwarding and delivery if the destination address
    matches one of the interface addresses.

    Connections:
    - `'user'` (mandatory, bi-directional): connection to `NetworkService`;
    - any other: connection to the network interface.

    For each packet, this module inspects its routing table (`SwitchTable`).
    If it knows the route to the destination interface, it sets sender address
    of the packet equal to address of the interface the packet will be sent
    from, and sends the packet via `Link.connection`, stored in the routing
    table.

    `NetworkSwitch` also records and checks SSN values. If the packet is too
    old (previous stored value of the SSN is less or equal to the received one),
    the packet is discarded.

    Since packets coming from `NetworkService` originally have only
    `destination_address` and `data` filled, this module also fills `osn` and
    `source_address` before forwarding the packet to any of its network
    interfaces.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.__table = SwitchTable()
        self.__osn_table = {}

    @property
    def table(self):
        return self.__table

    def handle_message(self, message, connection=None, sender=None):
        assert isinstance(message, NetworkPacket)

        # 1) Check source sequence number (SSN):
        # - if the switch never received packets from that originator
        #   (`originator_address`), it fills its OSN table with the `osn` from
        #   the packet and continues serving packet;
        # - if the switch ever received packets from the originator, the stored
        #   OSN is smaller then the received one, it updates stored OSN and
        #   continues processing the message;
        # - if the switch ever received packets from the originator, but the
        #   stored OSN is greater or equal to the received one, it drops the
        #   packet silently and stops serving.
        if message.originator_address is not None:
            assert message.osn is not None
            # Check that this message is not too old by checking its SSN:
            if message.originator_address not in self.__osn_table:
                self.__osn_table[message.originator_address] = message.osn
            elif message.osn <= self.__osn_table[message.originator_address]:
                return  # do not process this message due to old SSN
            else:
                self.__osn_table[message.originator_address] = message.osn

        # 2) By using the destination address, the Switch checks whether
        # ANY of its connected interface has the given address. If such
        # interface found, it means that the message destination is the
        # station the switch is contained in, so it sends the message up to
        # `NetworkService` for decapsulation and sending then it up to a user.
        for _, module in self.connections.as_dict().items():
            if (hasattr(module, 'address') and
                    module.address == message.destination_address):
                self.connections['user'].send(message)
                return

        # 3) If an interface with destination address not found, the switch
        # tries to forward the packet. It looks up its switching table to
        # find a record for the given destination:
        #
        # - if found, switch extracts the connection from the switching record;
        #
        # - if not found, the packet is silently dropped and the forwarding
        #   service is stopped.
        link = self.table.get(message.destination_address)
        if link is None:
            return
        iface_connection = self.connections[link.connection]

        # 4) Now the switch checks whether the packet came from the user
        # (`NetworkService`):
        #
        # - if the packet came from the user, switch fills its originator
        #   address from the interface address found in routing table,
        #   fills `osn` by incrementing the value stored in OSN table
        #   (or selecting the first one, if this is the first packet originated
        #   from that interface);
        #
        # - otherwise, the packet is treated to come from some network
        #   interface. In this case Switch only checks that originator address
        #   and OSN were filled by some another module (typically, originator
        #   switch).
        if connection.name == 'user':
            message.originator_address = iface_connection.module.address

            # Choose, assign and inc SSN for the given source address:
            if message.originator_address not in self.__osn_table:
                self.__osn_table[message.originator_address] = 0
            else:
                self.__osn_table[message.originator_address] += 1
            message.osn = self.__osn_table[message.originator_address]
        else:
            assert message.originator_address is not None
            assert message.osn is not None

        # 5) Finally, the Switch updates receiver and sender addresses,
        # and forwards the message to the proper interface.
        message.receiver_address = link.next_hop
        message.sender_address = iface_connection.module.address
        iface_connection.send(message)
        self.sim.logger.debug(
            f'forward packet {message} from connection {connection.name} '
            f'to {iface_connection.name}', src=self
        )

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent is not None else ''
        return f'{prefix}Switch'
