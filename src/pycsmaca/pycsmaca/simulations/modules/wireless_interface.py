import math
from enum import Enum

from numpy.random.mtrand import randint
from pydesim import Model, Statistic, Trace

from pycsmaca.simulations.modules import NetworkPacket


class PDUBase:
    class Type(Enum):
        DATA = 0
        ACK = 1

    @property
    def size(self):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    @property
    def sender_address(self):
        raise NotImplementedError

    @property
    def receiver_address(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)


class DataPDU(PDUBase):
    def __init__(
            self, packet, header_size, seqn,
            sender_address=None,
            receiver_address=None,
    ):
        assert isinstance(packet, NetworkPacket)
        self.__packet = packet
        self.__sender = (
            sender_address if sender_address is not None
            else packet.sender_address
        )
        self.__receiver = (
            receiver_address if receiver_address is not None
            else packet.receiver_address
        )
        self.__header_size = header_size
        self.__seqn = seqn

    @property
    def packet(self):
        return self.__packet

    @property
    def header_size(self):
        return self.__header_size

    @property
    def size(self):
        return self.header_size + self.packet.size

    @property
    def type(self):
        return self.Type.DATA

    @property
    def sender_address(self):
        return self.__sender

    @property
    def receiver_address(self):
        return self.__receiver

    @property
    def seqn(self):
        return self.__seqn

    def __str__(self):
        link = f'{self.sender_address}=>{self.receiver_address}'
        size = math.ceil(self.size)
        return f'PDU{{{link}, seqn:{self.seqn}, {size:d}b}}'


class AckPDU(PDUBase):
    def __init__(self, header_size, ack_size, sender_address, receiver_address):
        self.__header_size = header_size
        self.__ack_size = ack_size
        self.__sender_address = sender_address
        self.__receiver_address = receiver_address

    @property
    def size(self):
        return self.__header_size + self.__ack_size

    @property
    def type(self):
        return self.Type.ACK

    @property
    def sender_address(self):
        return self.__sender_address

    @property
    def receiver_address(self):
        return self.__receiver_address

    def __str__(self):
        link = f'{self.sender_address}=>{self.receiver_address}'
        size = f'{self.size}'
        return f'ACK{{{link}, {size}b}}'


class ChannelState(Model):
    """Stores the channel state and informs transmitter about updates.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self._is_busy = False

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

    def set_ready(self):
        self._is_busy = False
        self.transmitter.channel_ready()

    def set_busy(self):
        self._is_busy = True
        self.transmitter.channel_busy()

    @property
    def is_busy(self):
        return self._is_busy

    def __str__(self):
        return f'{self.parent}.channel'


class Transmitter(Model):
    """Models transmitter module at MAC layer.

    Connections:
    - `'channel'`: mandatory, to `Channel` instance;
    - `'radio'`: mandatory, to `Radio` module
    - `'queue'`: optional, to `Queue` module
    """
    class State(Enum):
        IDLE = 0
        BUSY = 1
        BACKOFF = 2
        TX = 3
        WAIT_ACK = 4

    def __init__(
            self, sim, address=None, phy_header_size=None, mac_header_size=None,
            ack_size=None, bitrate=None, preamble=None, max_propagation=0,

    ):
        super().__init__(sim)

        # Properties:
        self.__address = address
        self.__phy_header_size = (
            phy_header_size if phy_header_size is not None
            else sim.params.phy_header_size
        )
        self.__mac_header_size = (
            mac_header_size if mac_header_size is not None
            else sim.params.mac_header_size
        )
        self.__bitrate = bitrate if bitrate is not None else sim.params.bitrate
        self.__preamble = (
            preamble if preamble is not None else sim.params.preamble
        )
        self.__max_propagation = max_propagation
        self.__ack_size = (
            ack_size if ack_size is not None else sim.params.ack_size
        )

        # State variables:
        self.timeout = None
        self.cw = 65536
        self.backoff = -1
        self.num_retries = None
        self.pdu = None
        self.__state = Transmitter.State.IDLE
        self.__seqn = 0

        # Statistics:
        self.backoff_vector = Statistic()
        self.__start_service_time = None
        self.service_time = Statistic()
        self.num_sent = 0
        self.num_retries_vector = Statistic()
        self.__busy_trace = Trace()
        self.__busy_trace.record(sim.stime, 0)

        # Initialize:
        sim.schedule(0, self.start)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        if self.state != state:
            self.sim.logger.debug(
                f'{self.state.name} -> {state.name}', src=self
            )
        self.__state = state

    @property
    def address(self):
        return self.__address

    @address.setter
    def address(self, address):
        self.__address = address

    @property
    def phy_header_size(self):
        return self.__phy_header_size

    @property
    def mac_header_size(self):
        return self.__mac_header_size

    @property
    def ack_size(self):
        return self.__ack_size

    @property
    def bitrate(self):
        return self.__bitrate

    @property
    def preamble(self):
        return self.__preamble

    @property
    def max_propagation(self):
        return self.__max_propagation

    @property
    def busy_trace(self):
        return self.__busy_trace

    @property
    def channel(self):
        return self.connections['channel'].module

    @property
    def radio(self):
        return self.connections['radio'].module

    @property
    def queue(self):
        return self.connections['queue'].module

    def start(self):
        self.queue.get_next(self)

    def handle_message(self, packet, connection=None, sender=None):
        if connection.name == 'queue':
            assert self.state == Transmitter.State.IDLE

            self.cw = self.sim.params.cwmin
            self.backoff = randint(0, self.cw)
            self.num_retries = 1

            #
            # Create the PDU:
            #
            self.pdu = DataPDU(
                packet, seqn=self.__seqn,
                header_size=self.phy_header_size + self.mac_header_size,
                sender_address=self.address,
                receiver_address=packet.receiver_address
            )
            self.__seqn += 1

            self.__start_service_time = self.sim.stime
            self.backoff_vector.append(self.backoff)
            self.__busy_trace.record(self.sim.stime, 1)

            self.sim.logger.debug(
                f'backoff={self.backoff}; CW={self.cw},NR={self.num_retries}',
                src=self
            )

            if self.channel.is_busy:
                self.state = Transmitter.State.BUSY
            else:
                self.state = Transmitter.State.BACKOFF
                self.timeout = self.sim.schedule(
                    self.sim.params.difs, self.handle_backoff_timeout
                )
        else:
            raise RuntimeError(
                f'unexpected handle_message({packet}, connection={connection}, '
                f'sender={sender}) call'
            )

    def channel_ready(self):
        if self.state == Transmitter.State.BUSY:
            self.timeout = self.sim.schedule(
                self.sim.params.difs, self.handle_backoff_timeout
            )
            self.state = Transmitter.State.BACKOFF

    def channel_busy(self):
        if self.state == Transmitter.State.BACKOFF:
            self.sim.cancel(self.timeout)
            self.state = Transmitter.State.BUSY

    def finish_transmit(self):
        if self.state == Transmitter.State.TX:
            self.sim.logger.debug('TX finished', src=self)
            ack_duration = (
                (self.ack_size + self.mac_header_size + self.phy_header_size) /
                self.bitrate + self.preamble + 6 * self.max_propagation
            )
            self.timeout = self.sim.schedule(
                self.sim.params.sifs + ack_duration, self.handle_ack_timeout
            )
            self.state = Transmitter.State.WAIT_ACK

    def acknowledged(self):
        if self.state == Transmitter.State.WAIT_ACK:
            self.sim.logger.debug('received ACK', src=self)

            self.sim.cancel(self.timeout)
            self.pdu = None
            self.state = Transmitter.State.IDLE

            self.num_sent += 1
            self.service_time.append(self.sim.stime - self.__start_service_time)
            self.__start_service_time = None
            self.num_retries_vector.append(self.num_retries)
            self.num_retries = None
            self.__busy_trace.record(self.sim.stime, 0)

            #
            # IMPORTANT: Informing the queue that we can handle the next packet
            #
            self.queue.get_next(self)

    def handle_ack_timeout(self):
        assert self.state == Transmitter.State.WAIT_ACK
        self.num_retries += 1
        self.cw = min(2 * self.cw, self.sim.params.cwmax)
        self.backoff = randint(0, self.cw)

        self.backoff_vector.append(self.backoff)

        self.sim.logger.debug(
            f'backoff={self.backoff}; CW={self.cw}, NR={self.num_retries})',
            src=self
        )

        if self.channel.is_busy:
            self.state = Transmitter.State.BUSY
        else:
            self.state = Transmitter.State.BACKOFF
            self.timeout = self.sim.schedule(
                self.sim.params.difs, self.handle_backoff_timeout
            )

    def handle_backoff_timeout(self):
        if self.backoff == 0:
            self.state = Transmitter.State.TX
            self.sim.logger.debug(f'transmitting {self.pdu}', src=self)
            self.radio.transmit(self.pdu)
        else:
            assert self.backoff > 0
            self.backoff -= 1
            self.timeout = self.sim.schedule(
                self.sim.params.slot, self.handle_backoff_timeout
            )
            self.sim.logger.debug(f'backoff := {self.backoff}', src=self)

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}transmitter'


class Receiver(Model):
    """Module simulating MAC (+PHY) DCF receiver.

    Connections:
    - `'radio'`:
    - `'channel'`:
    - `'transmitter'`
    - `'up'`:
    """
    class State(Enum):
        IDLE = 0
        RX = 1
        TX1 = 2
        TX2 = 3
        COLLIDED = 4
        WAIT_SEND_ACK = 5
        SEND_ACK = 6

    def __init__(
            self, sim, address=None, sifs=None, phy_header_size=None,
            ack_size=None,
    ):
        super().__init__(sim)

        # Properties:
        self.__address = address
        self.__sifs = sifs if sifs is not None else sim.params.sifs
        self.__phy_header_size = (
            phy_header_size if phy_header_size is not None
            else sim.params.phy_header_size
        )
        self.__ack_size = (
            ack_size if ack_size is not None else sim.params.ack_size
        )

        # State variables:
        self.__state = Receiver.State.IDLE
        self.__rxbuf = set()
        self.__cur_tx_pdu = None

        # Statistics:
        self.__num_collisions = 0
        self.__num_received = 0
        self.__busy_trace = Trace()
        self.__busy_trace.record(sim.stime, 0)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, state):
        if state != self.__state:
            if self.__state == Receiver.State.IDLE:
                self.__busy_trace.record(self.sim.stime, 1)
            elif state == Receiver.State.IDLE:
                self.__busy_trace.record(self.sim.stime, 0)

            self.sim.logger.debug(
                f'{self.__state.name} -> {state.name}', src=self
            )
            if state is Receiver.State.COLLIDED:
                self.__num_collisions += 1
            self.__state = state

    @property
    def address(self):
        return self.__address

    @address.setter
    def address(self, address):
        self.__address = address

    @property
    def sifs(self):
        return self.__sifs

    @property
    def phy_header_size(self):
        return self.__phy_header_size

    @property
    def ack_size(self):
        return self.__ack_size

    @property
    def num_received(self):
        return self.__num_received

    @property
    def num_collisions(self):
        return self.__num_collisions
    
    @property
    def collision_ratio(self):
        num_ops = self.__num_received + self.__num_collisions
        if num_ops > 0:
            return self.__num_collisions / num_ops
        return 0

    @property
    def busy_trace(self):
        return self.__busy_trace

    @property
    def radio(self):
        return self.connections['radio'].module

    @property
    def channel(self):
        return self.connections['channel'].module

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

    @property
    def up(self):
        return self.connections['up'].module

    @property
    def collision_probability(self):
        if self.num_received > 0 or self.num_collisions > 0:
            return self.num_collisions / (
                    self.num_collisions + self.num_received)
        return 0

    def start_receive(self, pdu):
        if pdu in self.__rxbuf:
            self.sim.logger.error(
                f"PDU {pdu} is already in the buffer:\n{self.__rxbuf}",
                src=self
            )
            raise RuntimeError(f'PDU is already in the buffer, PDU={pdu}')

        if self.state is Receiver.State.IDLE and not self.__rxbuf:
            self.state = Receiver.State.RX
            self.channel.set_busy()

        elif (self.state is Receiver.State.RX or (
                self.state is Receiver.State.IDLE and self.__rxbuf)):
            self.state = Receiver.State.COLLIDED

        self.__rxbuf.add(pdu)

        # In all other states (e.g. TX2, WAIT_SEND_ACK, SEND_ACK) we just add
        # the packet to RX buffer.

    def finish_receive(self, pdu):
        assert pdu in self.__rxbuf
        self.__rxbuf.remove(pdu)

        if self.state is Receiver.State.RX:
            assert not self.__rxbuf
            if pdu.receiver_address == self.address:
                if pdu.type is DataPDU.Type.DATA:
                    self.state = Receiver.State.WAIT_SEND_ACK
                    self.__cur_tx_pdu = pdu
                    self.sim.schedule(self.sifs, self.handle_timeout)
                elif pdu.type is DataPDU.Type.ACK:
                    self.transmitter.acknowledged()
                    self.state = Receiver.State.IDLE
                    self.channel.set_ready()
                else:
                    raise RuntimeError(f'unsupported packet type {pdu.type}')
            else:
                self.state = Receiver.State.IDLE
                self.channel.set_ready()

        elif self.state is Receiver.State.COLLIDED:
            if not self.__rxbuf:
                self.state = Receiver.State.IDLE
                self.channel.set_ready()
            # Otherwise stay in COLLIDED state

        # In all other states (e.g. IDLE, TX1, TX2, ...) we just purge the
        # packet from the RX buffer.

    def start_transmit(self):
        if self.state is Receiver.State.IDLE:
            self.state = Receiver.State.TX1

        elif self.state in (Receiver.State.RX, Receiver.State.COLLIDED):
            self.state = Receiver.State.TX2

        assert self.state is not self.state.WAIT_SEND_ACK

    def finish_transmit(self):
        if self.state is Receiver.State.TX1:
            if self.__rxbuf:
                self.channel.set_busy()
                self.state = Receiver.State.COLLIDED
            else:
                self.state = Receiver.State.IDLE

        elif self.state is Receiver.State.TX2:
            if self.__rxbuf:
                self.state = Receiver.State.COLLIDED
            else:
                self.channel.set_ready()
                self.state = Receiver.State.IDLE

        elif self.state is Receiver.State.SEND_ACK:
            payload = self.__cur_tx_pdu.packet
            self.connections['up'].send(payload)
            self.__num_received += 1
            self.__cur_tx_pdu = None
            self.channel.set_ready()
            self.state = Receiver.State.IDLE

    def handle_timeout(self):
        assert self.state is Receiver.State.WAIT_SEND_ACK
        ack = AckPDU(
            header_size=self.phy_header_size,
            ack_size=self.ack_size,
            sender_address=self.address,
            receiver_address=self.__cur_tx_pdu.sender_address,
        )
        self.state = Receiver.State.SEND_ACK
        self.radio.transmit(ack)

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}receiver'


class WirelessInterface(Model):
    """Represents wireless interface card.

    Connections:
    - `'user'`: bidirectional connection to the interface user (`NetworkSwitch`)
    """
    def __init__(self, sim, address, queue, transmitter, receiver, radio):
        super().__init__(sim)
        self.__address = address
        receiver.address = address
        transmitter.address = address

        # Add children:
        self.children['transmitter'] = transmitter
        self.children['receiver'] = receiver
        self.children['queue'] = queue
        self.children['radio'] = radio
        channel_state = ChannelState(sim)
        self.children['channel_state'] = channel_state

        # Connect children:
        transmitter.connections.set(
            'channel', channel_state, rname='transmitter')
        transmitter.connections.set('queue', queue, rname='service')
        transmitter.connections.set('radio', radio, rname='transmitter')

        receiver.connections.set('channel', channel_state, rname='receiver')
        receiver.connections.set('radio', radio, rname='receiver')
        receiver.connections['transmitter'] = transmitter
        receiver.connections.set('up', self, rname='_receiver')
        queue.connections.set('user', self, rname='_queue')

    @property
    def address(self):
        return self.__address

    @address.setter
    def address(self, address):
        self.__address = address
        self.receiver.address = address
        self.transmitter.address = address

    @property
    def transmitter(self):
        return self.children['transmitter']

    @property
    def receiver(self):
        return self.children['receiver']

    @property
    def queue(self):
        return self.children['queue']

    @property
    def channel_state(self):
        return self.children['channel_state']

    @property
    def radio(self):
        return self.children['radio']

    def handle_message(self, message, connection=None, sender=None):
        if connection.name == 'user':
            self.connections['_queue'].send(message)
        elif connection.name == '_receiver':
            self.connections['user'].send(message)
        else:
            raise RuntimeError(
                f'unexpected message {message} from {sender} '
                f'via "{connection.name}"')

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}Interface({self.address})'
