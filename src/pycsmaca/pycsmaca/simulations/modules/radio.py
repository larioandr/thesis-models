import numpy as np
from numpy.linalg import norm
from pydesim import Model


class AirFrame:
    def __init__(self, pdu, preamble, bitrate):
        self.__pdu = pdu
        self.__preamble = preamble
        self.__bitrate = bitrate

    @property
    def pdu(self):
        return self.__pdu

    @property
    def preamble(self):
        return self.__preamble

    @property
    def bitrate(self):
        return self.__bitrate

    @property
    def duration(self):
        return self.pdu.size / self.bitrate + self.preamble

    def __str__(self):
        return f"Frame[{self.duration:.6f}s with {self.pdu}]"

    def __repr__(self):
        return str(self)


class Radio(Model):
    """Represents radio level of the wireless adapter.

    Connected modules:

    - `ConnectionManager`: `Radio` queries this module for a list of peers.


    Connections:

    - `'receiver'`: a module that receives frames (interface receiver)
    - `'transmitter'`: a module that transmits frames (interface transmitter)

    Parameters:

    - `position`: a 2-D tuple with radio module antenna coordinates.


    Event handlers:

    - `receive(frame)`: fired when the frame first symbol reaches the
        radio of this wireless adapter. Informs `Receiver` about this by
        calling `start_receive(frame.pdu)`.

    - `handle_frame_received(frame)`: fired when the frame is received
        completely. `Radio` doesn't check whether the frame was received
        successfully, it just sends the packet from the frame up to the
        receiver by calling `finish_receive(frame.pdu)`.

    - `handle_frame_transmitted()`: called when the transmission is finished.
        `Radio` informs both `Transmitter` and `Receiver` about this by
        calling `finish_transmit()` (without scheduling, using direct call).


    Methods:

    - `transmit(pdu)`: schedules `receive(frame)` on all peers and
        schedules `handle_frame_transmitted()`. When scheduling
        `receive(frame)`, it computes the propagation delay based on the
        distance between this radio module and the peer using
        `position` property.
    """
    def __init__(
            self, sim, conn_manager, preamble=None, bitrate=None,
            position=(0, 0), connection_radius=None
    ):
        super().__init__(sim)
        self.__connection_manager = conn_manager
        self.__position = np.asarray(position)
        self.__preamble = (
            preamble if preamble is not None else sim.params.preamble
        )
        self.__bitrate = bitrate if bitrate is not None else sim.params.bitrate
        self.__connection_radius = (
            connection_radius if connection_radius is not None
            else sim.params.connection_radius
        )
        # Initialization:
        sim.schedule(0, self._register_at_connection_manager)

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, value):
        assert len(value) == 2
        self.__position = np.asarray(value)

    @property
    def preamble(self):
        return self.__preamble

    @property
    def bitrate(self):
        return self.__bitrate

    @property
    def connection_manager(self):
        return self.__connection_manager

    @property
    def connection_radius(self):
        return self.__connection_radius

    @property
    def receiver(self):
        return self.connections['receiver'].module

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

    def transmit(self, pdu):
        frame = AirFrame(pdu, self.preamble, self.bitrate)
        self.sim.logger.debug(f'transmitting frame: {frame}', src=self)
        peers = self.connection_manager.get_peers(self)
        for peer in peers:
            distance = norm(self.position - peer.position)
            delay = distance / self.sim.params.speed_of_light
            self.sim.schedule(delay, peer.receive, args=(frame,))
        self.sim.schedule(frame.duration, self.handle_frame_transmitted)
        self.receiver.start_transmit()

    def receive(self, frame):
        """This method is called by peers when they send a frame to this radio.
        """
        assert isinstance(frame, AirFrame)
        self.receiver.start_receive(frame.pdu)
        self.sim.schedule(
            frame.duration, self.handle_frame_received, args=(frame,)
        )

    def handle_frame_received(self, frame):
        assert isinstance(frame, AirFrame)
        self.receiver.finish_receive(frame.pdu)

    def handle_frame_transmitted(self):
        self.sim.logger.debug('finished transmit', src=self)
        self.transmitter.finish_transmit()
        self.receiver.finish_transmit()

    def _register_at_connection_manager(self):
        self.__connection_manager.add_radio(self)

    def __str__(self):
        return f'{self.parent}.radio'


class ConnectionManager(Model):
    def __init__(self, sim):
        super().__init__(sim)
        self.connected_radios = {}

    def add_radio(self, radio):
        if radio not in self.connected_radios:
            self.connected_radios[radio] = []

        peers = []
        for peer in self.connected_radios.keys():
            if peer == radio:
                continue
            d = norm(peer.position - radio.position)
            if peer.connection_radius >= d and radio.connection_radius >= d:
                peers.append(peer)
                if radio not in self.connected_radios[peer]:
                    self.connected_radios[peer].append(radio)
                self.sim.logger.debug(
                    f'connected radio@{tuple(radio.position)} to '
                    f'radio@{tuple(peer.position)}',
                    src=self
                )

        self.connected_radios[radio] = peers

    def get_peers(self, radio):
        return self.connected_radios[radio]

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}ConnectionManager'
