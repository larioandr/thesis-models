from numpy import inf
from pydesim import Model, Trace, Statistic


class WireFrame:
    def __init__(self, packet, duration=0, header_size=0, preamble=0):
        self.packet = packet
        self.duration = duration
        self.header_size = header_size
        self.preamble = preamble

    @property
    def size(self):
        return self.header_size + (self.packet.size if self.packet else 0)

    def __str__(self):
        fields = ','.join([
            f'D={self.duration}', f'HDR={self.header_size}',
            f'PR={self.preamble}'
        ])
        body = f' | {self.packet}' if self.packet else ''
        return f'WireFrame[{fields}{body}]'


class WiredTransceiver(Model):
    """This module models a simple wired full-duplex transceiver.

    `WiredTransceiver` simulates a simple wired network card. It sends packets
    in `WireFrame`s instances with constant bitrate. Each frame can have
    a preamble and some header. After transmission, `WiredTransceiver` waits
    IFS before sending another frame.

    `WiredTransceiver` receives data from the queue. Upon construction, it
    schedules `_start()` call, which will request data from the queue. Note
    that `'queue'` connection MUST be added right after creation to make this
    work. Upon `_start()` call, `WiredTransceiver` will request packets with
    `queue.get_next(self)` call.

    `WiredTransceiver` requests other packets when TX operation finishes and
    it waits IFS. To request additional packets, it calls `queue.get_next()`.

    `WiredTransceiver` is expected to have the only one peer, which typically
    is another `WiredTransceiver` module. They MUST be connected with `'peer'`
    bi-directional connection. Reception is performed in two steps: when
    the frame reaches the transceiver, it marks RX as busy, and schedules
    frame reception end at `frame.duration`. Upon frame RX end, RX is marked
    as ready.

    If `WiredTransceiver` receives a packet from the `Queue` during another
    TX operation running, it will throw a `RuntimeError` exception.

    `WiredTransceiver` needs three connections:

    - `'peer'`: a bi-directional connection to another transceiver (mandatory);
    - `'queue'`: a bi-directional connection to the packets queue (mandatory);
    - `'up'`: a bi-directional connection to the switch (optional).

    If `'up'` connection is defined, the received packets are sent through it.
    Otherwise, they are silently dropped. Typically, `'up'` connects a
    `NetworkSwitch` module, or parent `NetworkInterface`.
    """
    def __init__(self, sim, bitrate=inf, header_size=0, preamble=0, ifs=0):
        super().__init__(sim)
        self.bitrate = bitrate
        self.header_size = header_size
        self.preamble = preamble
        self.ifs = ifs
        # State variables:
        self.__started = False
        self.__tx_frame = None
        self.__wait_ifs = False
        self.__rx_frame = None
        # Statistics:
        self.__num_received_frames = 0
        self.__num_received_bits = 0
        self.__rx_busy_trace = Trace()
        self.__rx_busy_trace.record(0, 0)
        self.__num_transmitted_packets = 0
        self.__num_transmitted_bits = 0
        self.__tx_busy_trace = Trace()
        self.__tx_busy_trace.record(0, 0)
        self.__service_time = Statistic()
        self.__service_started_at = None
        # Initialization:
        self.sim.schedule(self.sim.stime, self.start)

    @property
    def started(self):
        return self.__started

    @property
    def tx_ready(self):
        return not self.tx_busy

    @property
    def tx_busy(self):
        return self.__tx_frame is not None or self.__wait_ifs

    @property
    def rx_ready(self):
        return self.__rx_frame is None

    @property
    def rx_busy(self):
        return not self.rx_ready

    @property
    def num_received_frames(self):
        return self.__num_received_frames

    @property
    def num_received_bits(self):
        return self.__num_received_bits

    @property
    def rx_busy_trace(self):
        return self.__rx_busy_trace

    @property
    def num_transmitted_packets(self):
        return self.__num_transmitted_packets

    @property
    def num_transmitted_bits(self):
        return self.__num_transmitted_bits

    @property
    def tx_busy_trace(self):
        return self.__tx_busy_trace

    @property
    def service_time(self):
        return self.__service_time

    @property
    def tx_frame(self):
        return self.__tx_frame

    def start(self):
        self.connections['queue'].module.get_next(self)
        self.__started = True

    def handle_message(self, message, connection=None, sender=None):
        if connection.name == 'queue':
            if self.tx_busy:
                raise RuntimeError('new NetworkPacket while another TX running')
            duration = ((self.header_size + message.size) / self.bitrate +
                        self.preamble)
            frame = WireFrame(
                packet=message, duration=duration, header_size=self.header_size,
                preamble=self.preamble
            )
            self.connections['peer'].send(frame)
            self.sim.schedule(duration, self.handle_tx_end)
            self.__tx_frame = frame
            self.__tx_busy_trace.record(self.sim.stime, 1)
            self.__service_started_at = self.sim.stime
            self.sim.logger.debug(f'start transmitting frame {frame}', src=self)
        elif connection.name == 'peer':
            self.sim.schedule(
                message.duration, self.handle_rx_end, args=(message,)
            )
            self.__rx_frame = message
            self.__rx_busy_trace.record(self.sim.stime, 1)
            self.sim.logger.debug(f'start receiving frame {message}', src=self)

    def handle_tx_end(self):
        self.sim.schedule(self.ifs, self.handle_ifs_end)
        # Record statistics:
        self.__num_transmitted_packets += 1
        self.__num_transmitted_bits += self.__tx_frame.size
        # Update state variables:
        self.__wait_ifs = True
        self.__tx_frame = None
        self.sim.logger.debug(f'finish transmitting, waiting IFS', src=self)

    def handle_ifs_end(self):
        self.__wait_ifs = False
        self.connections['queue'].module.get_next(self)
        # Record statistics:
        self.__tx_busy_trace.record(self.sim.stime, 0)
        self.__service_time.append(self.sim.stime - self.__service_started_at)
        self.__service_started_at = None
        self.sim.logger.debug(f'IFS end, ready to transmit', src=self)

    def handle_rx_end(self, frame):
        if 'up' in self.connections:
            self.connections['up'].send(frame.packet)
        self.__rx_frame = None
        self.__num_received_frames += 1
        self.__num_received_bits += frame.size
        self.__rx_busy_trace.record(self.sim.stime, 0)
        self.sim.logger.debug(f'finish receiving frame', src=self)

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}RxTx'


class WiredInterface(Model):
    def __init__(self, sim, address, queue, transceiver):
        super().__init__(sim)
        self.__address = address
        self.__queue = queue
        self.__transceiver = transceiver
        # Making queue and transceiver self children:
        self.children['queue'] = queue
        self.children['transceiver'] = transceiver
        # Establishing internal connections:
        self.connections['_queue'] = self.__queue
        self.connections.set('_receiver', transceiver, rname='up')
        self.connections.set('_peer', transceiver, rname='peer')
        transceiver.connections.set('queue', queue, rname='service')

    @property
    def address(self):
        return self.__address

    @property
    def queue(self):
        return self.__queue

    @property
    def transceiver(self):
        return self.__transceiver

    def handle_message(self, message, connection=None, sender=None):
        if connection.name == 'user':
            self.connections['_queue'].send(message)
        elif connection.name == 'wire':
            self.connections['_peer'].send(message)
        elif connection.name == '_receiver':
            self.connections['user'].send(message)
        elif connection.name == '_peer':
            self.connections['wire'].send(message)

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}Interface({self.address})'
