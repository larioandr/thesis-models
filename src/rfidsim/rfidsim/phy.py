import numpy as np
from enum import Enum

import pyons
from pyons import Entity

from . import protocol as gen2
from . import vectors
from . import journal
from . import pyradise


SPEED_OF_LIGHT = 299792458.0    # meters per second
THERMAL_NOISE = -114.0

INIT_TRANSCEIVERS_STAGE = (3, 'Transceiver Created')
INIT_SIGNAL_STAGE = (4, "Signal created")
FINISH_TRANSCEIVERS_STAGE = (2, 'Transceiver Finished')


#######################################################################
# Model: NODE and ANTENNA - interfaces and parts for Reader and Tag
#
# Node provides an abstract interface for the reader and tag classes.
#######################################################################
class NodeType(Enum):
    TAG_NODE = 0
    READER_NODE = 1


class Node(Entity):
    """
    Node is an abstract base class for readers and tags. It provides
    accessors for antenna and transceiver, and also provides callback
    API to the transceiver, like ``rx_end()``, ``tx_end()``,
    ``rx_begin()``.

    Each node also must have a name, which is returned by ``__str__()``
    method by default.
    """
    def __init__(self):
        super().__init__()

    @property
    def turned_on(self):
        raise NotImplementedError()

    def receive_started(self): raise NotImplementedError()

    def receive_finished(self, frame, rx_power, snr=None, ber=None):
        raise NotImplementedError()

    def receive_error(self, snr, ber):
        raise NotImplementedError()

    def receive_collision(self, n_replies):
        raise NotImplementedError()

    def send_finished(self): raise NotImplementedError()

    @property
    def name(self): raise NotImplementedError()

    @name.setter
    def name(self, value): raise pyons.errors.NotSupportedError()

    @property
    def antenna(self): raise NotImplementedError()

    @antenna.setter
    def antenna(self, value): raise pyons.errors.NotSupportedError()

    @property
    def transceiver(self): raise NotImplementedError()

    @transceiver.setter
    def transceiver(self, value): raise pyons.errors.NotSupportedError()

    @property
    def speed(self): raise NotImplementedError()

    @speed.setter
    def speed(self, value): raise pyons.errors.NotSupportedError()

    @property
    def direction(self): raise NotImplementedError()

    @direction.setter
    def direction(self, value): raise pyons.errors.NotSupportedError()

    @property
    def velocity(self): return self.direction * self.speed

    @velocity.setter
    def velocity(self, value): raise pyons.errors.NotSupportedError()

    @property
    def node_type(self): raise NotImplementedError()

    @property
    def last_powered_on(self): raise NotImplementedError()

    @last_powered_on.setter
    def last_powered_on(self, value): raise pyons.errors.NotSupportedError()

    def __str__(self): return self.name


class Antenna(object):
    """
    An abstract base class for tag and reader antennas.
    """
    def __init__(self, rp, gain, polarization, node):
        super().__init__()
        self.rp = rp
        self.gain = gain
        self.polarization = polarization
        self.node = node

    @property
    def cable_loss(self): raise NotImplementedError()

    @cable_loss.setter
    def cable_loss(self, value):
        raise NotImplemented("cable_loss setter not supported for {}"
                             "".format(type(self)))

    @property
    def position(self): raise NotImplementedError()

    @position.setter
    def position(self, value):
        raise NotImplemented("position setter not supported for {}"
                             "".format(type(self)))

    @property
    def dir_forward(self): raise NotImplementedError()

    @dir_forward.setter
    def dir_forward(self, value):
        raise NotImplemented("dir_forward setter not supported for {}"
                             "".format(type(self)))

    @property
    def dir_right(self): raise NotImplementedError()

    @dir_right.setter
    def dir_right(self, value):
        raise NotImplemented("dir_right setter not supported for {}"
                             "".format(type(self)))

    def get_polarization_loss(self, sender_polarization):
        if self.polarization == sender_polarization:
            return 0.0
        else:
            return -3.0

    def _get_name(self):
        return self.node.name + ".antenna"

    def __str__(self):
        return self._get_name() + \
               "\n\tposition={a.position}" \
               "\n\tdir_forward={a.dir_forward}" \
               "\n\tdir_right={a.dir_right}" \
               "\n\tpolarization={a.polarization}" \
               "\n\tcable_loss={a.cable_loss}" \
               "\n\tradiation_pattern={a.rp}" \
               "\n\tgain={a.gain}".format(a=self)


class TagAntenna(Antenna):
    def __init__(self, rp=pyradise.isotropic_rp, gain=0.0,
                 polarization=1.0, node=None):
        super().__init__(rp=rp, gain=gain, polarization=polarization,
                         node=node)

    @property
    def cable_loss(self):
        return 0.0

    @property
    def position(self):
        return self.node.position

    @property
    def dir_forward(self):
        assert hasattr(self.node, 'orientation')
        return self.node.orientation

    @property
    def dir_right(self):
        assert hasattr(self.node, 'orientation')
        assert hasattr(self.node, 'up_direction')
        return np.cross(self.node.up_direction, self.node.orientation)


class ReaderAntenna(Antenna):
    def __init__(self, position=(0, 0, 0), dir_forward=(1, 0, 0),
                 dir_right=(0, 1, 0), rp=pyradise.isotropic_rp,
                 gain=0.0, cable_loss=-1.0, polarization=0.5, node=None,
                 lane=1, side='front'):
        super().__init__(rp=rp, gain=gain, polarization=polarization,
                         node=node)
        self._position = vectors.vec3(position)
        self._cable_loss = cable_loss
        self._dir_forward = vectors.normalize(dir_forward)
        self._dir_right = vectors.normalize(dir_right)
        self.lane = lane
        self.side = side

    @property
    def cable_loss(self):
        return self._cable_loss

    @cable_loss.setter
    def cable_loss(self, value):
        self._cable_loss = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = vectors.vec3(value)

    @property
    def dir_forward(self):
        return self._dir_forward

    @dir_forward.setter
    def dir_forward(self, value):
        self._dir_forward = vectors.normalize(value)

    @property
    def dir_right(self):
        return self._dir_right

    @dir_right.setter
    def dir_right(self, value):
        self._dir_right = vectors.normalize(value)

    def __str__(self):
        base = super().__str__()
        return base + "\n\tlane={a.lane}" \
                      "\n\tside={a.side}".format(a=self)


#######################################################################
# Model: TRANSCEIVER and DECIDERS
#
# Transceiver is a part of each Reader and Tag. It carries emits and
# receives signals, decides whether the signal was received and
# provides its owner with received frame.
#
# Required owner API:
# + owner.receive(frame) - called when the frame successfully received
# + owner.tx_end() - called when the signal transmission finishes
# + owner.model - a Model object
# + owner.modulation_loss
#
# Provided API:
# + send(frame) - called when the owner wants to send a frame
# + set_power(power) - called when the owner finds that its
#       transmitted power is updated
# + clear() - called when the owner is being removed, cancels signals
# + start_receive(signal) - called by the Signal when it reaches the
#       transceiver
# + finish_receive(signal) - called by the Signal when its last symbol
#       reaches the transceiver
# + cancel_receive(signal) - removes the signal from
#
# Events:
# + ('tx end', Transceiver) - end of transmission timeout
#######################################################################
#
class Decider(object):
    class ResultCode(Enum):
        OK = 0
        RECEIVE_ERROR = 1
        BROKEN = 2
        COLLISION = 3

    class Result(object):
        def __init__(self, code, snr=None, ber=None, n_rx_signals=0):
            self.code = code
            self.snr = snr
            self.ber = ber
            self.n_rx_signals = n_rx_signals

        def __str__(self):
            s_snr = "{:.2f}".format(self.snr) if self.snr is not None else "-"
            s_ber = "{:.2f}".format(self.ber) if self.ber is not None else "-"
            return "code:{}, SNR={}, BER={}, n_rx:{}".format(
                self.code, s_snr, s_ber, self.n_rx_signals)

    def __init__(self, transceiver, channel, ber_model):
        self.transceiver = transceiver
        self.channel = channel
        self.ber_model = ber_model

    def decide(self, signal, rx_signals): raise NotImplementedError()

    @property
    def name(self):
        return self.transceiver.name + ".decider"


class ReaderDecider(Decider):
    def __init__(self, transceiver=None, channel=None, ber_model=None):
        super().__init__(transceiver, channel, ber_model)

    def get_snr(self, rx_power, m, preamble_duration, blf):
        if (rx_power is None or m is None or preamble_duration is None or
                blf is None):
            return None

        noise = (pyradise.dbm2w(self.transceiver.node.circulator_noise) +
                 pyradise.dbm2w(self.channel.thermal_noise))
        noise = pyradise.w2dbm(noise)
        raw_snr = pyradise.signal2noise(
            rx_power=rx_power, noise_power=noise)
        sync = pyradise.sync_angle(
            snr=raw_snr, preamble_duration=preamble_duration)
        snr = pyradise.snr_extended(
            snr=raw_snr, sync_phi=sync, miller=m.value,
            symbol_duration=1.0 / blf)
        return snr

    def get_ber(self, snr):
        if snr is None:
            return 1.0
        return self.ber_model(snr=snr)

    def decide(self, signal, rx_signals):
        assert isinstance(signal.frame, gen2.TagFrame)
        assert signal in rx_signals

        if len(rx_signals) > 1:
            # pyons.fine("collision!", sender="ReaderDecider")
            # for simplicity consider that any collision causes signal loss
            return Decider.Result(
                Decider.ResultCode.COLLISION, len(rx_signals))
        if signal.broken:
            # pyons.fine("broken signal", sender="ReaderDecider")
            # if signal was broken (tag was turning off during TXOP),
            # signal is also lost
            return Decider.Result(Decider.ResultCode.BROKEN)
        if self.ber_model is not None:
            rx_power = min([power for time, power in signal.received_power
                            if power is not None])
            m = signal.frame.m
            preamble_duration = signal.frame.preamble_duration
            blf = signal.frame.blf
            snr = self.get_snr(rx_power, m, preamble_duration, blf)
            ber = self.get_ber(snr)
            # pyons.fine("RX={:3f}dBm SNR={:3f} BER={:.3f} receiver={} sender={}"
            #            "".format(rx_power, snr, ber,
            #                      signal.receiver.node.antenna.position,
            #                      signal.sender.node.antenna.position),
            #            sender=self.name)
        else:
            snr = None
            ber = 0.0
        probability = (1.0 - ber) ** signal.frame.body_bitlen
        received = (np.random.uniform() <= probability)
        result_code = (Decider.ResultCode.OK if received
                       else Decider.ResultCode.RECEIVE_ERROR)

        rec = journal.FrameBERRecord()
        rec.ber = ber
        rec.result = 1 if received else 0
        rec.reader_lane = signal.receiver.node.antenna.lane
        rec.reader_side = signal.receiver.node.antenna.side
        rec.reader_position = np.array(signal.receiver.node.antenna.position)
        rec.tag_lane = signal.sender.node.lane
        rec.tag_side = signal.sender.node.location
        rec.tag_position = np.array(signal.sender.node.antenna.position)
        rec.frame_bitlen = signal.frame.body_bitlen
        rec.probability = probability
        journal.Journal().write_ber_journal(rec)

        result = Decider.Result(result_code, snr, ber, len(rx_signals))
        return result


class TagDecider(Decider):
    def __init__(self, node=None, channel=None):
        super().__init__(node, channel, ber_model=None)

    def decide(self, signal, rx_signals):
        if signal.broken:
            result_code = Decider.ResultCode.BROKEN
        elif len(rx_signals) > 1:
            result_code = Decider.ResultCode.COLLISION
        else:
            result_code = Decider.ResultCode.OK
        return Decider.Result(result_code, None, None, len(rx_signals))


#######################################################################
# TRANSCEIVER
#######################################################################

class TransceiverType(Enum):
    PASSIVE = 0
    ACTIVE = 1


class Transceiver(Entity):
    def __init__(self, node=None, decider=None, channel=None,
                 modulation_loss=0.0):
        super().__init__()
        self.node = node
        self.decider = decider
        self.channel = channel
        self.modulation_loss = modulation_loss
        self._power = None
        self._tx_signals = []  # Signals being transmitted
        self._tx_end_timeout = None  # timeout ID
        self._rx_signals = []  # Signals being received
        self._num_rx_signals = 0  # this is to simplify running RXOPs tracking
        self._first_decided = None

    @property
    def name(self):
        return self.node.name + ".radio"

    @property
    def transceiver_type(self):
        if self.node.node_type is NodeType.READER_NODE:
            return TransceiverType.ACTIVE
        else:
            return TransceiverType.PASSIVE

    @property
    def power(self):
        return self._power

    def set_power(self, power):
        self._power = power
        for signal in self._tx_signals:
            signal.update_transmitter_power(power)

    def clear(self):
        for signal in self._tx_signals:
            signal.cancel()     # removes from the receiver, terminates
        self._tx_signals.clear()
        while self._num_rx_signals > 0:
            self._rx_signals[0].cancel()    # the signal will be removed here
        if self._tx_end_timeout is not None:
            pyons.cancel(self._tx_end_timeout)
            self._tx_end_timeout = None
        self._first_decided = None

    @Entity.managed
    def send(self, frame):
        assert isinstance(self.channel, Channel)
        if len(self._tx_signals) > 0:
            raise RuntimeError("send() while another TX is running")
        peers = self.channel.get_peers(self)
        for peer in peers:
            if peer.node.turned_on:
                signal = Signal(sender=self, receiver=peer, frame=frame,
                                power=self.power, channel=self.channel)
                self._tx_signals.append(signal)
                pyons.add_entity(signal)
        self._tx_end_timeout = pyons.create_timeout(frame.duration, 'tx-end')
        # If start sending, nothing would be received anyway
        for signal in self._rx_signals:
            if signal.state != Signal.State.FINISHED:
                signal.broken = True

    @Entity.eventhandler(lambda ev, src: ev == 'tx-end')
    def _handle_send_finished(self, event, source):
        assert event == 'tx-end' and source is self
        self._tx_end_timeout = None
        self._tx_signals.clear()
        self.node.send_finished()

    def start_receive(self, signal):
        if not self._rx_signals:
            self._first_decided = None
        self._rx_signals.append(signal)
        self._num_rx_signals += 1
        self.node.receive_started()

    def finish_receive(self, signal):
        if signal not in self._rx_signals:
            return  # nothing should be done if signal is alien

        decided = self.decider.decide(signal, self._rx_signals)
        assert isinstance(decided, Decider.Result)
        if self._first_decided is None:
            self._first_decided = decided

        # Check that all signals are finished
        self._num_rx_signals -= 1
        if self._num_rx_signals == 0:
            for s in self._rx_signals:
                assert s.state == Signal.State.FINISHED
                pyons.remove_entity(s)
            # If all signals are finished, they can be removed
            self._rx_signals.clear()

        if decided.code is Decider.ResultCode.OK:
            power = min([power for time, power in signal.received_power
                         if power is not None])
            self.node.receive_finished(signal.frame, power,
                                       decided.snr, decided.ber)
        elif self._num_rx_signals == 0:
            code = self._first_decided.code
            snr = self._first_decided.snr
            ber = self._first_decided.ber
            n_rx_signals = self._first_decided.n_rx_signals
            if code is Decider.ResultCode.OK:
                print("[!!!] current decided = {}".format(decided))
                print("[!!!] first decided   = {}".format(self._first_decided))
                assert False
            if code is Decider.ResultCode.COLLISION:
                self.node.receive_collision(n_rx_signals)
            else:
                self.node.receive_error(snr, ber)
            self._first_decided = None
            self.node.receive_finished(None, None, None, None)

    def cancel_receive(self, signal):
        if signal in self._rx_signals:
            self._rx_signals.remove(signal)
            self._num_rx_signals -= 1

    @Entity.initializer(stage=INIT_TRANSCEIVERS_STAGE)
    def _handle_transceiver_started(self):
        assert isinstance(self.channel, Channel)
        assert isinstance(self.node, Node)
        if self.transceiver_type is TransceiverType.ACTIVE:
            self.channel.set_active_transceiver(self)
        else:
            self.channel.add_passive_transceiver(self)

    @Entity.finalizer(stage=FINISH_TRANSCEIVERS_STAGE)
    def _handle_transceiver_finished(self):
        assert isinstance(self.channel, Channel)
        assert isinstance(self.node, Node)
        if self.transceiver_type is TransceiverType.ACTIVE:
            self.channel.set_active_transceiver(None)
        else:
            self.channel.remove_passive_transceiver(self)


#######################################################################
# Model: SIGNAL
#
# Signals are used to describe transmissions from reader and from tags.
# It helps to track power updates.
#
# Events:
# + ('signal begin', Signal) - fired when the first symbol reaches
#       the receiver
# + ('signal end', Signal) - fired when the last symbol reaches the
#       receiver
#######################################################################

class Signal(Entity):
    class State(Enum):
        INIT = 0
        STARTED = 1
        RECEIVING = 2
        FINISHED = 3
        TERMINATED = 4

    def __init__(self, sender, receiver, frame, power, channel):
        super().__init__()
        assert isinstance(sender, Transceiver)
        assert isinstance(receiver, Transceiver)
        self._sender = sender
        self._receiver = receiver
        self._frame = frame
        self._channel = channel
        self._power = power

        self._state = Signal.State.INIT
        self._tx_power = []
        self._rx_power = []
        self._started_at = None

        self.broken = False

        self._begin_timeout_id = None
        self._end_timeout_id = None

    @property
    def sender(self):
        return self._sender

    @property
    def receiver(self):
        return self._receiver

    @property
    def frame(self):
        return self._frame

    @property
    def channel(self):
        return self._channel

    @property
    def initial_power(self):
        return self._power

    @property
    def state(self):
        return self._state

    @property
    def transmitted_power(self):
        return self._tx_power

    @property
    def received_power(self):
        return self._rx_power

    def update_transmitter_power(self, power):
        self._tx_power.append((pyons.time(), power))
        received_power = self.channel.get_rx_power(
            sender=self.sender, receiver=self.receiver, tx_power=power)
        self._rx_power.append((pyons.time(), received_power))
        if power is None or received_power is None:
            self.broken = True

    def terminate(self):
        self._state = Signal.State.TERMINATED

    def cancel(self):
        if self._begin_timeout_id is not None:
            pyons.cancel(self._begin_timeout_id)
            self._begin_timeout_id = None

        if self._end_timeout_id is not None:
            pyons.cancel(self._end_timeout_id)
            self._end_timeout_id = None

        self.receiver.cancel_receive(self)
        self._state = Signal.State.TERMINATED

    @Entity.initializer(stage=INIT_SIGNAL_STAGE)
    def _signal_transmission_started(self):
        if self._state != Signal.State.INIT:
            return

        if self.sender is None:
            raise pyons.errors.MissingFieldError(
                self.__class__.__name__, 'sender')
        if self.receiver is None:
            raise pyons.errors.MissingFieldError(
                self.__class__.__name__, 'receiver')
        if self.frame is None:
            raise pyons.errors.MissingFieldError(
                self.__class__.__name__, 'frame')

        self.update_transmitter_power(self.initial_power)
        distance = vectors.length(self.receiver.node.antenna.position -
                                  self.sender.node.antenna.position)
        delay = distance / SPEED_OF_LIGHT
        duration = self.frame.duration
        self._begin_timeout_id = pyons.create_timeout(delay, 'rx-begin')
        self._end_timeout_id = pyons.create_timeout(duration + delay, 'rx-end')
        self._state = Signal.State.STARTED

        # pyons.debug(
        #     "started TX {}->{}, frame:{}, delay:{:.2f}us, duration:{:.2f}us"
        #     "".format(
        #         self.sender.node.name, self.receiver.node.name,
        #         self.frame, delay * 1e6, duration * 1e6),
        #     sender="Signal")

    @Entity.eventhandler(lambda ev, src: ev == 'rx-begin')
    def _handle_receive_begin(self, event, source):
        assert event == 'rx-begin' and source is self
        self._begin_timeout_id = None
        self._state = Signal.State.RECEIVING
        self.receiver.start_receive(self)
        # pyons.debug("started RX {}->{}, frame:{}".format(
        #     self.sender.node.name, self.receiver.node.name, self.frame),
        #     sender="Signal")

    @Entity.eventhandler(lambda ev, src: ev == 'rx-end')
    def _handle_receive_end(self, event, source):
        assert event == 'rx-end' and source is self
        self._end_timeout_id = None
        self._state = Signal.State.FINISHED
        self.receiver.finish_receive(self)
        # pyons.debug("finished RX {}->{}, frame:{}".format(
        #     self.sender.node.name, self.receiver.node.name, self.frame),
        #     sender="Signal")

    @Entity.death_condition()
    def _check_terminated(self):
        return self._state is Signal.State.TERMINATED

    def __str__(self):
        return "Signal{{sender={} receiver={} frame={}}}".format(
            self.sender.node.name, self.receiver.node.name, self.frame)


#######################################################################
# Model: CHANNEL
#
# Expected values for permittivity is in (3, 30), 70 for water,
# conductivity - in 0.00014 ... 0.15 (5 for water).
#######################################################################

class ChannelDescriptor(object):
    def __init__(self):
        super().__init__()
        self.thermal_noise = THERMAL_NOISE
        self.ground_permittivity = 15.0
        self.ground_conductivity = 3e-2
        self.use_doppler = True


class Channel(Entity):

    def __init__(self, descriptor):
        super().__init__()
        assert isinstance(descriptor, ChannelDescriptor)
        self.thermal_noise = descriptor.thermal_noise
        self.ground_permittivity = descriptor.ground_permittivity
        self.ground_conductivity = descriptor.ground_conductivity
        self.use_doppler = descriptor.use_doppler

        self.path_loss_model = pyradise.two_ray_path_loss
        self.ground_reflection = pyradise.reflection

        self._active_transceiver = None
        self._passive_transceivers = []

    @Entity.initializer(stage=(0, 'Channel Initialization'))
    def _initialize(self):
        ci = journal.ChannelInfoRecord()
        ci.thermal_noise = self.thermal_noise
        ci.permittivity = self.ground_permittivity
        ci.conductivity = self.ground_conductivity
        ci.use_doppler = self.use_doppler
        journal.Journal().write_channel_info(ci)

    @property
    def active_transceiver(self):
        return self._active_transceiver
    
    @property
    def passive_transceivers(self):
        return list(self._passive_transceivers)

    def set_active_transceiver(self, transceiver):
        self._active_transceiver = transceiver

    def add_passive_transceiver(self, transceiver):
        self._passive_transceivers.append(transceiver)

    def remove_passive_transceiver(self, transceiver):
        if transceiver in self._passive_transceivers:
            self._passive_transceivers.remove(transceiver)

    def get_peers(self, transceiver):
        if transceiver.transceiver_type is TransceiverType.ACTIVE:
            return list(self._passive_transceivers)
        else:
            return [self._active_transceiver]

    def get_channel_lifetime(self, default=0.0):
        if self.use_doppler:
            if self.active_transceiver is None:
                return default
            elif self.active_transceiver.node is None:
                return default
            else:
                last_powered_on = self.active_transceiver.node.last_powered_on
                if last_powered_on is None or last_powered_on < 0:
                    return default
                else:
                    return pyons.time() - last_powered_on
        else:
            return default

    def get_path_loss(self, sender, receiver):
        tx_antenna = sender.node.antenna
        rx_antenna = receiver.node.antenna
        assert isinstance(tx_antenna, Antenna)
        assert isinstance(rx_antenna, Antenna)

        if self.use_doppler:
            sender_velocity = sender.node.velocity
            receiver_velocity = receiver.node.velocity
        else:
            sender_velocity = np.array((0, 0, 0))
            receiver_velocity = np.array((0, 0, 0))

        channel_lifetime = self.get_channel_lifetime()

        # Compute path loss in linear scale
        pl = pyradise.two_ray_path_loss_3d(
            time=channel_lifetime,
            wavelen=self.active_transceiver.node.wavelen,
            tx_pos=tx_antenna.position,
            tx_dir_theta=tx_antenna.dir_forward,
            tx_dir_phi=tx_antenna.dir_right,
            tx_rp=tx_antenna.rp,
            tx_velocity=sender_velocity,
            rx_pos=rx_antenna.position,
            rx_dir_theta=rx_antenna.dir_forward,
            rx_dir_phi=rx_antenna.dir_right,
            rx_rp=rx_antenna.rp,
            rx_velocity=receiver_velocity,
            ground_reflection=self.ground_reflection,
            permittivity=self.ground_permittivity,
            conductivity=self.ground_conductivity,
            polarization=tx_antenna.polarization)

        # print("PATH-LOSS {dir} {pl} : time={time} wavelen={wavelen} "
        #       "tx_pos={tx_antenna.position} "
        #       "tx_dir_theta={tx_antenna.dir_forward} "
        #       "tx_dir_phi={tx_antenna.dir_right} "
        #       "tx_velocity={sender_velocity} "
        #       "rx_pos={rx_antenna.position} "
        #       "rx_dir_theta={rx_antenna.dir_forward} "
        #       "rx_dir_phi={rx_antenna.dir_right} "
        #       "rx_velocity={receiver_velocity} "
        #       "permittivity={self.ground_permittivity} "
        #       "conductivity={self.ground_conductivity} "
        #       "polarization={tx_antenna.polarization} "
        #       "".format(pl=pyradise.lin2db(pl),
        #                 dir=("R=>T" if isinstance(tx_antenna, ReaderAntenna)
        #                      else "T=>R"),
        #                 time=channel_lifetime,
        #                 wavelen=self.active_transceiver.node.wavelen,
        #                 tx_antenna=tx_antenna,
        #                 sender_velocity=sender_velocity,
        #                 rx_antenna=rx_antenna,
        #                 receiver_velocity=receiver_velocity,
        #                 self=self))

        return pyradise.lin2db(pl)

    def get_rx_power(self, sender, receiver, path_loss=None, tx_power=None):
        if tx_power is None and sender.power is None:
            return None

        if tx_power is None:
            tx_power = sender.power
        #
        # (simplification: should be carried out with radiation pattern,
        # but for now...) Assume that if sender and receiver antennas
        # "look" at the same side, no power is received
        #
        r1 = sender.node.antenna.position
        o1 = sender.node.antenna.dir_forward
        r2 = receiver.node.antenna.position
        o2 = receiver.node.antenna.dir_forward
        if np.dot(r2 - r1, o1) < 0 or np.dot(r1 - r2, o2) < 0:
            return THERMAL_NOISE

        if path_loss is None:
            path_loss = self.get_path_loss(sender=sender, receiver=receiver)

        polarization_loss = receiver.node.antenna.get_polarization_loss(
            sender.node.antenna.polarization)

        rx_power = (tx_power + sender.modulation_loss +
                    sender.node.antenna.cable_loss + sender.node.antenna.gain +
                    path_loss + receiver.node.antenna.gain + polarization_loss)

        # print("RX-POWER {dir} {rx_power} : sender.x={sender_x} "
        #       "receiver.x={receiver_x} t={t} tx_power={tx_power} "
        #       "modulation_loss={modulation_loss} "
        #       "cable_loss={cable_loss} tx_gain={tx_gain} "
        #       "PL={pl} rx_gain={rx_gain} "
        #       "polarization_loss={polarization_loss}".format(
        #         dir=("R=>T" if isinstance(sender.node.antenna, ReaderAntenna)
        #              else "T=>R"),
        #         sender_x=sender.node.antenna.position[0],
        #         receiver_x=receiver.node.antenna.position[0],
        #         t=(pyons.time() - self.active_transceiver.node.last_powered_on
        #            if self.use_doppler else 0.0),
        #         rx_power=rx_power, tx_power=tx_power,
        #         modulation_loss=sender.modulation_loss,
        #         cable_loss=sender.node.antenna.cable_loss,
        #         tx_gain=sender.node.antenna.gain,
        #         pl=path_loss, rx_gain=receiver.node.antenna.gain,
        #         polarization_loss=polarization_loss))

        return rx_power
