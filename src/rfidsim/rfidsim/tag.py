import binascii
from collections import Iterable
from enum import Enum
import numpy as np

import pyons
from pyons import Entity
import phy
import vectors
import protocol as gen2
import journal
import pyradise


INIT_TAG_STAGE = (2, "Init tag")
INIT_MOBILITY_MANAGER = (3, "Init mobility manager")
FINISH_TAG_STAGE = (7, "Finish tag")

UPDATE_POSITION_EVENT = 'update position'


class TagDescriptor(object):
    def __init__(self):
        super().__init__()
        self.identifier = 0
        self.channel = None
        self.position = (0, 0, 0)
        self.direction = (1, 0, 0)
        self.speed = 0.0
        self.orientation = (1, 0, 0)
        self.up_direction = (0, 0, 1)
        self.gain = 0.0
        self.rp = pyradise.isotropic_rp
        self.polarization = 1.0
        self.sensitivity = -18.0
        self.modulation_loss = -3.0
        self.s1_persistence = 0.5
        self.s2_persistence = 2.0
        self.s3_persistence = 2.0
        self.sl_persistence = 2.0
        self.epc = 'FFFFFFFFFFFFFFFFFFFFFFFF'
        self.tid = 'E000FFFFFFFFFFFF'
        self.location = 'front'     # or 'back'


class Tag(phy.Node):
    COMMAND_TIMEOUT_EVENT = 'command timeout'

    class State(Enum):
        OFF = 0
        READY = 1
        ARBITRATE = 2
        REPLY = 3
        ACKNOWLEDGED = 4
        OPEN = 5
        SECURED = 6
        KILLED = 7

    def __init__(self, descriptor):
        super().__init__()

        self.identifier = descriptor.identifier
        self.position = vectors.vec3(descriptor.position)
        # print("descriptor.position={}, self.position={}".format(
        #     descriptor.position, self.position))
        self._speed = descriptor.speed
        self.sensitivity = descriptor.sensitivity
        self.s1_persistence = descriptor.s1_persistence
        self.s2_persistence = descriptor.s2_persistence
        self.s3_persistence = descriptor.s3_persistence
        self.sl_persistence = descriptor.sl_persistence
        self.epc = descriptor.epc
        self.tid = descriptor.tid
        self.channel = descriptor.channel
        self.location = descriptor.location

        self.vehicle = None

        self._direction = vectors.normalize(descriptor.direction)
        self._orientation = vectors.normalize(descriptor.orientation)
        self._up_direction = vectors.normalize(descriptor.up_direction)

        self._antenna = phy.TagAntenna(
            rp=descriptor.rp, gain=descriptor.gain,
            polarization=descriptor.polarization, node=self)

        decider = phy.TagDecider(channel=descriptor.channel)
        self._transceiver = phy.Transceiver(self, decider, descriptor.channel,
                                            descriptor.modulation_loss)
        decider.transceiver = self._transceiver

        self.created_at = None

        self._state = Tag.State.OFF
        self._slot = None
        self._rn = None
        self._sessions = {gen2.Session.S0: gen2.InventoryFlag.A,
                          gen2.Session.S1: gen2.InventoryFlag.A,
                          gen2.Session.S2: gen2.InventoryFlag.A,
                          gen2.Session.S3: gen2.InventoryFlag.A}
        self._sl = False
        self._m = None
        self._dr = None
        self._blf = None
        self._trext = None
        self._session = None

        self._last_power_up = -np.inf
        self._last_power_down = -np.inf
        self._received_power = None
        self._last_position_update = 0

        self._command_timeout_id = None

    ###################################################################
    # PROPERTIES
    ###################################################################
    @property
    def name(self): return "Tag{}".format(self.identifier)

    @property
    def antenna(self): return self._antenna

    @property
    def transceiver(self): return self._transceiver

    @property
    def speed(self): return self._speed

    @speed.setter
    def speed(self, value): self._speed = value

    @property
    def direction(self): return self._direction

    @direction.setter
    def direction(self, value): self._direction = vectors.normalize(value)

    @property
    def velocity(self): return super().velocity

    @velocity.setter
    def velocity(self, value):
        self._speed = vectors.length(value)
        if self._speed > 1e-9:
            self._direction = vectors.normalize(value)

    @property
    def node_type(self): return phy.NodeType.TAG_NODE
    
    @property
    def last_powered_on(self): return self._last_power_up

    @property
    def state(self): return self._state

    @state.setter
    def state(self, value):
        if value != self._state:
            # pyons.fine("state: {} ==> {}".format(self._state.name,
            #                                      value.name),
            #            sender=self.name)
            self._cancel_command_timeout()
            self._state = value

    @property
    def orientation(self): return self._orientation

    @orientation.setter
    def orientation(self, value): self._orientation = vectors.normalize(value)

    @property
    def up_direction(self): return self._up_direction

    @up_direction.setter
    def up_direction(self, value):
        self._up_direction = vectors.normalize(value)

    @property
    def received_power(self): return self._received_power

    @property
    def energized(self): return self.state != Tag.State.OFF

    @property
    def blf(self): return self._blf

    @property
    def m(self): return self._m

    @property
    def preamble_duration(self):
        if self.m is None or self.trext is None or self.blf is None:
            return None
        return gen2.TagFrame(self.m, self.trext, self.blf).preamble_duration

    @property
    def dr(self): return self._dr

    @property
    def trext(self): return self._trext

    @property
    def session(self): return self._session

    @property
    def sessions(self): return self._sessions

    @property
    def sl(self): return self._sl

    @property
    def slot(self): return self._slot

    @property
    def vehicle_id(self):
        return None if self.vehicle is None else self.vehicle.vehicle_id

    @property
    def lane(self):
        return None if self.vehicle is None else self.vehicle.lane

    ###################################################################
    # Public API
    ###################################################################
    def update_received_power(self):
        self._received_power = self.channel.get_rx_power(
            sender=self.channel.active_transceiver, receiver=self.transceiver,
            tx_power=self.channel.active_transceiver.power)

        # pyons.fine("updated received power: {}".format(
        #     '0W' if self._received_power is None else "{}dBm".format(
        #         self._received_power)), sender=self.name)

        tx_power = (self._received_power
                    if (self._received_power is not None
                        and self._received_power >= self.sensitivity)
                    else phy.THERMAL_NOISE)

        self.transceiver.set_power(tx_power)
        if (self.state is not Tag.State.OFF
            and (self._received_power is None
                 or self._received_power < self.sensitivity)):
            # print("D>> state={} "
            #       "received_power={:.2f}dBm sensitivity={:.2f}dBm"
            #       "".format(self.state.name, self._received_power,
            #                 self.sensitivity))
            self._power_down()
        elif (self.state is Tag.State.OFF
              and (self._received_power is not None
                   and self._received_power >= self.sensitivity)):
            # print("U>> state={} "
            #       "received_power={:.2f}dBm sensitivity={:.2f}dBm"
            #       "".format(self.state.name, self._received_power,
            #                 self.sensitivity))
            self._power_up()

    def update_position(self):
        dt = pyons.time() - self._last_position_update
        self.position += self.velocity * dt
        self._last_position_update = pyons.time()

    ###################################################################
    # API for the Transceiver
    ###################################################################

    @property
    def turned_on(self):
        return self.energized

    @Entity.managed
    def receive_started(self):
        self._cancel_command_timeout()

    def receive_error(self, snr, ber):
        pass

    def receive_collision(self, n_replies):
        pass

    @Entity.managed
    def receive_finished(self, frame, rx_power, snr=None, ber=None):
        #
        # Ignore anything in the OFF state
        #
        if self.state == Tag.State.OFF:
            return

        #
        # If nothing received, treat as command timeout fired
        # FIXME: possible error if waited not too long
        #
        if frame is None:
            # pyons.fine(
            #     "[state={state}] receive failure{snr}{ber}"
            #     "".format(
            #         state=self.state.name,
            #         snr=('' if snr is None else " snr={}dBm".format(snr)),
            #         ber=('' if ber is None else " ber={:.2f}".format(ber))),
            #     sender=self.name)
            self._handle_command_timeout()
            return

        #
        # If frame was received successfully, extract the command
        # and dispatch it.
        #
        assert isinstance(frame, gen2.ReaderFrame)
        cmd = frame.cmd

        # pyons.fine(
        #     "[state={state}] receive: {cmd}, power={power}dBm{snr}{ber}"
        #     "".format(state=self.state.name, cmd=cmd, power=rx_power,
        #               snr=('' if snr is None else " snr={}dBm".format(snr)),
        #              ber=('' if ber is None else " ber={:.2f}".format(ber))),
        #     sender=self.name)

        if isinstance(cmd, gen2.Query):
            self._handle_query(cmd, frame.preamble)
        elif isinstance(cmd, gen2.QueryRep):
            self._handle_query_rep(cmd)
        elif isinstance(cmd, gen2.Ack):
            self._handle_ack(cmd)
        elif isinstance(cmd, gen2.ReqRn):
            self._handle_reqrn(cmd)
        elif isinstance(cmd, gen2.Read):
            self._handle_read(cmd)
        else:
            raise RuntimeError("unsupported command {}".format(type(cmd)))

    def send_finished(self):
        if self.state in [Tag.State.ARBITRATE, Tag.State.REPLY,
                          Tag.State.ACKNOWLEDGED]:
            self._create_command_timeout()

    ###################################################################
    # INITIALIZERS
    ###################################################################
    @Entity.initializer(stage=INIT_TAG_STAGE)
    def _initialize(self):
        self._power_down()
        self.created_at = pyons.time()
        self._last_position_update = pyons.time()
        pyons.add_entity(self.transceiver)

        tag_info = journal.TagInfoRecord()
        tag_info.created_at = pyons.time()
        tag_info.vehicle_id = self.vehicle_id
        tag_info.epc = self.epc
        tag_info.tid = self.tid
        tag_info.location = self.location
        tag_info.antenna_forward_dir = np.array(self.antenna.dir_forward)
        tag_info.antenna_right_dir = np.array(self.antenna.dir_right)
        tag_info.antenna_gain = self.antenna.gain
        tag_info.antenna_polarization = self.antenna.polarization
        tag_info.antenna_rp = self.antenna.rp
        tag_info.sensitivity = self.sensitivity
        tag_info.lane = self.lane
        tag_info.n_epc_read = 0
        tag_info.n_tid_read = 0
        tag_info.modulation_loss = self.transceiver.modulation_loss
        journal.Journal().write_tag_created(tag_info)

    ###################################################################
    # FINALIZERS
    ###################################################################
    @Entity.finalizer(stage=FINISH_TAG_STAGE)
    def _finish(self):
        self._power_down()
        self.transceiver.clear()
        journal.Journal().write_tag_destroyed(self.epc, pyons.time())
        pyons.remove_entity(self.transceiver)

    ###################################################################
    # DEATH CONDITIONS
    ###################################################################

    ###################################################################
    # STOP CONDITIONS
    ###################################################################

    ###################################################################
    # EVENT HANDLERS
    ###################################################################
    @Entity.eventhandler(lambda ev, src: ev == Tag.COMMAND_TIMEOUT_EVENT)
    def _handle_command_timeout(self, event, source):
        assert event == Tag.COMMAND_TIMEOUT_EVENT and source is self
        if self.state in [Tag.State.ARBITRATE, Tag.State.REPLY,
                          Tag.State.ACKNOWLEDGED]:
            self.state = Tag.State.ARBITRATE
        self._command_timeout_id = None

    ###################################################################
    # INTERNAL API
    ###################################################################
    def _send(self, reply):
        frame = gen2.TagFrame(m=self.m, trext=self.trext, blf=self.blf,
                              reply=reply)
        # pyons.fine("[state={state}] send: {reply}, "
        #            "duration={duration:.2f}us"
        #            "".format(state=self.state.name, reply=reply,
        #                      duration=frame.duration * 1e6),
        #                      sender=self.name)
        self.transceiver.send(frame)

    def _power_up(self):
        if self._state != Tag.State.OFF:
            return

        # 1) Updating state
        self._state = Tag.State.READY

        # 2) Updating sessions flags and SL
        dt = pyons.time() - self._last_power_down
        self._last_power_up = pyons.time()
        self._sessions[gen2.Session.S0] = gen2.InventoryFlag.A
        if dt > self.s1_persistence:
            self._sessions[gen2.Session.S1] = gen2.InventoryFlag.A
        if dt > self.s2_persistence:
            self._sessions[gen2.Session.S2] = gen2.InventoryFlag.A
        if dt > self.s3_persistence:
            self._sessions[gen2.Session.S3] = gen2.InventoryFlag.A
        if dt > self.sl_persistence:
            self._sl = False
        # pyons.fine("powered up, power={:.2f}dBm, position={}".format(
        #     self._received_power, self.position), sender=self.name)

    def _power_down(self):
        if self.state == Tag.State.OFF:
            return
        self._cancel_command_timeout()
        self._last_power_down = pyons.time()
        self.state = Tag.State.OFF
        self._slot = None
        self._rn = None
        self._m = None
        self._blf = None
        self._trext = None
        self._session = None
        # pyons.fine("powered down, power={:.2f}dBm, position={}".format(
        #     self._received_power if self._received_power is not None
        #     else phy.THERMAL_NOISE, self.position), sender=self.name)

    def _handle_query(self, query, preamble):
        self._cancel_command_timeout()  # QUERY is processed, anyway
        if self.state == Tag.State.OFF or self.state == Tag.State.KILLED:
            return

        # If received after already inventoried, switch session flag
        if self.state in [Tag.State.ACKNOWLEDGED, Tag.State.OPEN,
                          Tag.State.SECURED]:
            if self._session == query.session:
                self._sessions[self._session] = \
                    self._sessions[self._session].invert()

        # pyons.fine(("received query: "
        #             "\n\t* state    : {state}"
        #             "\n\t* position : {position}"
        #             "\n\t* sessions : S0={s0}, S1={s1}, S2={s2}, S3={s3}"
        #             "\n\t* SL       : {sl}"
        #             "\n\t* query    : {cmd}"
        #             "\n\t* preamble : {preamble}").format(
        #     state=self.state.name, position=self.position,
        #     s0=self.sessions[gen2.Session.S0].name,
        #     s1=self.sessions[gen2.Session.S1].name,
        #     s2=self.sessions[gen2.Session.S3].name,
        #     s3=self.sessions[gen2.Session.S2].name,
        #     sl=self.sl, cmd=query, preamble=preamble), sender=self.name)

        # Check the requested session and SL values
        if ((self._sessions[query.session] != query.target) or
                (query.sel == gen2.Sel.SL_YES and not self._sl) or
                (query.sel == gen2.Sel.SL_NO and self._sl)):
            # pyons.fine("ignoring query due to session/SL mismatch",
            #            sender=self.name)
            self.state = Tag.State.READY
            return

        journal.Journal().tag_info[self.epc].n_rounds += 1
        # If everything OK: compute slot, RN16, fill session, etc.
        self._slot = np.random.randint(0, 2 ** query.q)
        self._trext = query.trext
        self._session = query.session
        self._m = query.m
        self._dr = query.dr
        self._blf = self._dr.ratio / preamble.trcal

        # pyons.fine(("processed query:"
        #             "\n\t* slot   : {slot}"
        #             "\n\t* trext  : {trext}"
        #             "\n\t* session: {session}"
        #             "\n\t* m      : {m}"
        #             "\n\t* dr     : {dr}"
        #             "\n\t* blf    : {blf:.2f}").format(
        #    slot=self._slot, trext=self.trext, session=self.session, m=self.m,
        #     dr=self.dr, blf=self.blf), sender=self.name)

        if self._slot > 0:
            self.state = Tag.State.ARBITRATE
        else:
            self.state = Tag.State.REPLY
            self._rn = np.random.randint(0x0000, 0x10000)
            self._send(gen2.Rn16Reply(self._rn))

    def _handle_query_rep(self, command):
        assert isinstance(command, gen2.QueryRep)
        if self.state in [Tag.State.OFF, Tag.State.KILLED, Tag.State.READY]:
            return

        # Check session. If doesn't match, do nothing
        if command.session != self._session:
            return

        # if session matches, do not ignore QueryRep
        self._cancel_command_timeout()

        # If received after already inventoried, switch session flag
        if self.state in [Tag.State.ACKNOWLEDGED, Tag.State.OPEN,
                          Tag.State.SECURED]:
            self._sessions[self._session] = \
                self._sessions[self._session].invert()
            self.state = Tag.State.READY
            return

        self._slot = self._slot - 1 if self._slot > 0 else 0xFFFF
        if self._slot > 0:
            self.state = Tag.State.ARBITRATE
        else:
            self.state = Tag.State.REPLY
            self._rn = np.random.randint(0x0000, 0x10000)
            self._send(gen2.Rn16Reply(self._rn))

    def _handle_ack(self, command):
        assert isinstance(command, gen2.Ack)
        if self.state not in [Tag.State.OFF, Tag.State.KILLED, Tag.State.READY,
                              Tag.State.ARBITRATE]:
            self._cancel_command_timeout()
            if command.rn == self._rn:
                if self.state == Tag.State.REPLY:
                    self.state = Tag.State.ACKNOWLEDGED
                self._send(gen2.AckReply(self.epc))
            else:
                self.state = Tag.State.ARBITRATE

    def _handle_reqrn(self, command):
        assert isinstance(command, gen2.ReqRn)
        backscatter = False
        if self.state in [Tag.State.ARBITRATE, Tag.State.REPLY]:
            self._cancel_command_timeout()
            self.state = Tag.State.ARBITRATE
        elif self.state == Tag.State.ACKNOWLEDGED:
            if self._rn == command.rn:
                self._cancel_command_timeout()
                self.state = Tag.State.SECURED
                backscatter = True
        elif self.state in [Tag.State.OPEN, Tag.State.SECURED]:
            backscatter = (command.rn == self._rn)

        if backscatter:
            self._rn = np.random.randint(0x0000, 0x10000)
            self._send(gen2.ReqRnReply(rn=self._rn))

    def _handle_read(self, command):
        assert isinstance(command, gen2.Read)
        if self.state in [Tag.State.ARBITRATE, Tag.State.REPLY,
                          Tag.State.ACKNOWLEDGED]:
            self.state = Tag.State.ARBITRATE
        elif self.state in [Tag.State.OPEN, Tag.State.SECURED]:
            if command.bank == gen2.Bank.TID:
                bank = self.tid
            elif command.bank == gen2.Bank.EPC:
                bank = self.epc
            else:
                raise RuntimeError("only TID and EPC banks supported")

            assert isinstance(bank, Iterable)

            if isinstance(bank, str):
                bank = [b for b in binascii.unhexlify(bank)]
            else:
                bank = list(bank)
            n_bytes = len(bank)
            n_words = int(np.ceil(n_bytes / 2))

            # This may be an over-simplification, but if requested more
            # bytes then there are in the bank, extend the bank with
            # zeros
            if n_words < command.wordptr + command.wordcnt:
                bank += [0x00] * (2 * (command.wordptr + command.wordcnt)
                                  - n_bytes)

            first_address = 2 * command.wordptr
            last_address = 2 * (command.wordptr + command.wordcnt)
            words = bank[first_address:last_address]

            self._send(gen2.ReadReply(words=words, rn=self._rn, crc16=0x0000))

    @Entity.managed
    def _create_command_timeout(self):
        self._cancel_command_timeout()
        self._command_timeout_id = pyons.create_timeout(
            gen2.max_t2(self.blf), Tag.COMMAND_TIMEOUT_EVENT)

    def _cancel_command_timeout(self):
        if self._command_timeout_id is not None:
            pyons.cancel(self._command_timeout_id)
            self._command_timeout_id = None


@pyons.initializer(stage=INIT_MOBILITY_MANAGER)
def init_mobility_manager():
    model = pyons.get_model()
    pyons.create_timeout(model.position_update_interval, UPDATE_POSITION_EVENT)


@pyons.eventhandler(lambda ev, src: ev == UPDATE_POSITION_EVENT)
def handle_update_position(event, source):
    assert event == UPDATE_POSITION_EVENT and source is None
    model = pyons.get_model()
    for tag in model.tags:
        assert isinstance(tag, Tag)
        tag.update_position()
        tag.update_received_power()
        j = journal.Journal()
        if j.channel_state_logging_enabled:
            rec = build_channel_state_record(model.reader, tag, model.channel)
            j.write_channel_state(rec)
    pyons.create_timeout(model.position_update_interval, UPDATE_POSITION_EVENT)


def build_channel_state_record(reader, tag, channel, tag_rx_power=None):
    r_trans = reader.transceiver
    t_trans = tag.transceiver
    rt_path_loss = channel.get_path_loss(r_trans, t_trans)
    tr_path_loss = channel.get_path_loss(t_trans, r_trans)

    if tag_rx_power is None:
        tag_rx_power = tag.received_power

    reader_rx_power = channel.get_rx_power(t_trans, r_trans, tr_path_loss)

    r_decider = r_trans.decider
    assert isinstance(r_decider, phy.ReaderDecider)
    reader_snr = r_decider.get_snr(reader_rx_power, tag.m,
                                   tag.preamble_duration, tag.blf)
    reader_ber = r_decider.get_ber(reader_snr)

    rec = journal.ChannelStateRecord()
    rec.timestamp = pyons.time()
    rec.reader_position = np.array(reader.antenna.position)
    rec.reader_side = reader.antenna.side
    rec.reader_lane = reader.antenna.lane
    rec.tag_position = np.array(tag.antenna.position)
    rec.tag_location = tag.location
    rec.tag_lane = tag.vehicle.lane if tag.vehicle is not None else None
    rec.tag_speed = tag.speed
    rec.channel_lifetime = channel.get_channel_lifetime()
    rec.tag_rx_power = tag_rx_power
    rec.reader_rx_power = reader_rx_power
    rec.reader_snr = reader_snr
    rec.reader_ber = reader_ber
    rec.rt_path_loss = rt_path_loss
    rec.tr_path_loss = tr_path_loss
    rec.vehicle_id = tag.vehicle_id
    return rec
