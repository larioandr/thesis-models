from enum import Enum
import numpy as np
from collections import Iterable

import pyons
from pyons import Entity
import phy
import protocol as gen2
import journal
import pyradise


class ReaderDescriptor(object):
    def __init__(self):
        super().__init__()
        self.antennas = []
        self.rounds_per_antenna = 1
        self.tari = 6.25e-6
        self.rtcal = 18.75e-6
        self.data0_multiplier = None
        self.trcal = 33.5e-6
        self.rtcal_multiplier = None
        self.tx_power = 31.5
        self.circulator_noise = -80.0
        self.switch_power = False
        self.power_on_interval = 2.0
        self.power_off_interval = 0.1
        self.sl = gen2.Sel.SL_ALL
        self.session = gen2.Session.S0
        self.session_strategy = Reader.SessionStrategy.ONLY_A
        self.rounds_per_inventory_flag = 1
        self.frequency = 860e6
        self.modulation_loss = 0.0
        self.tag_encoding = gen2.TagEncoding.FM0
        self.dr = gen2.DR.DR_8
        self.trext = False
        self.q = 2
        self.channel = None
        self.ber_model = pyradise.ber_over_rayleigh
        self.read_tid = True


INIT_READER_STAGE = (0, "Init reader")
FINISH_READER_STAGE = (9, "Finish reader")


class Reader(phy.Node):
    DELAYED_SEND_EVENT = "delayed send"
    REPLY_TIMEOUT_EVENT = "reply timeout"
    POWER_ON_TIMEOUT_EVENT = "power on timeout"
    POWER_OFF_TIMEOUT_EVENT = "power off timeout"

    class SessionStrategy(Enum):
        ONLY_A = 0
        ONLY_B = 1
        ALTER = 2

    def __init__(self, descriptor):
        super().__init__()
        assert isinstance(descriptor, ReaderDescriptor)
        self.antennas = descriptor.antennas
        for antenna in self.antennas:
            antenna.node = self
        self.rounds_per_antenna = descriptor.rounds_per_antenna
        self.tari = descriptor.tari
        self.rtcal = (descriptor.rtcal if descriptor.data0_multiplier is None
                      else (descriptor.data0_multiplier + 1) * descriptor.tari)
        self.trcal = (descriptor.trcal if descriptor.rtcal_multiplier is None
                      else self.rtcal * descriptor.rtcal_multiplier)
        self.max_tx_power = descriptor.tx_power
        self.frequency = descriptor.frequency
        self.circulator_noise = descriptor.circulator_noise
        self.switch_power = descriptor.switch_power
        self.power_on_interval = descriptor.power_on_interval
        self.power_off_interval = descriptor.power_off_interval
        self.channel = descriptor.channel
        self.read_tid = descriptor.read_tid

        self.sl = descriptor.sl
        self.session = descriptor.session
        self.session_strategy = descriptor.session_strategy
        self.rounds_per_inventory_flag = descriptor.rounds_per_inventory_flag
        self.m = descriptor.tag_encoding
        self.dr = descriptor.dr
        self.trext = descriptor.trext
        self.q = descriptor.q

        decider = phy.ReaderDecider(None, descriptor.channel,
                                    descriptor.ber_model)
        self._transceiver = phy.Transceiver(
            node=self, decider=decider, channel=descriptor.channel,
            modulation_loss=descriptor.modulation_loss)
        decider.transceiver = self._transceiver

        self._round_index = 0
        self._power = None
        self._antenna_index = 0
        self._last_powered_on = -np.inf
        self._last_powered_off = -np.inf
        self._last_tx_end = -np.inf
        self._last_rx_end = -np.inf
        self._inventory_flag = None

        self._reply_timeout_id = None
        self._delayed_send_timeout_id = None
        self._power_on_timeout_id = None
        self._power_off_timeout_id = None

        self._slot = None

        # Round and slot info caches
        self.round_info = None
        self.tag_read_data = None
        self._round_started_at = None
        self._slot_started_at = None
        self._slot_durations = []

    ###################################################################
    # PROPERTIES
    ###################################################################
    @property
    def transceiver(self): return self._transceiver

    @property
    def speed(self): return 0.0

    @property
    def direction(self): return np.array([1, 0, 0])

    @property
    def velocity(self): return np.array([0, 0, 0])

    @property
    def node_type(self): return phy.NodeType.READER_NODE

    @property
    def last_powered_on(self): return self._last_powered_on

    @property
    def name(self): return "Reader"

    @property
    def radiated_power(self):
        return (None if self._power is None
                else self._power + self.antenna.cable_loss)

    @property
    def is_powered_on(self): return self._power is not None

    @property
    def antenna(self): return self.antennas[self._antenna_index]

    @property
    def last_powered_off(self): return self._last_powered_off

    @property
    def wavelen(self): return phy.SPEED_OF_LIGHT / self.frequency

    @property
    def blf(self): return self.dr.ratio / self.trcal

    @property
    def data0(self): return self.tari

    @property
    def data1(self): return self.rtcal - self.data0

    ###################################################################
    # Public API
    ###################################################################
    @Entity.managed
    def power_on(self):
        if self.is_powered_on:
            return
        # pyons.fine("powered on", sender=self.name)
        self._cancel_power_timeout()
        self._power = self.max_tx_power
        self._last_powered_on = pyons.time()
        self._inventory_flag = gen2.InventoryFlag.A
        self.transceiver.set_power(self._power)
        for transceiver in self.channel.passive_transceivers:
            transceiver.node.update_received_power()
        if self.power_on_interval is not None:
            self._power_off_timeout_id = pyons.create_timeout(
                self.power_on_interval, Reader.POWER_OFF_TIMEOUT_EVENT)
        self._handle_round_start()

    @Entity.managed
    def power_off(self):
        if not self.is_powered_on:
            return
        # pyons.fine("powered off", sender=self.name)
        self._power = None
        self._last_powered_off = pyons.time()
        self._inventory_flag = None
        self._cancel_power_timeout()
        self._cancel_delayed_send()
        self._cancel_reply_timeout()
        self.transceiver.set_power(None)
        self.transceiver.clear()
        for transceiver in self.channel.passive_transceivers:
            transceiver.node.update_received_power()
        if self.power_off_interval is not None:
            self._power_on_timeout_id = pyons.create_timeout(
                self.power_off_interval, Reader.POWER_ON_TIMEOUT_EVENT)

    ###################################################################
    # API for the Transceiver
    ###################################################################
    @property
    def turned_on(self):
        return self.is_powered_on

    @Entity.managed
    def send_finished(self):
        self._last_tx_end = pyons.time()
        dt = gen2.max_t1(rtcal=self.rtcal, blf=self.blf) + gen2.t3()
        self._reply_timeout_id = pyons.create_timeout(
            dt, Reader.REPLY_TIMEOUT_EVENT)

    @Entity.managed
    def receive_started(self):
        self._cancel_reply_timeout()

    def receive_error(self, snr, ber):
        assert isinstance(self.round_info, journal.InventoryRoundRecord)
        self.round_info.n_errors += 1

    def receive_collision(self, n_replies):
        assert isinstance(self.round_info, journal.InventoryRoundRecord)
        self.round_info.n_collisions += 1

    @Entity.managed
    def receive_finished(self, frame, rx_power, snr=None, ber=None):
        if frame is None:
            # pyons.fine("receive error {snr}{ber}".format(
            #     snr=('' if snr is None else ' snr={:.2f}dB'.format(snr)),
            #     ber=('' if ber is None else ' ber={:.2f}'.format(ber))),
            #     sender=self.name)
            self._reply_timeout_id = pyons.create_timeout(
                gen2.max_t2(self.blf), Reader.REPLY_TIMEOUT_EVENT)
            return

        assert isinstance(frame, gen2.TagFrame)

        # pyons.fine("received: {reply}, power={power}dBm{snr}{ber}".format(
        #     reply=frame.reply, power=rx_power,
        #     snr=('' if snr is None else ' snr={:.2f}dB'.format(snr)),
        #     ber=('' if ber is None else ' ber={:.2f}'.format(ber))),
        #     sender=self.name)

        self._cancel_reply_timeout()
        self._cancel_delayed_send()

        self._last_rx_end = pyons.time()
        reply = frame.reply
        if isinstance(reply, gen2.Rn16Reply):
            self._handle_rn16_reply(reply)
        elif isinstance(reply, gen2.AckReply):
            self._handle_ack_reply(reply)
        elif isinstance(reply, gen2.ReqRnReply):
            self._handle_reqrn_reply(reply)
        elif isinstance(reply, gen2.ReadReply):
            self._handle_read_reply(reply)
        else:
            raise RuntimeError(
                "reader doesn't support reply {}".format(type(reply)))

    ###################################################################
    # INITIALIZERS
    ###################################################################
    @Entity.initializer(stage=INIT_READER_STAGE)
    def _initialize(self):
        if len(self.antennas) == 0:
            raise RuntimeError("no antennas attached to the reader")
        self._antenna_index = 0
        self._round_index = 0
        self.power_on()
        pyons.add_entity(self.transceiver)

        reader_info = journal.ReaderInfoRecord()
        reader_info.n_antennas = len(self.antennas)
        reader_info.tx_power = self.max_tx_power
        reader_info.circulator_noise = self.circulator_noise
        reader_info.power_switch_enabled = self.switch_power
        reader_info.session_switch_enabled = (self.session_strategy ==
                                              Reader.SessionStrategy.ALTER)
        reader_info.power_on_duration = self.power_on_interval
        reader_info.power_off_duration = self.power_off_interval
        reader_info.n_rounds_per_antenna = self.rounds_per_antenna
        reader_info.n_rounds_per_session = self.rounds_per_inventory_flag
        reader_info.n_slots = 2 ** self.q
        reader_info.q = self.q
        reader_info.m = self.m
        reader_info.trext = self.trext
        reader_info.dr = self.dr
        reader_info.blf = self.blf
        reader_info.data0 = self.data0
        reader_info.data1 = self.data1
        reader_info.rtcal = self.rtcal
        reader_info.trcal = self.trcal
        journal.Journal().write_reader_info(reader_info)

        for antenna_index in range(0, len(self.antennas)):
            antenna = self.antennas[antenna_index]
            assert isinstance(antenna, phy.ReaderAntenna)
            antenna_info = journal.ReaderAntennaInfoRecord()
            antenna_info.index = antenna_index
            antenna_info.side = antenna.side
            antenna_info.lane = antenna.lane
            antenna_info.position = antenna.position
            antenna_info.forward_dir = np.array(antenna.dir_forward)
            antenna_info.right_dir = np.array(antenna.dir_right)
            antenna_info.gain = antenna.gain
            antenna_info.rp = antenna.rp
            antenna_info.polarization = antenna.polarization
            antenna_info.cable_loss = antenna.cable_loss
            journal.Journal().write_reader_antenna_info(antenna_info)

    ###################################################################
    # FINALIZERS
    ###################################################################
    @Entity.finalizer(stage=FINISH_READER_STAGE)
    def _finish(self):
        self.transceiver.clear()
        self.power_off()
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
    @Entity.eventhandler(lambda ev, src: (isinstance(ev, Iterable) and
                                          ev[0] == Reader.DELAYED_SEND_EVENT))
    def _send_delayed(self, event, source):
        assert source is self
        msg, frame = event
        self.transceiver.send(frame)
        self._delayed_send_timeout_id = None

    @Entity.eventhandler(lambda ev, src: ev == Reader.REPLY_TIMEOUT_EVENT)
    def _handle_reply_timeout(self, event, source):
        assert event == Reader.REPLY_TIMEOUT_EVENT and source is self
        # pyons.fine("no reply received", sender=self.name)
        self._reply_timeout_id = None
        # STATISTICS ====>
        if self.tag_read_data.epc is None:
            self.round_info.n_empty_slots += 1
        # <===== END OF STATISTICS
        self._handle_slot_end()

    @Entity.eventhandler(lambda ev, src: ev == Reader.POWER_ON_TIMEOUT_EVENT)
    def _handle_power_on_timeout(self, event, source):
        assert event == Reader.POWER_ON_TIMEOUT_EVENT and source is self
        self._power_on_timeout_id = None
        self.power_on()

    @Entity.eventhandler(lambda ev, src: ev == Reader.POWER_OFF_TIMEOUT_EVENT)
    def _handle_power_off_timeout(self, event, source):
        assert event == Reader.POWER_OFF_TIMEOUT_EVENT and source is self
        self._power_off_timeout_id = None
        self.power_off()

    ###################################################################
    # INTERNAL API
    ###################################################################
    def _send(self, cmd):
        if isinstance(cmd, gen2.Query):
            preamble = gen2.ReaderFrame.Preamble(
                tari=self.tari, rtcal=self.rtcal, trcal=self.trcal)
        else:
            preamble = gen2.ReaderFrame.Sync(tari=self.tari, rtcal=self.rtcal)
        frame = gen2.ReaderFrame(preamble=preamble, cmd=cmd)

        # pyons.fine("send: {cmd}, duration={duration:.2f}us".format(
        #     cmd=cmd, duration=frame.duration*1e6), sender=self.name)

        self._cancel_delayed_send()

        t2 = gen2.min_t2(self.blf)
        time_from_rx_end = pyons.time() - self._last_rx_end
        if time_from_rx_end >= t2:
            self.transceiver.send(frame)
        else:
            dt = t2 - time_from_rx_end
            self._delayed_send_timeout_id = pyons.create_timeout(
                dt, (Reader.DELAYED_SEND_EVENT, frame))

    #
    # Inventory round pseudo-events handlers
    #
    def _handle_round_start(self):
        if not self.is_powered_on:
            raise RuntimeError("round start during power-off")

        self._round_index += 1
        self._slot = 0

        #
        # Switching antenna if needed
        #
        if self.rounds_per_antenna is not None and (
                        self._round_index % self.rounds_per_antenna == 0):
            n_antennas = len(self.antennas)
            self._antenna_index = (self._antenna_index + 1) % n_antennas

        #
        # Switching inventory flag if needed
        #
        if self.session_strategy == Reader.SessionStrategy.ALTER:
            if (self.rounds_per_inventory_flag is not None) and \
                    (self._round_index % self.rounds_per_inventory_flag == 0):
                self._inventory_flag = self._inventory_flag.invert()

        # pyons.fine(("\n\t*---------- NEW ROUND ------------*"
        #             "\n\t* round index  : {round_index}"
        #             "\n\t* antenna index: {antenna_index}"
        #             "\n\t* Q            : {q}"
        #             "\n\t* session      : {session}"
        #             "\n\t* target       : {target}").format(
        #     round_index=self._round_index, antenna_index=self._antenna_index,
        #     q=self.q, session=self.session, target=self._inventory_flag),
        #     sender=self.name)

        for transceiver in self.channel.passive_transceivers:
            transceiver.node.update_position()
            transceiver.node.update_received_power()

        #
        # Refresh statistics
        #
        # STATISTICS ====>
        self.tag_read_data = self._create_tag_read_data()
        self.round_info = self._create_round_info()
        self._slot_durations = []
        self._slot_started_at = pyons.time()
        self._round_started_at = pyons.time()
        model = pyons.get_model()
        self.round_info.n_tags = len([tag for tag in model.tags
                                      if tag.energized])
        self.round_info.n_vehicles_registered = len(model.vehicles)
        self.round_info.antenna_index = self._antenna_index
        # <===== END OF STATISTICS

        #
        # Generating and sending Query
        #
        query = gen2.Query(dr=self.dr, m=self.m, trext=self.trext, sel=self.sl,
                           session=self.session, target=self._inventory_flag,
                           q=self.q)
        self._send(query)

    def _handle_round_end(self):
        # STATISTICS ====>
        assert isinstance(self.round_info, journal.InventoryRoundRecord)
        self.round_info.duration = pyons.time() - self._round_started_at
        self.round_info.min_slot_duration = np.min(self._slot_durations)
        self.round_info.max_slot_duration = np.max(self._slot_durations)
        self.round_info.avg_slot_duration = np.average(self._slot_durations)
        journal.Journal().write_inventory_round(self.round_info)
        self.round_info = None
        self._slot_durations = None
        # <===== END OF STATISTICS
        self._handle_round_start()

    def _handle_slot_start(self):
        # pyons.fine("start slot #{}".format(self._slot), sender=self.name)

        # STATISTICS ====>
        self.tag_read_data = self._create_tag_read_data()
        self._slot_started_at = pyons.time()
        # <===== END OF STATISTICS

        if not self.is_powered_on:
            raise RuntimeError("slot begin during power-off")
        qrep = gen2.QueryRep(session=self.session)
        self._send(qrep)

    def _handle_slot_end(self):
        # STATISTICS ====>
        assert isinstance(self.tag_read_data, journal.TagReadRecord)
        assert isinstance(self.round_info, journal.InventoryRoundRecord)
        if self.tag_read_data.epc is not None:
            if self.tag_read_data.tid is not None:
                self.round_info.n_tid_reads += 1
            else:
                self.round_info.n_epc_only_reads += 1
            journal.Journal().write_tag_read(self.tag_read_data)
            pyons.info("** received tag: EPC={}, TID={}".format(
                self.tag_read_data.epc, self.tag_read_data.tid))
        self._slot_durations.append(pyons.time() - self._slot_started_at)
        self.tag_read_data = None
        # <===== END OF STATISTICS
        self._slot += 1
        if self._slot < 2 ** self.q:
            self._handle_slot_start()
        else:
            self._handle_round_end()

    #
    # Received replies handlers
    #
    def _handle_rn16_reply(self, reply):
        self._rn = reply.rn
        self._send(gen2.Ack(rn=self._rn))

    def _handle_ack_reply(self, reply):
        epc = (reply.epc if isinstance(reply.epc, str)
               else "".join([format(x, '02X') for x in reply.epc]))

        assert isinstance(self.tag_read_data, journal.TagReadRecord)
        tag = pyons.get_model().get_tag(epc)
        self.tag_read_data.epc = epc
        self.tag_read_data.vehicle_lane = tag.lane
        self.tag_read_data.location = tag.location

        if self.read_tid:
            self._send(gen2.ReqRn(rn=self._rn))
        else:
            self._handle_slot_end()

    def _handle_reqrn_reply(self, reply):
        self._rn = reply.rn
        self._send(gen2.Read(bank=gen2.Bank.TID, wordptr=0, wordcnt=4,
                             rn=self._rn))

    def _handle_read_reply(self, reply):
        tid = (reply.words if isinstance(reply.words, str)
               else "".join([format(x, '02X') for x in reply.words]))
        self.tag_read_data.tid = tid
        self._handle_slot_end()

    #
    # Timeouts cancellers
    #
    def _cancel_reply_timeout(self):
        if self._reply_timeout_id is not None:
            pyons.cancel(self._reply_timeout_id)
            self._reply_timeout_id = None

    def _cancel_delayed_send(self):
        if self._delayed_send_timeout_id is not None:
            pyons.cancel(self._delayed_send_timeout_id)
            self._delayed_send_timeout_id = None

    def _cancel_power_timeout(self):
        if self._power_on_timeout_id is not None:
            pyons.cancel(self._power_on_timeout_id)
            self._power_on_timeout_id = None
        if __name__ == '__main__':
            if self._power_off_timeout_id is not None:
                pyons.cancel(self._power_off_timeout_id)
                self._power_off_timeout_id = None

    #
    # Journal records, cache, etc
    #
    def _create_tag_read_data(self):
        trd = journal.TagReadRecord()
        trd.slot_index = self._slot
        trd.slot_timestamp = pyons.time()
        trd.reader_antenna_lane = self.antenna.lane
        trd.reader_antenna_side = self.antenna.side
        trd.round_index = self._round_index
        trd.epc = None
        trd.tid = None
        trd.vehicle_lane = None
        trd.location = None
        return trd

    def _create_round_info(self):
        ri = journal.InventoryRoundRecord()
        ri.index = self._round_index
        ri.n_collisions = 0
        ri.n_errors = 0
        ri.n_epc_only_reads = 0
        ri.n_tid_reads = 0
        ri.n_empty_slots = 0
        ri.n_tags = 0
        ri.antenna_side = self.antenna.side
        ri.antenna_lane = self.antenna.lane
        ri.session = self.session
        ri.duration = None
        ri.min_slot_duration = None
        ri.max_slot_duration = None
        ri.avg_slot_duration = None
        return ri
