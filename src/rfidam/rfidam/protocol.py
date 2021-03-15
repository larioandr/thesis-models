from dataclasses import dataclass
from functools import cached_property

from rfidlib.protocol.symbols import TagEncoding, DR, Session, InventoryFlag, \
    Sel, Bank, get_blf, nominal_t1, max_t2, min_t2, t3, t4
from rfidlib.protocol.commands import ReaderPreamble, ReaderSync, \
    ReaderFrame, Query, QueryRep, Ack, ReqRn, Read
from rfidlib.protocol.responses import TagPreamble, TagFrame, Rn16, Epc, \
    Handle, Data


defaults = {
    'session': Session.S1,
    'sel': Sel.ALL,
    'target': InventoryFlag.A,
    'crc5': 127,
    'crc16': 0xAAAA,
    'rn': 0x5555,
    'wordptr': 0,
    'bank': Bank.TID,
    'pc': 0,
}


@dataclass
class LinkProps:
    tari: float
    rtcal: float
    trcal: float
    m: TagEncoding
    dr: DR
    trext: bool
    q: int
    t_off: float
    use_tid: bool
    n_data_words: int
    n_epcid_bytes: int = 12

    @cached_property
    def n_slots(self) -> int:
        return 2**self.q


class RTLink:
    def __init__(self, props: LinkProps):
        self.props = props

    @cached_property
    def preamble(self):
        return ReaderPreamble(
            tari=self.props.tari,
            rtcal=self.props.rtcal,
            trcal=self.props.trcal)

    @cached_property
    def sync(self):
        return ReaderSync(tari=self.props.tari, rtcal=self.props.rtcal)

    @cached_property
    def query(self):
        return ReaderFrame(self.preamble, Query(
            q=self.props.q,
            m=self.props.m,
            dr=self.props.dr,
            trext=self.props.trext,
            sel=defaults['sel'],
            session=defaults['session'],
            target=defaults['target'],
            crc5=defaults['crc5']))

    @cached_property
    def query_rep(self):
        return ReaderFrame(self.sync, QueryRep(session=defaults['session']))

    @cached_property
    def ack(self):
        return ReaderFrame(self.sync, Ack(rn=defaults['rn']))

    @cached_property
    def req_rn(self):
        return ReaderFrame(self.sync, ReqRn(
            rn=defaults['rn'],
            crc16=defaults['crc16']))

    @cached_property
    def read(self):
        return ReaderFrame(self.sync, Read(
            bank=defaults['bank'],
            wordptr=defaults['wordptr'],
            wordcnt=self.props.n_data_words,
            rn=defaults['rn'],
            crc16=defaults['crc16']
        ))


class TRLink:
    def __init__(self, props: LinkProps):
        self._props = props

    @cached_property
    def blf(self):
        return get_blf(self._props.dr, self._props.trcal)

    @property
    def props(self):
        return self._props

    @cached_property
    def preamble(self):
        return TagPreamble(
            m=self.props.m,
            trext=self.props.trext,
            blf=self.blf)

    @cached_property
    def rn16(self):
        return TagFrame(self.preamble, Rn16(rn=defaults['rn']))

    @cached_property
    def epc(self):
        epcid = 'A0' * self._props.n_epcid_bytes
        return TagFrame(self.preamble, Epc(
            epcid=epcid,
            pc=defaults['pc'],
            crc16=defaults['crc16']))

    @cached_property
    def handle(self):
        return TagFrame(self.preamble, Handle(
            rn=defaults['rn'],
            crc16=defaults['crc16']))

    @cached_property
    def data(self):
        words = 'CDEF' * self._props.n_data_words
        return TagFrame(self.preamble, Data(
            words=words,
            rn=defaults['rn'],
            crc16=defaults['crc16'],
            header=0))


class Timings:
    def __init__(self, props: LinkProps):
        self._props = props

    @property
    def props(self):
        return self._props

    @cached_property
    def blf(self):
        return get_blf(self._props.dr, self._props.trcal)

    @cached_property
    def t1(self):
        return nominal_t1(self.props.rtcal, self.blf)

    @cached_property
    def t2(self):
        return 0.5 * (min_t2(self.blf) + max_t2(self.blf))

    @cached_property
    def t3(self):
        return t3()

    @cached_property
    def t4(self):
        return max(t4(self.props.rtcal), self.t1 + self.t3)


class Protocol:
    def __init__(self, props: LinkProps):
        self._props = props

    @cached_property
    def props(self):
        return self._props

    @cached_property
    def rt_link(self):
        return RTLink(self._props)

    @cached_property
    def tr_link(self):
        return TRLink(self._props)

    @cached_property
    def timings(self):
        return Timings(self._props)
