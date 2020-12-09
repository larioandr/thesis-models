from enum import Enum
import numpy as np
import binascii
import collections
import pyons


class DR(Enum):
    DR_8 = 0
    DR_643 = 1

    @staticmethod
    def encode(dr):
        return str(dr.value)

    @property
    def ratio(self):
        return 8.0 if self == DR.DR_8 else 64.0/3


class TagEncoding(Enum):
    FM0 = 1
    M2 = 2
    M4 = 4
    M8 = 8

    @staticmethod
    def encode(m):
        return format(int(np.log2(m.value)), '02b')


class Bank(Enum):
    RESERVED = 0
    EPC = 1
    TID = 2
    USER = 3

    @staticmethod
    def encode(bank):
        return format(bank.value, '02b')


class InventoryFlag(Enum):
    A = 0
    B = 1

    @staticmethod
    def encode(flag):
        return str(flag.value)

    def invert(self):
        return InventoryFlag.A if self == InventoryFlag.B else InventoryFlag.B


class Sel(Enum):
    SL_ALL = 0
    SL_NO = 2
    SL_YES = 3

    @staticmethod
    def encode(field):
        return format(field.value, '02b')


class Session(Enum):
    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3

    @staticmethod
    def encode(session):
        return format(session.value, '02b')


def encode_ebv(value, first_block=True):
    prefix = '0' if first_block else '1'
    if value < 128:
        return prefix + format(value, '07b')
    else:
        return encode_ebv(value >> 7, first_block=False) + \
               encode_ebv(value % 128, first_block=first_block)


def encode(value, width=0, use_ebv=False):
    if isinstance(value, DR):
        return DR.encode(value)
    elif isinstance(value, TagEncoding):
        return TagEncoding.encode(value)
    elif isinstance(value, Bank):
        return Bank.encode(value)
    elif isinstance(value, InventoryFlag):
        return InventoryFlag.encode(value)
    elif isinstance(value, Sel):
        return Sel.encode(value)
    elif isinstance(value, Session):
        return Session.encode(value)
    elif isinstance(value, bool):
        return '1' if value else '0'
    elif isinstance(value, int):
        if use_ebv:
            return encode_ebv(value)
        else:
            if width > 0:
                return format(value, "0{width}b".format(width=width))
            else:
                return format(value, "b")
    elif isinstance(value, str):
        return encode(list(binascii.unhexlify(value.strip())))
    elif isinstance(value, bytes):
        return encode(list(bytes))
    elif isinstance(value, collections.Iterable):
        return "".join(format(x, "08b") for x in value)
    else:
        raise ValueError("unsupported field type={}".format(type(value)))


class ReaderFrame(object):

    class Sync(object):
        def __init__(self, tari, rtcal=None,
                     data1_multiplier=2.0, delim=12.5e-6):
            self.delim = delim
            self.tari = tari
            self.rtcal = (rtcal if rtcal is not None
                          else tari * (data1_multiplier + 1))

        @property
        def duration(self):
            return self.delim + self.tari + self.rtcal

        @property
        def data0(self):
            return self.tari

        @property
        def data1(self):
            return self.rtcal - self.tari

        def __str__(self):
            return "SYNC{{delim={delim:.2f}us tari" \
                   "={tari:.2f}us rtcal={rtcal:.2f}us}}" \
                   "".format(delim=self.delim * 1e6, tari=self.tari * 1e6,
                             rtcal=self.rtcal * 1e6)

    class Preamble(Sync):
        def __init__(self, tari, rtcal=None, trcal=None, data1_multiplier=2.0,
                     trcal_multiplier=2.0, delim=12.5e-6):
            super().__init__(tari=tari, rtcal=rtcal,
                             data1_multiplier=data1_multiplier, delim=delim)
            self.trcal = (trcal if trcal is not None
                          else self.rtcal * trcal_multiplier)

        @property
        def duration(self):
            return super().duration + self.trcal

        def __str__(self):
            return "PREAMBLE{{delim={delim:.2f}us tari={tari:.2f}us " \
                   "rtcal={rtcal:.2f}us trcal={trcal:.2f}us}}" \
                   "".format(delim=self.delim * 1e6, tari=self.tari * 1e6,
                             rtcal=self.rtcal * 1e6, trcal=self.trcal * 1e6)

    def __init__(self, preamble, cmd):
        self.preamble = preamble
        self.cmd = cmd

    @property
    def duration(self):
        bits_cnt = self.cmd.count_bits()
        if self.preamble is None:
            raise pyons.errors.MissingFieldError(
                self.__class__.__name__, 'preamble')
        return (self.preamble.duration + self.preamble.data0 * bits_cnt[0] +
                self.preamble.data1 * bits_cnt[1])

    def __str__(self):
        return "ReaderFrame{{preamble={} command={}}}".format(
            self.preamble, self.cmd)


class Command(object):

    def __init__(self):
        pass

    @property
    def code(self):
        raise NotImplementedError() 
    
    @property
    def name(self):
        return NotImplementedError()

    def _encode_body(self):
        raise NotImplementedError()

    def encode(self):
        return self.code + self._encode_body()

    @property
    def bitlen(self):
        return len(self.encode())

    def count_bits(self):
        symbols = self.encode()
        cnt = {0: 0, 1: 0}
        for sym in symbols:
            cnt[int(sym)] += 1
        return cnt


class Query(Command):
    def __init__(self, dr, m, trext, sel, session, target, q, crc5=0):
        super().__init__()
        self.dr = dr
        self.m = m
        self.trext = trext
        self.sel = sel
        self.session = session
        self.target = target
        self.q = q
        self.crc5 = crc5

    @property
    def code(self):
        return '1000'

    @property
    def name(self):
        return "Query"
    
    @property
    def bitlen(self):
        return 22

    def _encode_body(self):
        return encode(self.dr) + encode(self.m) + encode(self.trext) + \
               encode(self.sel) + encode(self.session) + \
               encode(self.target) + encode(self.q, width=4) + \
               encode(self.crc5, width=5)

    def __str__(self):
        return "{name}{{dr={dr} m={m} trext={trext} sel={sel} " \
               "session={session} target={target} q={q} " \
               "crc={crc:02X}; encoded={encoded}}}" \
               "".format(name=self.name,
                         dr=('8' if self.dr == DR.DR_8 else '64/3'),
                         m=self.m.name, trext=encode(self.trext),
                         sel=self.sel.name, session=self.session.name,
                         target=self.target.name, q=self.q, crc=self.crc5,
                         encoded=self.encode())


class QueryRep(Command):
    def __init__(self, session):
        super().__init__()
        self.session = session

    @property
    def code(self):
        return '00'

    @property
    def name(self):
        return "QueryRep"
    
    @property
    def bitlen(self):
        return 4

    def _encode_body(self):
        return encode(self.session)

    def __str__(self):
        return "{name}{{session={session}; encoded={encoded}}}".format(
            name=self.name, session=self.session.name, encoded=self.encode())


class Ack(Command):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn

    @property
    def code(self):
        return '01'

    @property
    def name(self):
        return "ACK"
    
    @property
    def bitlen(self):
        return 18

    def _encode_body(self):
        return encode(self.rn, width=16)

    def __str__(self):
        return "{name}{{rn={rn:04X}; encoded={encoded}}}".format(
            name=self.name, rn=self.rn, encoded=self.encode())


class ReqRn(Command):
    def __init__(self, rn, crc16=0):
        super().__init__()
        self.rn = rn
        self.crc16 = crc16

    @property
    def code(self):
        return '11000001'

    @property
    def name(self):
        return "Req_RN"
    
    @property
    def bitlen(self):
        return 40

    def _encode_body(self):
        return encode(self.rn, width=16) + encode(self.crc16, width=16)

    def __str__(self):
        return "{name}{{rn={rn:04X} crc={crc:04X}; encoded={encoded}}}".format(
            name=self.name, rn=self.rn, crc=self.crc16, encoded=self.encode())


class Read(Command):
    def __init__(self, bank, wordptr, wordcnt, rn, crc16=0):
        super().__init__()
        self.bank = bank
        self.wordptr = wordptr
        self.wordcnt = wordcnt
        self.rn = rn
        self.crc16 = crc16

    @property
    def code(self):
        return '11000010'

    @property
    def name(self):
        return "Read"

    def _encode_body(self):
        return encode(self.bank) + encode(self.wordptr, use_ebv=True) + \
               encode(self.wordcnt, width=8) + \
               encode(self.rn, width=16) + encode(self.crc16, width=16)

    def __str__(self):
        return "{name}{{bank={bank} wordptr={ptr:X} " \
               "wordcnt={cnt} rn={rn:04X} crc={crc:04X}; encoded={encoded}}}" \
               "".format(name=self.name, bank=self.bank.name, ptr=self.wordptr,
                         cnt=self.wordcnt, rn=self.rn, crc=self.crc16,
                         encoded=self.encode())


class TagFrame(object):
    def __init__(self, m=None, trext=None, blf=None, reply=None):
        self.reply = reply
        self.m = m
        self.trext = trext
        self.blf = blf

    @property
    def preamble(self):
        if self.m is None:
            raise pyons.errors.MissingFieldError(self.__class__.__name__, 'm')
        if self.trext is None:
            raise pyons.errors.MissingFieldError(
                self.__class__.__name__, 'trext')

        if self.m == TagEncoding.FM0:
            return '1010v1' if not self.trext else '0000000000001010v1'
        else:
            return '0000010111' if not self.trext else '0000000000000000010111'

    @property
    def preamble_bitlen(self):
        if self.m == TagEncoding.FM0:
            return 18 if self.trext else 6
        else:
            return 22 if self.trext else 10

    @property
    def preamble_duration(self):
        return self.preamble_bitlen * self.m.value / self.blf

    def encode(self):
        # 'e' is end-of-signaling 'dummy' data-1
        return self.preamble + self.reply.encode() + 'e'
    
    @property
    def bitlen(self):
        # +1 for end-of-signaling 'dummy' data-1
        return self.preamble_bitlen + self.reply.bitlen + 1

    @property
    def body_bitlen(self):
        return self.reply.bitlen

    @property
    def duration(self):
        return self.bitlen * self.m.value / self.blf

    def __str__(self):
        return "TagFrame{{m={m} trext={trext} blf={blf}KHz reply={reply}}}" \
               "".format(m=self.m.name, trext=(1 if self.trext else 0),
                         blf=(self.blf / 1e3 if self.blf is not None
                              else 'undefined'), reply=self.reply)


class Reply(object):
    def __init__(self):
        pass

    def encode(self):
        raise NotImplementedError()

    @property
    def bitlen(self):
        return len(self.encode())


class Rn16Reply(Reply):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn

    def encode(self):
        return encode(self.rn, width=16)

    @property
    def bitlen(self):
        return 16

    def __str__(self):
        return "Rn16{{rn={rn:04X}}}".format(rn=self.rn)


class AckReply(Reply):
    def __init__(self, epc, pc=0, crc16=0):
        super().__init__()
        self.epc = epc
        self.pc = pc
        self.crc16 = crc16

    def encode(self):
        return encode(self.pc, width=16) + encode(self.epc) + \
               encode(self.crc16, width=16)

    def __str__(self):
        if isinstance(self.epc, collections.Iterable) \
                and not isinstance(self.epc, str):
            epc = "".join([format(x, '02X') for x in self.epc])
        else:
            epc = self.epc
        return "AckReply{{pc={pc:04X} epc={epc} crc={crc:04X}}}" \
               "".format(pc=self.pc, epc=epc, crc=self.crc16)


class ReqRnReply(Reply):
    def __init__(self, rn, crc16=0):
        super().__init__()
        self.rn = rn
        self.crc16 = crc16

    def encode(self):
        return encode(self.rn, width=16) + encode(self.crc16, width=16)

    @property
    def bitlen(self):
        return 32

    def __str__(self):
        return "ReqRnReply{{rn={rn:04X} crc={crc:04X}}}".format(
            rn=self.rn, crc=self.crc16)


class ReadReply(Reply):
    def __init__(self, words, rn, crc16, header=0):
        super().__init__()
        self.header = header
        self.words = words
        self.rn = rn
        self.crc16 = crc16

    def encode(self):
        return encode(self.header, width=1) + encode(self.words) + \
               encode(self.rn, width=16) + \
               encode(self.crc16, width=16)

    def __str__(self):
        if isinstance(self.words, collections.Iterable):
            words = "".join([format(x, '02X') for x in self.words])
        else:
            words = self.words
        return "ReadReply{{header={header:01X} words={words} rn={rn:04X} " \
               "crc={crc:04X}}}".format(header=self.header, words=words,
                                        rn=self.rn, crc=self.crc16)


def min_t1(rtcal, blf, frt=0.1):
    return max(rtcal, 10.0/blf) * (1. - frt) - 2e-6


def nominal_t1(rtcal, blf):
    return max(rtcal, 10/blf)


def max_t1(rtcal, blf, frt=0.1):
    return max(rtcal, 10.0/blf) * (1. + frt) + 2e-6


def min_t2(blf):
    return 3./blf


def max_t2(blf):
    return 20./blf


def t3():
    return 0.


def t4(rtcal):
    return 2.*rtcal
