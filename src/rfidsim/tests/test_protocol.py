import unittest

import pyons
from rfidsim import protocol as gen2


class TestEncode(unittest.TestCase):
    def test_encode_dr(self):
        self.assertEqual(gen2.encode(gen2.DR.DR_8), '0')
        self.assertEqual(gen2.encode(gen2.DR.DR_643), '1')

    def test_encode_m(self):
        self.assertEqual(gen2.encode(gen2.TagEncoding.FM0), '00')
        self.assertEqual(gen2.encode(gen2.TagEncoding.M2), '01')
        self.assertEqual(gen2.encode(gen2.TagEncoding.M4), '10')
        self.assertEqual(gen2.encode(gen2.TagEncoding.M8), '11')

    def test_encode_bank(self):
        self.assertEqual(gen2.encode(gen2.Bank.RESERVED), '00')
        self.assertEqual(gen2.encode(gen2.Bank.EPC), '01')
        self.assertEqual(gen2.encode(gen2.Bank.TID), '10')
        self.assertEqual(gen2.encode(gen2.Bank.USER), '11')

    def test_encode_inventory_flag(self):
        self.assertEqual(gen2.encode(gen2.InventoryFlag.A), '0')
        self.assertEqual(gen2.encode(gen2.InventoryFlag.B), '1')

    def test_encode_sel(self):
        self.assertIn(gen2.encode(gen2.Sel.SL_ALL), ['00', '01'])
        self.assertEqual(gen2.encode(gen2.Sel.SL_YES), '11')
        self.assertEqual(gen2.encode(gen2.Sel.SL_NO), '10')

    def test_encode_session(self):
        self.assertEqual(gen2.encode(gen2.Session.S0), '00')
        self.assertEqual(gen2.encode(gen2.Session.S1), '01')
        self.assertEqual(gen2.encode(gen2.Session.S2), '10')
        self.assertEqual(gen2.encode(gen2.Session.S3), '11')

    def test_encode_ebv(self):
        self.assertEqual(gen2.encode(0, use_ebv=True), '00000000')
        self.assertEqual(gen2.encode(1, use_ebv=True), '00000001')
        self.assertEqual(gen2.encode(127, use_ebv=True), '01111111')
        self.assertEqual(gen2.encode(128, use_ebv=True), '1000000100000000')
        self.assertEqual(gen2.encode(16383, use_ebv=True), '1111111101111111')
        self.assertEqual(gen2.encode(16384, use_ebv=True), '100000011000000000000000')

    def test_encode_str(self):
        self.assertEqual(gen2.encode(''), '')
        self.assertEqual(gen2.encode('89ABCDEF'), '10001001101010111100110111101111')

    def test_encode_list(self):
        self.assertEqual(gen2.encode([]), '')
        self.assertEqual(gen2.encode([0xFE, 0xDC, 0xBA, 0x98]), '11111110110111001011101010011000')


class TestReaderPreambles(unittest.TestCase):
    def setUp(self):
        self.delim = 12.5e-6
        self.tari = 6.25e-6
        self.data0 = self.tari
        self.data1 = 2 * self.data0
        self.rtcal = self.data0 + self.data1
        self.trcal = 2 * self.rtcal
        self.tol = 1e-9

    def test_sync_created_with_rtcal_specified(self):
        sync = gen2.ReaderFrame.Sync(tari=self.tari, rtcal=self.rtcal, delim=self.delim)
        self.assertAlmostEqual(sync.delim, self.delim, delta=self.tol)
        self.assertAlmostEqual(sync.data0, self.data0, delta=self.tol)
        self.assertAlmostEqual(sync.data1, self.data1, delta=self.tol)
        self.assertAlmostEqual(sync.rtcal, self.rtcal, delta=self.tol)
        self.assertAlmostEqual(sync.tari, self.tari, delta=self.tol)
        self.assertAlmostEqual(sync.duration, self.delim + self.tari + self.rtcal, delta=self.tol)

    def test_sync_created_with_data1_multiplier(self):
        sync = gen2.ReaderFrame.Sync(tari=self.tari, data1_multiplier=self.data1 / self.data0, delim=self.delim)
        self.assertAlmostEqual(sync.delim, self.delim, delta=self.tol)
        self.assertAlmostEqual(sync.data0, self.data0, delta=self.tol)
        self.assertAlmostEqual(sync.data1, self.data1, delta=self.tol)
        self.assertAlmostEqual(sync.rtcal, self.rtcal, delta=self.tol)
        self.assertAlmostEqual(sync.tari, self.tari, delta=self.tol)
        self.assertAlmostEqual(sync.duration, self.delim + self.tari + self.rtcal, delta=self.tol)

    def test_preamble_with_trcal_specified(self):
        preamble = gen2.ReaderFrame.Preamble(tari=self.tari, rtcal=self.rtcal, trcal=self.trcal, delim=self.delim)
        self.assertAlmostEqual(preamble.delim, self.delim, delta=self.tol)
        self.assertAlmostEqual(preamble.data0, self.data0, delta=self.tol)
        self.assertAlmostEqual(preamble.data1, self.data1, delta=self.tol)
        self.assertAlmostEqual(preamble.rtcal, self.rtcal, delta=self.tol)
        self.assertAlmostEqual(preamble.tari, self.tari, delta=self.tol)
        self.assertAlmostEqual(preamble.trcal, self.trcal, delta=self.tol)
        self.assertAlmostEqual(preamble.duration, self.delim + self.tari + self.rtcal + self.trcal, delta=self.tol)

    def test_preamble_with_trcal_multiplier(self):
        preamble = gen2.ReaderFrame.Preamble(tari=self.tari, rtcal=self.rtcal,
                                             trcal_multiplier=self.trcal/self.rtcal, delim=self.delim)
        self.assertAlmostEqual(preamble.delim, self.delim, delta=self.tol)
        self.assertAlmostEqual(preamble.data0, self.data0, delta=self.tol)
        self.assertAlmostEqual(preamble.data1, self.data1, delta=self.tol)
        self.assertAlmostEqual(preamble.rtcal, self.rtcal, delta=self.tol)
        self.assertAlmostEqual(preamble.tari, self.tari, delta=self.tol)
        self.assertAlmostEqual(preamble.trcal, self.trcal, delta=self.tol)
        self.assertAlmostEqual(preamble.duration, self.delim + self.tari + self.rtcal + self.trcal, delta=self.tol)


class TestQueryRepCommand(unittest.TestCase):
    def setUp(self):
        self.data0 = 6.25e-6
        self.data1 = 2 * self.data0
        self.sync = gen2.ReaderFrame.Sync(tari=self.data0, data1_multiplier=self.data1 / self.data0)
        self.cmd1 = gen2.QueryRep(gen2.Session.S0)
        self.cmd2 = gen2.QueryRep(gen2.Session.S3)
        self.frame1 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd1)
        self.frame2 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd2)
        self.tol = 1e-9

    def test_encoding(self):
        self.assertEqual(self.cmd1.encode(), "0000")
        self.assertEqual(self.cmd2.encode(), "0011")

    def test_bitlen(self):
        self.assertEqual(len(self.cmd1.encode()), self.cmd1.bitlen)

    def test_duration(self):
        pd = self.sync.duration
        d0 = self.data0
        d1 = self.data1
        self.assertAlmostEqual(self.frame1.duration, pd + d0 * 4, delta=self.tol)
        self.assertAlmostEqual(self.frame2.duration, pd + (d0 + d1) * 2, delta=self.tol)


class TestQueryCommand(unittest.TestCase):
    def setUp(self):
        self.data0 = 6.25e-6
        self.data1 = 2 * self.data0
        self.preamble = gen2.ReaderFrame.Preamble(tari=self.data0, data1_multiplier=self.data1 / self.data0,
                                                  trcal_multiplier=2.)
        self.cmd1 = gen2.Query(dr=gen2.DR.DR_8, m=gen2.TagEncoding.FM0, trext=False, sel=gen2.Sel.SL_ALL,
                               session=gen2.Session.S0, target=gen2.InventoryFlag.A, q=0, crc5=0)
        self.cmd2 = gen2.Query(dr=gen2.DR.DR_643, m=gen2.TagEncoding.FM0, trext=True, sel=gen2.Sel.SL_ALL,
                               session=gen2.Session.S3, target=gen2.InventoryFlag.A, q=15, crc5=0)
        self.cmd3 = gen2.Query(dr=gen2.DR.DR_8, m=gen2.TagEncoding.M8, trext=False, sel=gen2.Sel.SL_YES,
                               session=gen2.Session.S0, target=gen2.InventoryFlag.B, q=0, crc5=31)
        self.cmd4 = gen2.Query(dr=gen2.DR.DR_643, m=gen2.TagEncoding.M8, trext=True, sel=gen2.Sel.SL_YES,
                               session=gen2.Session.S3, target=gen2.InventoryFlag.B, q=15, crc5=31)
        self.frame1 = gen2.ReaderFrame(preamble=self.preamble, cmd=self.cmd1)
        self.frame2 = gen2.ReaderFrame(preamble=self.preamble, cmd=self.cmd2)
        self.frame3 = gen2.ReaderFrame(preamble=self.preamble, cmd=self.cmd3)
        self.frame4 = gen2.ReaderFrame(preamble=self.preamble, cmd=self.cmd4)
        self.tol = 1e-9

    def test_encoding(self):
        self.assertEqual(self.cmd1.encode(), "1000000000000000000000")
        self.assertEqual(self.cmd2.encode(), "1000100100110111100000")
        self.assertEqual(self.cmd3.encode(), "1000011011001000011111")
        self.assertEqual(self.cmd4.encode(), "1000111111111111111111")

    def test_count_bits(self):
        self.assertEqual(self.cmd1.count_bits(), {0: 21, 1: 1})
        self.assertEqual(self.cmd2.count_bits(), {0: 13, 1: 9})
        self.assertEqual(self.cmd3.count_bits(), {0: 11, 1: 11})
        self.assertEqual(self.cmd4.count_bits(), {0: 3, 1: 19})

    def test_bitlen(self):
        self.assertEqual(len(self.cmd1.encode()), self.cmd1.bitlen)

    def test_duration(self):
        pd = self.preamble.duration
        d0 = self.data0
        d1 = self.data1
        self.assertAlmostEqual(self.frame1.duration, pd + d0 * 21 + d1 * 1, delta=self.tol)
        self.assertAlmostEqual(self.frame2.duration, pd + d0 * 13 + d1 * 9, delta=self.tol)
        self.assertAlmostEqual(self.frame3.duration, pd + d0 * 11 + d1 * 11, delta=self.tol)
        self.assertAlmostEqual(self.frame4.duration, pd + d0 * 3 + d1 * 19, delta=self.tol)


class TestAckCommand(unittest.TestCase):
    def setUp(self):
        self.data0 = 6.25e-6
        self.data1 = 2 * self.data0
        self.tol = 1e-9
        self.sync = gen2.ReaderFrame.Sync(tari=self.data0, data1_multiplier=self.data1 / self.data0)
        self.cmd1 = gen2.Ack(rn=0x0000)
        self.cmd2 = gen2.Ack(rn=0xFFFF)
        self.frame1 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd1)
        self.frame2 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd2)

    def test_encoding(self):
        self.assertEqual(self.cmd1.encode(), '010000000000000000')
        self.assertEqual(self.cmd2.encode(), '011111111111111111')

    def test_count_bits(self):
        self.assertEqual(self.cmd1.count_bits(), {0: 17, 1: 1})
        self.assertEqual(self.cmd2.count_bits(), {0: 1, 1: 17})

    def test_bitlen(self):
        self.assertEqual(len(self.cmd1.encode()), self.cmd1.bitlen)

    def test_duration(self):
        pd = self.sync.duration
        d0 = self.data0
        d1 = self.data1
        self.assertAlmostEqual(self.frame1.duration, pd + 17*d0 + d1, delta=self.tol)
        self.assertAlmostEqual(self.frame2.duration, pd + d0 + 17*d1, delta=self.tol)


class TestReqRnCommand(unittest.TestCase):
    def setUp(self):
        self.data0 = 6.25e-6
        self.data1 = 2 * self.data0
        self.tol = 1e-9
        self.sync = gen2.ReaderFrame.Sync(tari=self.data0, data1_multiplier=self.data1 / self.data0)
        self.cmd1 = gen2.ReqRn(0x0000, 0x0000)
        self.cmd2 = gen2.ReqRn(0xFFFF, 0x0000)
        self.cmd3 = gen2.ReqRn(0x0000, 0xFFFF)
        self.cmd4 = gen2.ReqRn(0xFFFF, 0xFFFF)
        self.frame1 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd1)
        self.frame2 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd2)
        self.frame3 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd3)
        self.frame4 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd4)

    def test_encoding(self):
        self.assertEqual(self.cmd1.encode(), '1100000100000000000000000000000000000000')
        self.assertEqual(self.cmd2.encode(), '1100000111111111111111110000000000000000')
        self.assertEqual(self.cmd3.encode(), '1100000100000000000000001111111111111111')
        self.assertEqual(self.cmd4.encode(), '1100000111111111111111111111111111111111')

    def test_count_bits(self):
        self.assertEqual(self.cmd1.count_bits(), {0: 37, 1: 3})
        self.assertEqual(self.cmd2.count_bits(), {0: 21, 1: 19})
        self.assertEqual(self.cmd3.count_bits(), {0: 21, 1: 19})
        self.assertEqual(self.cmd4.count_bits(), {0: 5, 1: 35})

    def test_bitlen(self):
        self.assertEqual(len(self.cmd1.encode()), self.cmd1.bitlen)

    def test_duration(self):
        pd = self.sync.duration
        d0 = self.data0
        d1 = self.data1
        self.assertAlmostEqual(self.frame1.duration, pd + 37 * d0 + 3 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame2.duration, pd + 21 * d0 + 19 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame3.duration, pd + 21 * d0 + 19 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame4.duration, pd + 5 * d0 + 35 * d1, delta=self.tol)


class TestReadCommand(unittest.TestCase):
    def setUp(self):
        self.data0 = 6.25e-6
        self.data1 = 2 * self.data0
        self.tol = 1e-9
        self.sync = gen2.ReaderFrame.Sync(tari=self.data0, data1_multiplier=self.data1 / self.data0)
        self.cmd1 = gen2.Read(bank=gen2.Bank.RESERVED, wordptr=0, wordcnt=0x00, rn=0x0000, crc16=0x0000)
        self.cmd2 = gen2.Read(bank=gen2.Bank.RESERVED, wordptr=16383, wordcnt=0x00, rn=0x0000, crc16=0x0000)
        self.cmd3 = gen2.Read(bank=gen2.Bank.RESERVED, wordptr=16384, wordcnt=0x00, rn=0x0000, crc16=0x0000)
        self.cmd4 = gen2.Read(bank=gen2.Bank.USER, wordptr=0, wordcnt=0xFF, rn=0x0000, crc16=0xFFFF)
        self.cmd5 = gen2.Read(bank=gen2.Bank.RESERVED, wordptr=127, wordcnt=0x00, rn=0xFFFF, crc16=0x0000)
        self.cmd6 = gen2.Read(bank=gen2.Bank.USER, wordptr=127, wordcnt=0xFF, rn=0xFFFF, crc16=0xFFFF)
        self.frame1 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd1)
        self.frame2 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd2)
        self.frame3 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd3)
        self.frame4 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd4)
        self.frame5 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd5)
        self.frame6 = gen2.ReaderFrame(preamble=self.sync, cmd=self.cmd6)
        self.frame_no_preamble = gen2.ReaderFrame(preamble=None, cmd=self.cmd1)

    def test_encoding(self):
        self.assertEqual(self.cmd1.encode(), '1100001000000000000000000000000000000000000000000000000000')
        self.assertEqual(self.cmd2.encode(), '110000100011111111011111110000000000000000000000000000000000000000')
        self.assertEqual(self.cmd3.encode(),
                         '11000010001000000110000000000000000000000000000000000000000000000000000000')
        self.assertEqual(self.cmd4.encode(), '1100001011000000001111111100000000000000001111111111111111')
        self.assertEqual(self.cmd5.encode(), '1100001000011111110000000011111111111111110000000000000000')
        self.assertEqual(self.cmd6.encode(), '1100001011011111111111111111111111111111111111111111111111')

    def test_count_bits(self):
        self.assertEqual(self.cmd1.count_bits(), {0: 55, 1: 3})
        self.assertEqual(self.cmd2.count_bits(), {0: 48, 1: 18})
        self.assertEqual(self.cmd3.count_bits(), {0: 68, 1: 6})
        self.assertEqual(self.cmd4.count_bits(), {0: 29, 1: 29})
        self.assertEqual(self.cmd5.count_bits(), {0: 32, 1: 26})
        self.assertEqual(self.cmd6.count_bits(), {0: 6, 1: 52})

    def test_bitlen(self):
        self.assertEqual(len(self.cmd1.encode()), self.cmd1.bitlen)

    def test_duration(self):
        pd = self.sync.duration
        d0 = self.data0
        d1 = self.data1
        with self.assertRaises(pyons.errors.MissingFieldError):
            # noinspection PyStatementEffect
            self.frame_no_preamble.duration
        self.assertAlmostEqual(self.frame1.duration, pd + 55 * d0 + 3 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame2.duration, pd + 48 * d0 + 18 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame3.duration, pd + 68 * d0 + 6 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame4.duration, pd + 29 * d0 + 29 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame5.duration, pd + 32 * d0 + 26 * d1, delta=self.tol)
        self.assertAlmostEqual(self.frame6.duration, pd + 6 * d0 + 52 * d1, delta=self.tol)


class TestTagFramePreamble(unittest.TestCase):

    def setUp(self):
        self.tol = 1e-9

    def test_fm0_frame(self):
        for trext in [False, True]:
            preamble_prefix = '' if not trext else '000000000000'
            frame = gen2.TagFrame(m=gen2.TagEncoding.FM0, trext=trext)
            self.assertEqual(frame.preamble, preamble_prefix + '1010v1')
            self.assertEqual(frame.preamble_bitlen, 6 if not trext else 18)
            for blf in [40e6, 160e6, 320e6, 640e6]:
                frame.blf = blf
                self.assertAlmostEqual(frame.preamble_duration, frame.preamble_bitlen / blf, delta=self.tol)

    def test_miller_frame(self):
        for trext in [False, True]:
            for m in [gen2.TagEncoding.M2, gen2.TagEncoding.M4, gen2.TagEncoding.M8]:
                preamble_prefix = '0000' if not trext else '0000000000000000'
                frame = gen2.TagFrame(m=m, trext=trext)
                self.assertEqual(frame.preamble, preamble_prefix + '010111')
                self.assertEqual(frame.preamble_bitlen, 10 if not trext else 22)
                for blf in [40e6, 160e6, 320e6, 640e6]:
                    frame.blf = blf
                    self.assertAlmostEqual(frame.preamble_duration, frame.preamble_bitlen / blf * m.value,
                                           delta=self.tol)


class TestRn16Reply(unittest.TestCase):
    def setUp(self):
        self.msg1 = gen2.Rn16Reply(rn=0x0000)
        self.msg2 = gen2.Rn16Reply(rn=0xFFFF)

    def test_encoding(self):
        self.assertEqual(self.msg1.encode(), '0000000000000000')
        self.assertEqual(self.msg2.encode(), '1111111111111111')

    def test_bitlen(self):
        self.assertEqual(self.msg1.bitlen, 16)


class TestAckReply(unittest.TestCase):
    def setUp(self):
        self.msg1 = gen2.AckReply(epc=[0x00], pc=0x0000, crc16=0x0000)
        self.msg2 = gen2.AckReply(epc=[0xFF, 0xFF], pc=0x0000, crc16=0xFFFF)
        self.msg3 = gen2.AckReply(epc='ABCDEF', pc=0xFFFF, crc16=0x0000)

    def test_encoding(self):
        self.assertEqual(self.msg1.encode(), '0000000000000000000000000000000000000000')
        self.assertEqual(self.msg2.encode(), '000000000000000011111111111111111111111111111111')
        self.assertEqual(self.msg3.encode(), '11111111111111111010101111001101111011110000000000000000')

    def test_bitlen(self):
        self.assertEqual(self.msg1.bitlen, 40)
        self.assertEqual(self.msg2.bitlen, 48)
        self.assertEqual(self.msg3.bitlen, 56)


class TestReqRnReply(unittest.TestCase):
    def setUp(self):
        self.msg1 = gen2.ReqRnReply(rn=0x0000, crc16=0x0000)
        self.msg2 = gen2.ReqRnReply(rn=0xFFFF, crc16=0x0000)
        self.msg3 = gen2.ReqRnReply(rn=0x0000, crc16=0xFFFF)
        self.msg4 = gen2.ReqRnReply(rn=0xFFFF, crc16=0xFFFF)

    def test_body_encoding(self):
        self.assertEqual(self.msg1.encode(), '00000000000000000000000000000000')
        self.assertEqual(self.msg2.encode(), '11111111111111110000000000000000')
        self.assertEqual(self.msg3.encode(), '00000000000000001111111111111111')
        self.assertEqual(self.msg4.encode(), '11111111111111111111111111111111')

    def test_bitlen(self):
        self.assertEqual(self.msg1.bitlen, 32)
        self.assertEqual(self.msg1.bitlen, self.msg2.bitlen)
        self.assertEqual(self.msg1.bitlen, self.msg3.bitlen)
        self.assertEqual(self.msg1.bitlen, self.msg4.bitlen)


class TestReadReply(unittest.TestCase):
    def setUp(self):
        self.msg1 = gen2.ReadReply(words=[], rn=0x0000, crc16=0x0000)
        self.msg2 = gen2.ReadReply(words='FFFFFFFF', rn=0x0000, crc16=0xFFFF)
        self.msg3 = gen2.ReadReply(words=[0xFF, 0x00], rn=0xFFFF, crc16=0x0000)

    def test_encoding(self):
        self.assertEqual(self.msg1.encode(), '000000000000000000000000000000000')
        self.assertEqual(self.msg2.encode(), '01111111111111111111111111111111100000000000000001111111111111111')
        self.assertEqual(self.msg3.encode(), '0111111110000000011111111111111110000000000000000')

    def test_bitlen(self):
        self.assertEqual(self.msg1.bitlen, 33)
        self.assertEqual(self.msg2.bitlen, 65)
        self.assertEqual(self.msg3.bitlen, 49)
