from dataclasses import dataclass

from rfidlib.protocol.symbols import TagEncoding




@dataclass
class Parameters:
    tari: float
    rtcal: float
    trcal: float
    m: TagEncoding
    trext: bool
