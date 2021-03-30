from dataclasses import dataclass
from typing import Tuple

from rfidlib.protocol.symbols import InventoryFlag


@dataclass
class RoundSpec:
    flag: InventoryFlag
    turn_off: bool = False

    @property
    def as_tuple(self):
        return self.flag, self.turn_off


def parse_scenario(s: str) -> Tuple[RoundSpec, ...]:
    """
    Parse string that encodes a scenario into a sequence of RoundSpec.

    Scenario specification looks like "AABBxABx", where symbols 'A' and 'B'
    specify inventory flag, and 'x' indicates that reader turns off in after
    the round. Denoting round spec as (X, e), where X = A,B and e = True
    iff reader turns off.

    Example
    -------
    >>> parse_scenario("AB")
    ((A, False), (B, False))

    >>> parse_scenario("ABBxAAx")
    ((A, False), (B, False), (B, True), (A, False), (A, True))

    Parameters
    ----------
    s : str

    Returns
    -------
    scenario : tuple of RoundSpec
    """
    if s == "":
        return ()
    pos, max_pos = 0, len(s) - 1
    scenario = []
    while pos <= max_pos:
        flag = InventoryFlag.parse(s[pos])
        if pos < max_pos and s[pos + 1] == 'x':
            turn_off = True
            pos += 2
        else:
            turn_off = False
            pos += 1
        scenario.append(RoundSpec(flag, turn_off))
    return tuple(scenario)
