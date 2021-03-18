from collections import namedtuple
from dataclasses import dataclass, field
from typing import Sequence, Set, Optional, List, Tuple
import numpy as np
from rfidlib.protocol.symbols import InventoryFlag

from rfidam.inventory import get_rx_prob
from rfidam.protocol import Protocol


@dataclass
class ModelParams:
    protocol: Protocol
    arrivals: Sequence[float]
    speed: float
    length: float
    scenario: str  # e.g. AABBxAB, x - turn off period
    ber: float


@dataclass
class RoundRecord:
    index: int
    duration: float = 0.0
    started_at: float = 0.0
    flag: InventoryFlag = InventoryFlag.A
    active_tags: Set[int] = field(default_factory=set)
    identified_tags: Set[int] = field(default_factory=set)


@dataclass
class TagRecord:
    index: Optional[int] = None
    arrived_at: float = 0.0
    departed_at: float = 0.0
    identified: bool = False
    n_id: int = 0
    first_round: int = -1
    last_round: int = -1


class Journal:
    def __init__(self):
        self.rounds: List[RoundRecord] = list()
        self.tags: List[TagRecord] = list()
        self._next_round_index = 0
        self._next_tag_index = 0

    def add_round(self, index: int) -> RoundRecord:
        record = RoundRecord(index=index)
        self.rounds.append(record)
        return record

    def add_tag(self, index: int) -> TagRecord:
        record = TagRecord(index=index)
        self.tags.append(record)
        return record


@dataclass
class TagState:
    index: int
    flag: InventoryFlag
    active: bool
    position: float
    record: TagRecord


@dataclass
class State:
    time: float
    reader_flag: InventoryFlag
    scenario_offset: int = 0
    next_arrival_index: int = 0
    next_tag_index: int = 0
    next_round_index: int = 0
    tags: Sequence[TagState] = field(default_factory=list)



def simulate(params: ModelParams):
    """
    Simulate moving tags identification.
    """
    arrivals = np.asarray(params.arrivals)

    # Check arrival timestamps are ascending:
    if arrivals[0] < 0 or ((arrivals[1:] - arrivals[:-1]) < 0).any():
        raise ValueError("arrival timestamps must be positive ascending")

    # Check at least one symbol in scenario is A or B:
    if 'A' not in params.scenario and 'B' not in params.scenario:
        raise ValueError("at least one A or B symbol must be at scenario")

    # Since A or B in scenario, find the first A or B and set the time
    # and offset correspondingly
    scenario_offset = 0
    while params.scenario[scenario_offset] not in ['A', 'B']:
        scenario_offset += 1

    # Create state with scenario offset found above, create journal:
    state = State(
        time=(params.protocol.props.t_off * scenario_offset),
        scenario_offset=scenario_offset,
        reader_flag=InventoryFlag.parse(params.scenario[scenario_offset])
    )
    journal = Journal()

    if len(params.arrivals) == 0:
        return journal

    # Initialize next arrival:
    next_arrival = arrivals[state.next_arrival_index]

    while state.time < next_arrival:
        # Check current scenario symbol:
        # - if it is A or B, then set inventory flag and start round
        # - otherwise, turn reader off
        symbol = params.scenario[reader.scenario_offset].upper()
        if symbol == 'X':
            time += params.protocol.props.t_off
            for tag in curr_tags:
                tag.flag = InventoryFlag.A
        else:
            reader.flag = InventoryFlag.A if symbol == 'A' else InventoryFlag.B
            simulate_round(state, params.protocol, params.ber)


def simulate_round(state: State, params: ModelParams, journal: Journal) -> None:
    # Choose active tags (flag equals to reader flag) and initialize their
    # slot counters:
    n_slots = params.protocol.props.n_slots
    active_tags = [tag for tag in state.tags if tag.flag == state.reader_flag]
    slots = np.random.randint(0, n_slots, size=len(active_tags))
    curr_slot = 0

    # Extract protocol parameters those are used often:
    t1 = params.protocol.timings.t1
    t2 = params.protocol.timings.t2
    t3 = params.protocol.timings.t3
    rt_link = params.protocol.rt_link
    tr_link = params.protocol.tr_link

    round_index = state.next_round_index
    state.next_round_index += 1

    round_record = journal.add_round(round_index)
    state.next_round_index += 1
    round_record.flag = state.reader_flag
    round_record.started_at = state.time
    round_record.active_tags = [tag.index for tag in active_tags]
    for tag in state.tags:
        if tag.record.first_round < 0:
            tag.record.first_round = round_index
        tag.record.last_round = round_index

    # Start slots
    next_query_cmd = rt_link.query
    round_duration = 0.0
    while curr_slot < n_slots:
        # Transmit Query/QueryRep and wait T1:
        round_duration += next_query_cmd.duration + t1
        # Select tags those are going to respond in this slot:
        reply_tags = [t for i, t in enumerate(active_tags) if slots[i] == 0]
        n_reply_tags = len(reply_tags)
        # If no tags are going to reply, just wait T3 more:
        if n_reply_tags == 0:
            round_duration += t3
        elif n_reply_tags > 1:
            round_duration += tr_link.rn16.duration + t2
        else:
            # Single tag is replying
            reply_slot_ret = simulate_reply_slot(params.protocol, params.ber)
            round_duration += reply_slot_ret.duration
            if reply_slot_ret.identified:
                reply_tags[0].record.identified = True
                reply_tags[0].record.n_id += 1



ReplySlotSimRet = namedtuple('ReplySlotSimRet', (
    'duration',
    'invert_flag',
    'identified'
))


def simulate_reply_slot(protocol: Protocol, ber: float) -> ReplySlotSimRet:
    rt_link = protocol.rt_link
    tr_link = protocol.tr_link
    t1 = protocol.timings.t1
    t2 = protocol.timings.t2

    duration = tr_link.rn16.duration + t2
    if np.random.random_sample() > get_rx_prob(tr_link.rn16, ber):
        return ReplySlotSimRet(duration, invert_flag=False, identified=False)

    duration += rt_link.ack.duration + t1 + tr_link.epc.duration + t2
    if np.random.random_sample() > get_rx_prob(tr_link.epc, ber):
        return ReplySlotSimRet(duration, invert_flag=True, identified=False)

    if not protocol.props.use_tid:
        return ReplySlotSimRet(duration, invert_flag=True, identified=True)

    duration += rt_link.req_rn.duration + t1 + tr_link.handle.duration + t2
    if np.random.random_sample() > get_rx_prob(tr_link.handle, ber):
        return ReplySlotSimRet(duration, invert_flag=True, identified=True)

    duration += rt_link.read.duration + t1 + tr_link.data.duration + t2
    identified = np.random.random_sample() > get_rx_prob(tr_link.data, ber)
    return ReplySlotSimRet(duration, invert_flag=True, identified=identified)


def choose_slots(tags: Sequence[TagState], protocol: Protocol) -> None:
    """
    Assign random slots to tags.
    """
    for tag in tags:
        tag.slot = np.random.randint(0, protocol.props.n_slots)


def create_tag(time: float, journal: Journal):