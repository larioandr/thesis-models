from dataclasses import dataclass, field
from typing import Sequence, List, Set
import numpy as np
from rfidlib.protocol.symbols import InventoryFlag

from rfidam.inventory import create_round_model, create_slot_model
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
    duration: float
    started_at: float
    flag: InventoryFlag
    active_tags: Set[int] = field(default_factory=set)
    identified_tags: Set[int] = field(default_factory=set)


@dataclass
class TagRecord:
    index: int
    arrived_at: float = 0.0
    departed_at: float = 0.0
    identified: bool = False
    n_id: int = 0
    first_round: int = 0
    last_round: int = 0


@dataclass
class Journal:
    rounds: Sequence[RoundRecord] = field(default_factory=list)
    tags: Sequence[TagRecord] = field(default_factory=list)


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
            active_tags = [tag for tag in curr_tags if tag.flag == reader.flag]

            # Assign slots to tags:
            for tag in active_tags:
                tag.slot = np.random.randint(0, params.protocol.props.n_slots)

            # Start inventory round:
            reader.slot = 0
            query_cmd = params.protocol.rt_link.query
            while reader.slot < params.protocol.props.n_slots:
                # Transmit Query/QueryRep and wait T1:
                time += query_cmd.duration + params.protocol.timings.t1
                query_cmd = params.protocol.rt_link.query_rep
                # Check whether any tag transmits:
                reply_tags = [tag for tag in active_tags if tag.slot == 0]
                if len(reply_tags) == 0:
                    time += params.protocol.timings.t3
                elif len(reply_tags) > 1:
                    time += params.protocol.tr_link.rn16
                    time += params.protocol.timings.t2
                else:
                    pass  # TODO


def simulate_round(state: State, protocol: Protocol, ber: float) -> None:
    # Choose active tags (flag equals to reader flag) and initialize their
    # slot counters:
    n_slots = protocol.props.n_slots
    active_tags = [tag for tag in state.tags if tag.flag == state.reader_flag]
    slots = np.random.randint(0, n_slots, size=len(active_tags))
    curr_slot = 0

    # Extract protocol parameters those are used often:
    t1 = protocol.timings.t1
    t2 = protocol.timings.t2
    t3 = protocol.timings.t3
    rt_link = protocol.rt_link
    tr_link = protocol.tr_link

    # Start slots
    next_query_cmd = rt_link.query
    while curr_slot < n_slots:
        # Transmit Query/QueryRep and wait T1:
        state.time += next_query_cmd.duration + t1
        # Select tags those are going to respond in this slot:
        reply_tags = [t for i, t in enumerate(active_tags) if slots[i] == 0]
        n_reply_tags = len(reply_tags)
        # If no tags are going to reply, just wait T3 more:
        if n_reply_tags == 0:
            state.time += t3
        elif n_reply_tags > 1:
            state.time += tr_link.rn16.duration + t2
        else:
            # Single tag is replying
            pass # TODO


def choose_slots(tags: Sequence[TagState], protocol: Protocol) -> None:
    """
    Assign random slots to tags.
    """
    for tag in tags:
        tag.slot = np.random.randint(0, protocol.props.n_slots)


def create_tag(time: float, journal: Journal):