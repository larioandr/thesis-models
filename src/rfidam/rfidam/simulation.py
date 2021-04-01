from dataclasses import dataclass, field
from typing import Sequence, Set, Optional, List, Tuple, Dict, Any
import numpy as np
from rfidlib.protocol.symbols import InventoryFlag

from rfidam.inventory import get_rx_prob
from rfidam.protocol import Protocol
from rfidam.scenario import RoundSpec


@dataclass
class ModelParams:
    protocol: Protocol
    arrivals: Sequence[float]
    speed: float
    length: float
    scenario: Tuple[RoundSpec]
    ber: float


@dataclass
class TagJournalRecord:
    arrived_at: float = 0.0       # model time when tag arrived
    departed_at: float = 0.0      # model time when tag departed
    first_round_index: int = 0    # first round index (simulation-wide)
    num_rounds_total: int = 0     # total number of rounds the tag was in area
    num_rounds_active: int = 0    # number of rounds the tag participated in
    num_identified: int = 0       # number of times the tag was completely read


@dataclass
class RoundJournalRecord:
    index: int
    num_tags_total: int = 0
    num_tags_active: int = 0
    num_tags_identified: int = 0
    started_at: float = 0.0
    duration: float = 0.0         # without power off, even if it is needed
    num_collisions: int = 0


@dataclass
class ScenarioStats:
    means: np.ndarray
    errors: np.ndarray
    num_samples: np.ndarray

    @staticmethod
    def build(all_samples: Sequence[Sequence[float]]) -> "ScenarioStats":
        num_samples = np.asarray([len(samples) for samples in all_samples])
        all_samples = [np.asarray(samples) if len(samples) > 0 else np.zeros(1)
                       for samples in all_samples]
        return ScenarioStats(
            means=np.asarray([samples.mean() for samples in all_samples]),
            errors=np.asarray([samples.std() for samples in all_samples]),
            num_samples=num_samples
        )


@dataclass
class Journal:
    rounds: List[RoundJournalRecord] = field(default_factory=list)
    tags: List[TagJournalRecord] = field(default_factory=list)

    def durations(self, scenario_length: int) -> ScenarioStats:
        """
        Get a list of round durations grouped by the offset in scenario.

        Scenario length may be any positive number, but it is reasonable
        to set it equal to `(K * len(params.scenario))`.

        Returns
        -------
        stats : ScenarioStats
        """
        durations = [list() for _ in range(scenario_length)]
        for round_ in self.rounds:
            durations[round_.index % scenario_length].append(round_.duration)
        return ScenarioStats.build(durations)

    def id_prob(self):
        """
        Get probability that tag was identified, no matter when arrived.
        """
        num_tags = len(self.tags)
        num_identified_tags = \
            len([tag for tag in self.tags if tag.num_identified > 0])
        return num_identified_tags / num_tags if num_tags > 0 else 1.0

    def id_probs(self, scenario_length) -> np.ndarray:
        """
        Get probabilities that tag was identified depending on the round
        offset when it arrived.
        """
        identified = np.zeros(scenario_length)
        total = np.zeros(scenario_length)
        for tag in self.tags:
            offset = tag.first_round_index % scenario_length
            total[offset] += 1
            if tag.num_identified > 0:
                identified[offset] += 1
        return np.asarray(
            [x / y if y > 0 else 1.0 for x, y in zip(identified, total)])

    def num_active(self, scenario_length: int) -> ScenarioStats:
        """
        Get statistics of number of active tags per round depending on
        the round offset from the extended scenario start.
        """
        num_active = [list() for _ in range(scenario_length)]
        for round_ in self.rounds:
            num_active[round_.index % scenario_length].append(
                round_.num_tags_active)
        return ScenarioStats.build(num_active)

    def num_rounds_active(self, scenario_length: int) -> ScenarioStats:
        """
        Get the number of rounds the tag took part in depending on the offset
        the tag arrived at.
        """
        num_rounds = [list() for _ in range(scenario_length)]
        for tag in self.tags:
            offset = tag.first_round_index % scenario_length
            num_rounds[offset].append(tag.num_rounds_active)
        return ScenarioStats.build(num_rounds)


@dataclass
class Tag:
    pos: Optional[float] = None
    flag: InventoryFlag = InventoryFlag.A
    record: TagJournalRecord = None


@dataclass
class Round:
    spec: RoundSpec
    index: int
    record: RoundJournalRecord = None


def simulate(params: ModelParams, verbose: bool = False) -> Journal:
    time = 0.0  # model time
    max_tags = len(params.arrivals)

    journal = Journal()

    tags: List[Tag] = []
    tag_offset: int = 0
    round_index = 0
    arrival_offset = 0
    round_duration = 0.0

    # MAX_ITER = 10
    num_iter = 0

    while arrival_offset < max_tags or tag_offset < len(tags):
        num_iter += 1
        if verbose:
            if num_iter > 0 and num_iter % 1000 == 0:
                print(f"* {num_iter} iterations passed, time = {time}, "
                      f"generated {arrival_offset}/{len(params.arrivals)} "
                      f"tags, "
                      f"positions: {[tag.pos for tag in tags[tag_offset:]]}")

        _move_tags(tags, tag_offset, round_duration, params.speed)

        # Create and add new tags to the area:
        new_tags = _create_new_tags(
            params.arrivals[arrival_offset:],
            time=time,
            speed=params.speed,
            round_index=round_index)

        for tag in new_tags:
            tags.append(tag)
            journal.tags.append(tag.record)
            arrival_offset += len(new_tags)

        # Depart old tags
        # -- we need to depart here, since in rare cases new tags may become
        #    too old just after creation.
        tag_offset = _depart_tags(tags, tag_offset, params.length, time)

        round_ = Round(
            spec=params.scenario[round_index % len(params.scenario)],
            index=round_index,
            record=RoundJournalRecord(index=round_index)
        )
        journal.rounds.append(round_.record)

        # print("starting round #", round_index, round_.spec)
        round_duration = _sim_round(params.protocol, params.ber, round_,
                                    tags[tag_offset:])
        time += round_duration
        round_index += 1

    return journal


def _move_tags(tags: List[Tag], tag_offset: int, dt: float,
               speed: float) -> None:
    """Update tags positions."""
    dx = dt * speed
    for tag in tags[tag_offset:]:
        tag.pos += dx


def _depart_tags(tags: List[Tag], tag_offset: int, length: float,
                 time: float) -> int:
    """
    Filter out tags with positions greater than length. For these
    tags also set departed_at to time value and "forget" position.

    Returns
    -------
    tag_offset : int
        offset of the first tag in the area in tags list
    """
    num_tags = len(tags)
    while tag_offset < num_tags and tags[tag_offset].pos >= length:
        tag = tags[tag_offset]
        tag.pos = None
        tag.departed_at = time
        tag_offset += 1
    return tag_offset


def _create_new_tags(arrivals: Sequence[float], time: float, speed: float,
                     round_index: int) -> Tuple[Tag]:
    """
    Create new tags arrived up to `time`.

    Returns
    -------
    tuple of Tag(s)
    """
    max_arrivals = len(arrivals)
    offset = 0
    tags = []
    while offset < max_arrivals and time >= arrivals[offset]:
        arrived_at = arrivals[offset]
        record = TagJournalRecord(arrived_at=arrived_at,
                                  first_round_index=round_index)
        tags.append(Tag(speed * (time - arrived_at), record=record))
        offset += 1
    return tuple(tags)


def _arrive_tags(tags: List[Tag], journal: Journal, time: float, speed: float,
                 round_index: int, arrivals: Sequence[float],
                 arrival_offset: int) -> int:
    """
    Add tags arrived to this moment (time).

    Returns
    -------
    num_tags : int
        number of tags created
    """
    num_arrivals = len(arrivals)
    num_new_tags = 0
    while arrival_offset < num_arrivals and time >= arrivals[arrival_offset]:
        arrived_at = arrivals[arrival_offset]
        record = TagJournalRecord(
            arrived_at=arrived_at,
            first_round_index=round_index)
        tags.append(Tag((time - arrived_at) * speed, record=record))
        journal.tags.append(record)
        arrival_offset += 1
        num_new_tags += 1
    return num_new_tags


def _sim_round(protocol: Protocol, ber: float, round_: Round,
               tags: Sequence[Tag]) -> float:
    """
    Simulate round and return its duration.
    """
    link = protocol.props
    rt_link, tr_link = protocol.rt_link, protocol.tr_link
    active_tags = [tag for tag in tags if tag.flag == round_.spec.flag]

    # Extract values often used:
    t1, t2 = protocol.timings.t1, protocol.timings.t2

    # Record number of tags in area and number of active tags
    # to tag and round journals:
    for tag in tags:
        tag.record.num_rounds_total += 1
    for tag in active_tags:
        tag.record.num_rounds_active += 1
    round_.num_tags_total = len(tags)
    round_.num_tags_active = len(active_tags)

    # Select random slot for each tag:
    tags_slots = np.random.randint(0, link.n_slots, len(active_tags))

    # Compute number of tags in each slot:
    num_tags_per_slot = np.zeros(link.n_slots)
    for slot in tags_slots:
        num_tags_per_slot[slot] += 1

    n_empty = (num_tags_per_slot == 0).sum()
    n_collided = (num_tags_per_slot > 1).sum()
    round_.num_collisions = n_collided

    # Compute round duration including all empty and collided slots,
    # Query and QueryRep commands. These durations don't depend on
    # success of particular replies transmissions.
    duration = (
            rt_link.query.duration +
            (link.n_slots - 1) * rt_link.query_rep.duration +
            n_empty * protocol.timings.t4 +
            n_collided * (t1 + t2 + tr_link.rn16.duration))

    # Now we model reply slots
    for tag, slot in zip(active_tags, tags_slots):
        # Skip, if there is a collision (duration is already computed above):
        if num_tags_per_slot[slot] > 1:
            continue

        # Attempt to transmit RN16:
        duration += t1 + tr_link.rn16.duration + t2
        if np.random.uniform() > get_rx_prob(tr_link.rn16, ber):
            # print("> failed to receive RN16")
            continue  # Error in RN16 reception

        # Reader transmits Ack, tag attempts to transmit EPCID.
        # Since tag transmitted EPCID, and we model no NACKs, invert tag flag.
        duration += rt_link.ack.duration + t1 + tr_link.epc.duration + t2
        tag.flag = tag.flag.invert()
        if np.random.uniform() > get_rx_prob(tr_link.epc, ber):
            # print("> failed to receive EPC")
            continue  # Error in EPCID reception

        if not link.use_tid:
            # print("> Tag identified!")
            tag.record.num_identified += 1
            round_.record.num_tags_identified += 1
            continue  # Tag was identified, nothing more needed

        # Reader transmits Req_Rn, tag attempts to transmit Handle:
        duration += rt_link.req_rn.duration + t1 + tr_link.handle.duration + t2
        if np.random.uniform() > get_rx_prob(tr_link.handle, ber):
            # print("> failed to receive HANDLE")
            continue  # Error in Handle reception

        # Reader transmits Read, tag attempts to transmit Data:
        duration += rt_link.read.duration + t1 + tr_link.data.duration + t2
        if np.random.uniform() <= get_rx_prob(tr_link.data, ber):
            # print("> received DATA!")
            tag.record.num_identified += 1
            round_.record.num_tags_identified += 1
            continue
        # print("> failed to receive DATA")

    # If reader turns off after round, reset all tags flags and add
    # turn off period to round duration.
    if round_.spec.turn_off:
        for tag in tags:
            tag.flag = InventoryFlag.A
        duration += link.t_off

    # Write round duration and return it:
    round_.duration = duration
    return duration
