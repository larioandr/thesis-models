import numpy as np
from functools import lru_cache as cache, cached_property
from collections import namedtuple, Iterable
from typing import Union

from rfidlib.protocol.responses import TagFrame

from rfidam.protocol import Protocol
from rfidam.baskets import BasketsOccupancyProblem


@cache
def get_rx_prob(frame: TagFrame, ber: float):
    """
    Compute probability of successful tag frame reception.
    """
    return np.power(1 - ber, frame.msg.bitlen)


SlotValues = namedtuple('SlotValues', ['empty', 'reply', 'collided'])


@cache
def create_slot_model(
        protocol: Protocol,
        ber: float,
        n_tags: int) -> 'SlotModel':
    """
    Factory function for creating slot model.

    This routine is cached, so calling it multiple time won't add much
    overhead.

    Parameters
    ----------
    protocol : Protocol
    ber : float
    n_tags : int

    Returns
    -------
    model : SlotModel
    """
    return SlotModel(protocol, ber=ber, n_tags=n_tags)


@cache
def create_round_model(
        protocol: Protocol,
        ber: float,
        n_tags: int) -> 'RoundModel':
    """
    Factory function for creating round model.

    This routine is cached, so calling it multiple time won't add much
    overhead.

    Parameters
    ----------
    protocol : Protocol
    ber : float
    n_tags : int

    Returns
    -------
    model : SlotModel
    """
    return RoundModel(protocol, ber=ber, n_tags=n_tags)


class SlotModel:
    """
    Slot model provides routines for estimation of slot durations and
    probabilities.
    """
    def __init__(self, protocol: Protocol, ber: float, n_tags: int):
        self._protocol = protocol
        self._ber = ber
        self._n_tags = n_tags

    @property
    def protocol(self):
        return self._protocol

    @property
    def ber(self):
        return self._ber

    @property
    def n_tags(self):
        return self._n_tags

    @cached_property
    def durations(self) -> SlotValues:
        """
        Get slot durations.

        Fields:
        - empty: empty slot duration
        - collided: float
        - reply: tuple (probs, durations) of np.ndarray(s)

        Returns
        -------
        empty: float
            empty slot duration
        collided: float
            collided slot duration
        reply: (probs, durations)
            tuple of two np.ndarray instances for probabilities and durations
        """
        t1 = self._protocol.timings.t1
        t2 = self._protocol.timings.t2
        rt_link = self._protocol.rt_link
        tr_link = self._protocol.tr_link

        # Empty and collided slots have constant durations:
        t_empty = self._protocol.timings.t4
        t_collided = t1 + t2 + tr_link.rn16.duration

        # For reply slot, we need to compute durations of each command
        # (with additional T1 + T2 intervals) and probabilities:
        p_rn16 = get_rx_prob(tr_link.rn16, self._ber)
        p_epc = get_rx_prob(tr_link.epc, self._ber)
        p_handle = get_rx_prob(tr_link.handle, self._ber)

        t_reply = 0.0
        t_reply = (t1 + t2 + tr_link.rn16.duration) + \
            p_rn16 * (t1 + t2 + rt_link.ack.duration + tr_link.epc.duration)
        if self._protocol.props.use_tid:
            t_reply += (p_rn16 * p_epc) * \
                (t1 + t2 + rt_link.req_rn.duration + tr_link.handle.duration)
            t_reply += (p_rn16 * p_epc * p_handle) * \
                (t1 + t2 + rt_link.read.duration + tr_link.data.duration)

        return SlotValues(empty=t_empty, collided=t_collided, reply=t_reply)

    @cached_property
    def probs(self) -> SlotValues:
        """
        Get slot probabilities.
        """
        n_slots = self.protocol.props.n_slots
        baskets_model = BasketsOccupancyProblem.create(n_slots, self.n_tags)
        p0 = baskets_model.avg_count.empty / n_slots
        p1 = baskets_model.avg_count.single / n_slots
        p2 = 1 - p0 - p1
        return SlotValues(empty=p0, reply=p1, collided=p2)


class RoundModel:
    def __init__(self, protocol: Protocol, n_tags: int, ber: float):
        self._protocol = protocol
        self._n_tags = n_tags
        self._ber = ber

    @property
    def protocol(self) -> Protocol:
        return self._protocol

    @property
    def n_tags(self) -> int:
        return self._n_tags

    @property
    def ber(self) -> float:
        return self._ber

    @cached_property
    def slots(self) -> SlotModel:
        return SlotModel(self._protocol, self._ber, self._n_tags)

    @cached_property
    def round_duration(self) -> float:
        """
        Estimate average round duration.
        """
        n_slots = self.protocol.props.n_slots
        rt_link = self.protocol.rt_link

        queries_duration = rt_link.query.duration + \
            (n_slots - 1) * rt_link.query_rep.duration

        return queries_duration + self._protocol.props.n_slots * (
            self.slots.probs.empty * self.slots.durations.empty +
            self.slots.probs.collided * self.slots.durations.collided +
            self.slots.probs.reply * self.slots.durations.reply)


def estimate_rounds_props(protocol: Protocol, n_tags: int, ber: float,
                          n_iters: Union[int, Iterable] = 5000):
    """
    Estimate slots and round durations using Monte-Carlo method.

    Parameters
    ----------
    protocol : Protocol
    n_tags : int
    ber : float
    n_iters : int or iterable

    Returns
    -------
    round_duration : float
        average round duration
    slots_durations : SlotValues
        average durations of empty, reply and collided slots
    slots_counts : SlotValues
        average number of empty, reply and collided slots in a round
    """
    durations = {
        'rounds': [],
        'slots': SlotValues([], [], [])
    }
    slots_count_list = []

    n_slots = protocol.props.n_slots
    rt_link = protocol.rt_link
    tr_link = protocol.tr_link
    t1, t2, t4 = protocol.timings.t1, protocol.timings.t2, protocol.timings.t4

    p_rn16 = get_rx_prob(tr_link.rn16, ber)
    p_epcid = get_rx_prob(tr_link.epc, ber)
    p_handle = get_rx_prob(tr_link.handle, ber)

    def get_reply_slot():
        """Get duration of a slot with single tag reply wo Query/QueryRep.
        """
        slot_duration = 0.0
        # Query/QueryRep is not included - account only RN16:
        slot_duration += t1 + tr_link.rn16.duration + t2
        if np.random.uniform() > p_rn16:
            return slot_duration
        # Reader sends ACK, tag responds with EPCID:
        slot_duration += rt_link.ack.duration + t1 + tr_link.epc.duration + t2
        if not protocol.props.use_tid or np.random.uniform() > p_epcid:
            return slot_duration
        # Reader sends Req_Rn, tag responds with Handle:
        slot_duration += rt_link.req_rn.duration + t1 + \
            tr_link.handle.duration + t2
        if np.random.uniform() > p_handle:
            return slot_duration
        slot_duration += rt_link.read.duration + t1 + \
            tr_link.data.duration + t2
        return slot_duration

    # Count 'n_iters' rounds durations:
    iters = n_iters if isinstance(n_iters, Iterable) else range(n_iters)
    for _ in iters:
        round_duration = 0.0
        slots = np.zeros(n_slots)

        # Count Query + (N-1) QueryRep durations:
        round_duration += rt_link.query.duration + \
            (n_slots - 1) * rt_link.query_rep.duration

        # Select random slots for each tag:
        for _ in range(n_tags):
            slots[np.random.randint(0, n_slots)] += 1

        # Count slots:
        slots_count = SlotValues(
            empty=sum(slots == 0),
            reply=sum(slots == 1),
            collided=sum(slots > 1))
        assert slots_count.empty + slots_count.reply + \
            slots_count.collided == n_slots
        slots_count_list.append(slots_count)

        # Compute durations of empty and collided slots:
        t_empty_slot = t4
        t_collided_slot = t1 + tr_link.rn16.duration + t2
        durations['slots'].collided.append(t_collided_slot)
        durations['slots'].empty.append(t_empty_slot)

        # Add empty and collided slots to round duration:
        round_duration += (
            t_empty_slot * slots_count.empty +
            t_collided_slot * slots_count.collided)

        # Since in reply slots error may appear at any response, estimate
        # these slots durations one-by-one:
        for _ in range(slots_count.reply):
            slot_duration = get_reply_slot()
            durations['slots'].reply.append(slot_duration)
            round_duration += slot_duration

        # Record round duration:
        durations['rounds'].append(round_duration)

    # Compute statistics:
    avg_round_duration = np.mean(durations['rounds'])
    slots_durations_stats = SlotValues(
        empty=(
            np.mean(durations['slots'].empty)
            if durations['slots'].empty else np.nan),
        collided=(
            np.mean(durations['slots'].collided)
            if durations['slots'].collided else np.nan),
        reply=(
            np.mean(durations['slots'].reply)
            if durations['slots'].reply else np.nan)
    )
    slots_counts_stats = SlotValues(
        empty=np.mean([slots.empty for slots in slots_count_list]),
        collided=np.mean([slots.collided for slots in slots_count_list]),
        reply=np.mean([slots.reply for slots in slots_count_list]))

    return avg_round_duration, slots_durations_stats, slots_counts_stats
