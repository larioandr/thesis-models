from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeVar, Any, Iterator, NewType
from itertools import product
from functools import reduce


# --- Distributions
class Distribution:
    def __call__(self):
        raise NotImplementedError


class UniformDist(Distribution):
    def __init__(self, min: float, max: float):
        self.__min = min
        self.__max = max
    
    @property
    def min(self) -> float:
        return self.__min
    
    @property
    def max(self) -> float:
        return self.__max

    def __call__(self):
        pass # TODO


class ExpDist(Distribution):
    def __init__(self, mean: float):
        self.__mean = mean
    
    @property
    def mean(self) -> float:
        return self.__mean

    def __call__(self):
        pass # TODO


# ---- Model objects
TagPersistence = namedtuple('TagPersistence', ('S1', 'S2', 'S3'))


@dataclass
class Config:
    num_lanes: int = 2
    lane_width: float = 3.5
    speed: float = 40  # kmph
    vehicle_direction: tuple[float, float, float] = (1, 0, 0)
    vehicle_length: float = 4.0
    vehicle_plates: tuple[str, str] = ('front', 'back')
    plate_height: float = 0.5
    pos_update_interval: float = 1e-2
    vehicle_start_offset: tuple[float, float, float] = (-10, 0, 0)
    vehicle_lifetime: float = 2.0
    vehicle_interval: Distribution = field(
        default_factory=lambda: UniformDist(0.9, 1.1))
    tag_modulation_loss: float = -10.0
    tag_sensitivity: float = -18.0
    tag_antenna_angle: float = 0.0
    tag_antenna_radiation_pattern: str = "dipole"
    tag_antenna_gain: float = 2.0
    tag_antenna_polarization: float = 1.0
    reader_antenna_sides: tuple[str, str] = ('front', 'back')
    reader_antenna_angle: float = 45.0
    reader_antenna_offset: float = 1.0
    reader_antenna_radiation_pattern: str = "dipole"
    reader_antenna_gain: float = 8.0
    reader_antenna_polarization: float = 0.5
    reader_cable_loss: float = -1.0
    reader_tx_power: float = 31.5
    reader_circulator_noise: float = -80.0
    rounds_per_antenna: int = 1
    rounds_per_inventory_flag: int = 1
    session_strategy: int = "A"
    tari: str = "12.5us"
    m: int = 4
    data0Mul: float = 2.0
    rtcalMul: float = 2.0
    sl: str = "ALL"
    session: str = "S0"
    dr: str = "8"
    trext: bool = False
    q: int = 4
    frequency: float = 860e6
    power_on_interval: float = 2000e-3
    power_off_interval: float = 100e-3
    doppler: bool = True
    thermal_noise: float = -114.0
    permittivity: float = 15.0
    conductivity: float = 3e-2
    ber_model: str = "rayleigh"


#
# Variation
#
T = TypeVar('T')


# noinspection PyShadowingBuiltins
class RangeValue(Iterable[T]):
    def __init__(self, min: T, max: T, step: T):
        if min > max:
            raise ValueError(f"min ({min}) exceeds max ({max})")
        if step < 0:
            raise ValueError(f"negative step disallowed")
        self.__min = min
        self.__max = max
        self.__step = step
        # variables for iterator protocol:
        self.__next_value = None
        self.__iter_end = True

    @property
    def min(self) -> T:
        return self.__min
    
    @property
    def max(self) -> T:
        return self.__max
    
    @property
    def step(self) -> T:
        return self.__step

    def __iter__(self) -> Iterator[T]:
        self.__next_value = self.__min
        self.__iter_end = False
        return self

    def __next__(self) -> T:
        if self.__next_value is None:
            raise StopIteration
        yield self.__next_value
        if self.__iter_end:
            self.__next_value = None
        else:
            self.__next_value += self.__step
            if self.__next_value > self.__max:
                self.__next_value = self.__max
                self.__iter_end = True


ParamValue = NewType('ParamValue', dict[str, Any])


class Variate(Iterable[ParamValue]):
    def values(self) -> tuple[dict[str, Any]]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[_T_co]:
        pass


def join(*tuples) -> tuple[Any]:
    """Join (concat) tuples into a single tuple."""
    return tuple(reduce(lambda x, y: x + y, tuples))


class VarFn(Variate):
    def __init__(self, items: Iterable[Variate], fn: str):
        self.__fn = fn
        self.__items = items
    
    @property
    def fn(self):
        return self.__fn

    @property
    def items(self):
        return self.__items
    
    def values(self) -> tuple[dict[str, Any]]:
        apply = {
            'join': join,
            'product': product,
            'zip': zip
        }[self.__fn]
        return tuple(apply(*(i.values() for i in self.__items)))


class VarVal(Variate):
    def __init__(self, mapping: dict[str, ParamValue]):
        self.__mapping = mapping
    
    @property
    def mapping(self):
        return dict(self.__mapping)
    
    def values(self) -> tuple[dict[str, Any]]:
        keys = []
        values = []
        for key, value in self.__mapping.items():
            keys.append(key)
            values.append(value)
        combinations = product(values)
        return tuple(
            {key: sample[i] for i, key in enumerate(keys)}
            for sample in combinations
        )
