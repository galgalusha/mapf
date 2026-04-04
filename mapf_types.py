from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]
Config: TypeAlias = list[Coord]
Configs: TypeAlias = list[Config]

Action: TypeAlias = tuple[int, int]

@dataclass(frozen=True)
class Agent:
    id: int
    start: Coord
    target: Coord    
    # This field is excluded from hash and equality checks
    data: dict = field(default_factory=dict, hash=False, compare=False)

Plan: TypeAlias = dict[Agent, list[Action]]

class Conflict(NamedTuple):
    """Avoiding being at given coord at a given time"""
    time: int
    coord: Coord
