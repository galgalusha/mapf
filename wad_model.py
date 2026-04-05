from typing import Dict, List, NamedTuple, Optional, TypeAlias
from mapf_types import *
from enum import Enum


## Agent Stages ##
class AgentStage(Enum):
    GETTING_ON = "GETTING_ON"
    DRIVING = "DRIVING"
    DROPPING_OFF = "DROPPING_OFF"


## Agent Data Keys ##
GETTING_ON_POINT = "GETTING_ON_POINT"
DROPPING_OFF_POINT = "DROPPING_OFF_POINT"
ROAD = "ROAD"
PRIORITY = "PRIORITY"
AGENT_STAGE = "AGENT_STAGE"
DIST_TABLE_TO_ROAD_ON = "DIST_TABLE_TO_ROAD_ON"
DIST_TABLE_TO_ROAD_OFF = "DIST_TABLE_TO_ROAD_OFF"


## Hyper Parameters ##
CELL_SIZE = 11
MAX_TIME = 80000


class Cell(NamedTuple):
    r_idx: int
    c_idx: int

    @property
    def width(self) -> int:
        return CELL_SIZE

    @property
    def height(self) -> int:
        return CELL_SIZE

    @property
    def pos(self) -> Coord:
        return (self.r_idx * CELL_SIZE, self.c_idx * CELL_SIZE)


class Segment(NamedTuple):
    coords: list[Coord]
    direction: tuple[int, int]


class Road(NamedTuple):
    start_cell: Cell
    end_cell: Cell
    segments: list[Segment]
    entry_point: Coord
    exit_point: Coord

class RoadSystem:
    def __init__(self):
        self.roads: dict[tuple[Cell, Cell], Road] = {}
        self.segments_by_coord: dict[Coord, list[Segment]] = {}
        self.cells: list[Cell] = []

    def add_road(self, road: "Road"):
        self.roads[(road.start_cell, road.end_cell)] = road
        for segment in road.segments:
            for coord in segment.coords:
                if coord not in self.segments_by_coord:
                    self.segments_by_coord[coord] = []
                self.segments_by_coord[coord].append(segment)

    def find_road(self, start_cell: Cell, end_cell: Cell) -> "Optional[Road]":
        return self.roads.get((start_cell, end_cell))

Direction: TypeAlias = tuple[int, int]

UP: Direction = (-1, 0)
DOWN: Direction = (1, 0)
LEFT: Direction = (0, -1)
RIGHT: Direction = (0, 1)

DIRECTIONS: list[Direction] = [UP, DOWN, LEFT, RIGHT]
DIRECTION_CHARS: dict[Direction, str] = {
    UP: "^",
    DOWN: "v",
    LEFT: "<",
    RIGHT: ">",
}


