from typing import Dict, List, NamedTuple, Optional, TypeAlias

from mapf_types import Coord

## Hyper Parameters ##
CELL_SIZE = 11
MAX_TIME = 80000


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


class Segment(NamedTuple):
    coords: list[Coord]
    direction: Direction


class Cell(NamedTuple):
    r_idx: int
    c_idx: int


class Road(NamedTuple):
    start_cell: Cell
    end_cell: Cell
    segments: list[Segment]
    entry_point: Coord
    exit_point: Coord


class RoadSystem:
    def __init__(self):
        self.roads: Dict[tuple[Cell, Cell], Road] = {}
        self.segments_by_coord: Dict[Coord, List[Segment]] = {}

    def add_road(self, road: Road):
        self.roads[(road.start_cell, road.end_cell)] = road
        for segment in road.segments:
            for coord in segment.coords:
                if coord not in self.segments_by_coord:
                    self.segments_by_coord[coord] = []
                self.segments_by_coord[coord].append(segment)

    def find_road(self, start_cell: Cell, end_cell: Cell) -> Optional[Road]:
        return self.roads.get((start_cell, end_cell))