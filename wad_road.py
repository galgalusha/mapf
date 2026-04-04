import heapq
import sys
import tty
import termios
from typing import Optional
from dist_table import neighbors
from mapf_types import Coord, Grid
from wad_model import *


SHARING_BONUS = -0.5
CROSSING_PENALTY = 1.0
OPPOSITE_PENALTY = 100.0


def get_cells(grid: Grid) -> list[Cell]:
    rows, cols = grid.shape
    cells = []
    for r in range(0, rows, CELL_SIZE):
        for c in range(0, cols, CELL_SIZE):
            height = min(CELL_SIZE, rows - r)
            width = min(CELL_SIZE, cols - c)
            cells.append(
                Cell(
                    r_idx=r // CELL_SIZE,
                    c_idx=c // CELL_SIZE,
                    width=width,
                    height=height,
                    pos=(r, c),
                )
            )
    return cells


def find_cell_passable_point(grid: Grid, cell: Cell) -> Optional[Coord]:
    start_r, start_c = cell.r_idx * CELL_SIZE, cell.c_idx * CELL_SIZE
    center_r, center_c = start_r + cell.height // 2, start_c + cell.width // 2

    best_point: Optional[Coord] = None
    min_dist = float("inf")

    for r_offset in range(cell.height):
        for c_offset in range(cell.width):
            r, c = start_r + r_offset, start_c + c_offset
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c]:
                dist = abs(r - center_r) + abs(c - center_c)
                if dist < min_dist:
                    min_dist = dist
                    best_point = (r, c)
    return best_point


def _reconstruct_path(came_from: dict, current: Coord) -> list[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _coords_to_segments(path: list[Coord]) -> list[Segment]:
    if len(path) < 2:
        return []

    segments = []
    current_segment_coords = [path[0]]
    current_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])

    for i in range(1, len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        direction = (p2[0] - p1[0], p2[1] - p1[1])

        current_segment_coords.append(p1)  # Add current point to segment

        if direction != current_direction:
            segments.append(
                Segment(coords=list(current_segment_coords), direction=current_direction)
            )
            current_segment_coords = [p1]  # Start new segment from current point
            current_direction = direction

    current_segment_coords.append(path[-1])
    segments.append(
        Segment(coords=list(current_segment_coords), direction=current_direction)
    )

    return segments


def create_road(
    grid: Grid,
    road_system: RoadSystem,
    start_cell: Cell,
    end_cell: Cell,
    entry_point: Coord,
    exit_point: Coord,
) -> Optional[Road]:
    def heuristic(a: Coord, b: Coord) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set: list[tuple[float, Coord]] = [(heuristic(entry_point, exit_point), entry_point)]
    came_from: dict[Coord, Coord] = {}
    g_score: dict[Coord, float] = {entry_point: 0.0}

    while open_set:
        _, current_pos = heapq.heappop(open_set)

        if current_pos == exit_point:
            path = _reconstruct_path(came_from, current_pos)
            segments = _coords_to_segments(path)
            return Road(start_cell, end_cell, segments, entry_point, exit_point)

        for next_pos in neighbors(grid, current_pos):
            direction = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])

            cost = 1.0

            segments_at_next = road_system.segments_by_coord.get(next_pos, [])
            is_sharing = any(s.direction == direction for s in segments_at_next)
            is_opposite_at_next = any(
                s.direction == (-direction[0], -direction[1]) for s in segments_at_next
            )
            is_crossing = bool(segments_at_next) and not is_sharing

            if is_sharing:
                cost += SHARING_BONUS
            if is_crossing:
                cost += CROSSING_PENALTY

            prev_pos = came_from.get(current_pos)
            if prev_pos:
                prev_direction = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
                if direction == prev_direction:
                    segments_at_current = road_system.segments_by_coord.get(current_pos, [])
                    is_opposite_at_current = any(
                        s.direction == (-direction[0], -direction[1])
                        for s in segments_at_current
                    )
                    if is_opposite_at_next and is_opposite_at_current:
                        cost += OPPOSITE_PENALTY

            tentative_g_score = g_score.get(current_pos, float("inf")) + cost
            if tentative_g_score < g_score.get(next_pos, float("inf")):
                came_from[next_pos] = current_pos
                g_score[next_pos] = tentative_g_score
                f_score = tentative_g_score + heuristic(next_pos, exit_point)
                heapq.heappush(open_set, (f_score, next_pos))

    return None


def create_road_system(grid: Grid) -> RoadSystem:
    road_system = RoadSystem()
    cells = get_cells(grid)
    road_system.cells = cells
    cell_points = {cell: find_cell_passable_point(grid, cell) for cell in cells}

    valid_cells = [cell for cell in cells if cell_points[cell] is not None]

    for i in range(len(valid_cells)):
        for j in range(len(valid_cells)):
            if i == j:
                continue

            cell1, cell2 = valid_cells[i], valid_cells[j]
            point1, point2 = cell_points[cell1], cell_points[cell2]

            if road_system.find_road(cell1, cell2) is None:
                print(f"Creating road from {cell1} to {cell2}")
                road = create_road(grid, road_system, cell1, cell2, point1, point2)
                if road:
                    road_system.add_road(road)
                else:
                    print(f"Failed to create road from {cell1} to {cell2}")
    return road_system


def print_road_system(grid: Grid, road_system: RoadSystem):
    rows, cols = grid.shape
    
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch += sys.stdin.read(2)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    curr_r, curr_c = 0, 0
    start_cell = None
    goal_cell = None
    selected_road = None
    saved_roads = {}
    visible_digits = set()

    while True:
        print("\033[H\033[J", end="") # clear screen
        vis_grid = [[".." for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if not grid[r, c]:
                    vis_grid[r][c] = "##"

        roads_to_draw = []
        for d in visible_digits:
            if d in saved_roads:
                roads_to_draw.append(saved_roads[d])
        if selected_road:
            roads_to_draw.append(selected_road)

        for road in roads_to_draw:
            for segment in road.segments:
                direction = segment.direction
                char = DIRECTION_CHARS.get(direction, "?")
                for coord in segment.coords:
                    r, c = coord
                    if grid[r, c]:
                        vis_grid[r][c] = f"\033[32m{char}{char}\033[0m"

        def is_in_cell(r, c, cell_obj):
            if not cell_obj: return False
            return cell_obj.pos[0] <= r < cell_obj.pos[0] + cell_obj.height and \
                   cell_obj.pos[1] <= c < cell_obj.pos[1] + cell_obj.width

        for r in range(rows):
            for c in range(cols):
                bg_color = ""
                if r == curr_r and c == curr_c:
                    bg_color = "\033[41m" # Red
                elif is_in_cell(r, c, start_cell):
                    bg_color = "\033[43m" # Yellow
                elif is_in_cell(r, c, goal_cell):
                    bg_color = "\033[46m" # Cyan
                
                if bg_color:
                    if vis_grid[r][c].startswith("\033[32m"):
                        vis_grid[r][c] = vis_grid[r][c].replace("\033[32m", bg_color + "\033[32m")
                    else:
                        vis_grid[r][c] = bg_color + vis_grid[r][c] + "\033[0m"

        for r in range(rows):
            print("".join(vis_grid[r]))
            
        if selected_road is None and goal_cell is not None:
            print("\nNo road found between selected cells.")
        print("\nUse arrow keys to move. Space/Enter to select. Any other key resets selection. 'q' to quit.")
        
        key = getch()
        if key == 'q' or key == '\x03': # q or Ctrl+C
            break
        elif key == '\x1b[A' and curr_r > 0: curr_r -= 1 # Up
        elif key == '\x1b[B' and curr_r < rows - 1: curr_r += 1 # Down
        elif key == '\x1b[C' and curr_c < cols - 1: curr_c += 1 # Right
        elif key == '\x1b[D' and curr_c > 0: curr_c -= 1 # Left
        elif key in [' ', '\r', '\n']: # Select
            r_idx = curr_r // CELL_SIZE
            c_idx = curr_c // CELL_SIZE
            pos = (r_idx * CELL_SIZE, c_idx * CELL_SIZE)
            height = min(CELL_SIZE, rows - pos[0])
            width = min(CELL_SIZE, cols - pos[1])
            cell = Cell(r_idx=r_idx, c_idx=c_idx, width=width, height=height, pos=pos)
            if start_cell is None:
                start_cell = cell
            elif goal_cell is None:
                goal_cell = cell
                selected_road = road_system.find_road(start_cell, goal_cell)
        elif key.isdigit():
            if selected_road:
                saved_roads[key] = selected_road
                visible_digits.add(key)
                start_cell = None
                goal_cell = None
                selected_road = None
            else:
                if key in saved_roads:
                    if key in visible_digits:
                        visible_digits.remove(key)
                    else:
                        visible_digits.add(key)
                start_cell = None
                goal_cell = None
                selected_road = None
        else:
            # Any other key resets
            if start_cell is not None:
                start_cell = None
                goal_cell = None
                selected_road = None