import heapq
from mapf_types import *

actions: list[Action] = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


def find_astar_path(grid: Grid, start: Coord, goal: Coord, constraints: set[Conflict]) -> list[Action]:
    constraint_set = {(c.time, c.coord) for c in constraints}
    
    def heuristic(pos: Coord) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def is_valid(pos: Coord) -> bool:
        r, c = pos
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c]
    
    def get_neighbors(pos: Coord, time: int) -> list[tuple[Coord, Action]]:
        neighbors = []
        r, c = pos        
        for action in actions:
            dr, dc = action
            new_r, new_c = r + dr, c + dc
            new_pos = (new_r, new_c)
            if is_valid(new_pos) and (time + 1, new_pos) not in constraint_set:
                neighbors.append((new_pos, action))
        return neighbors
    
    initial_f = heuristic(start)
    
    # Skip start position if it's constrained at time 0
    if (0, start) in constraint_set:
        return []
    
    open_set: list[tuple[int, int, Coord, list[Action]]] = [
        (initial_f, 0, start, [])
    ]
    visited = set()  # (time, pos) pairs we've explored
    
    while open_set:
        f_score, time, pos, path = heapq.heappop(open_set)
        
        # Check if we reached the goal
        if pos == goal:
            return path
        
        state = (time, pos)
        if state in visited:
            continue
        visited.add(state)
        
        # Expand neighbors
        for next_pos, action in get_neighbors(pos, time):
            next_time = time + 1
            next_state = (next_time, next_pos)
            
            if next_state not in visited:
                g_score = next_time  # Cost is time steps
                h_score = heuristic(next_pos)
                f_score_next = g_score + h_score
                
                new_path = path + [action]
                heapq.heappush(
                    open_set,
                    (f_score_next, next_time, next_pos, new_path)
                )    
    return []  # No path found
