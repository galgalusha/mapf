from collections import deque

from mapf_types import Coord, Grid


def is_valid(grid: Grid, pos: Coord) -> bool:
    r, c = pos
    rows, cols = grid.shape
    return 0 <= r < rows and 0 <= c < cols and bool(grid[r, c])


def neighbors(grid: Grid, pos: Coord) -> list[Coord]:
    r, c = pos
    nbrs: list[Coord] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nxt = (r + dr, c + dc)
        if is_valid(grid, nxt):
            nbrs.append(nxt)
    return nbrs


def compute_dist_table(grid: Grid, goal: Coord) -> dict[Coord, int]:
    dist: dict[Coord, int] = {goal: 0}
    q: deque[Coord] = deque([goal])
    while q:
        v = q.popleft()
        for u in neighbors(grid, v):
            if u in dist:
                continue
            dist[u] = dist[v] + 1
            q.append(u)
    return dist
