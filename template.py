import random
from typing import Optional
import numpy as np
from utils import *
from dist_table import compute_dist_table, is_valid, neighbors
from print import print_plan
from mapf_types import Action, Agent, Coord, Grid, Plan


def d_get(dist_tables: list[dict[Coord, int]], agent_idx: int, v: Coord) -> int:
    return dist_tables[agent_idx].get(v, 10**9)


def solve(grid: Grid, agents: list[Agent]) -> Optional[Plan]:
    return None


if __name__ == "__main__":
    grid: Grid = get_grid('./assets/empty-4-4.map')
    scene = get_scenario('./assets/empty-4-4.scen')
    agents = to_agents(scene)

    result = solve(grid, agents)
    if not result:
        print('no solution')
        exit(1)
    else:
        plan: Plan = result
        print('saved plan')
        save_configs_for_visualizer(to_configs(grid, plan), './output/out.txt')
