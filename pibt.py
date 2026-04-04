import random
from typing import Optional
import numpy as np
from utils import *
from dist_table import compute_dist_table, is_valid, neighbors
from print import print_plan
from mapf_types import Action, Agent, Coord, Grid, Plan

SWAP_ENABLED = True

def d_get(dist_tables: list[dict[Coord, int]], agent_idx: int, v: Coord) -> int:
    return dist_tables[agent_idx].get(v, 10**9)


def is_agent_at_goal_and_goal_is_dead_end(
    grid: Grid,
    v: Coord,
    occupied_now: dict[Coord, int],
    agents: list[Agent],
) -> bool:
    idx = occupied_now.get(v)
    return idx is not None and len(neighbors(grid, v)) == 1 and agents[idx].target == v


def is_swap_required(
    grid: Grid,
    dist_tables: list[dict[Coord, int]],
    agents: list[Agent],
    occupied_now: dict[Coord, int],
    pusher: int,
    puller: int,
    agent_current_coord: Coord,
    desired_coord: Coord,
) -> bool:
    v_pusher = agent_current_coord
    v_puller = desired_coord

    while d_get(dist_tables, pusher, v_puller) < d_get(dist_tables, pusher, v_pusher):
        candidates = [
            u
            for u in neighbors(grid, v_puller)
            if u != v_pusher and not is_agent_at_goal_and_goal_is_dead_end(grid, u, occupied_now, agents)
        ]
        if len(candidates) >= 2:
            return False
        if len(candidates) == 0:
            break
        v_pusher, v_puller = v_puller, candidates[0]

    return d_get(dist_tables, puller, v_pusher) < d_get(dist_tables, puller, v_puller) and (
        d_get(dist_tables, pusher, v_pusher) == 0
        or d_get(dist_tables, pusher, v_puller) < d_get(dist_tables, pusher, v_pusher)
    )


def is_swap_possible(
    grid: Grid,
    agents: list[Agent],
    occupied_now: dict[Coord, int],
    v_pusher_origin: Coord,
    v_puller_origin: Coord,
) -> bool:
    v_pusher = v_pusher_origin
    v_puller = v_puller_origin

    while v_puller != v_pusher_origin:
        candidates = [
            u
            for u in neighbors(grid, v_puller)
            if u != v_pusher and not is_agent_at_goal_and_goal_is_dead_end(grid, u, occupied_now, agents)
        ]
        if len(candidates) >= 2:
            return True
        if len(candidates) == 0:
            return False
        v_pusher, v_puller = v_puller, candidates[0]
    return False


def is_swap_required_and_possible(
    grid: Grid,
    dist_tables: list[dict[Coord, int]],
    agents: list[Agent],
    q_from: list[Coord],
    q_to: list[Optional[Coord]],
    occupied_now: dict[Coord, int],
    i: int,
    candidate_coords: list[Coord],
) -> Optional[int]:
    if not SWAP_ENABLED:
        return None
    if not candidate_coords:
        return None

    desired = candidate_coords[0]
    j = occupied_now.get(desired)
    if (
        j is not None
        and j != i
        and q_to[j] is None
        and is_swap_required(grid, dist_tables, agents, occupied_now, i, j, q_from[i], q_from[j])
        and is_swap_possible(grid, agents, occupied_now, q_from[j], q_from[i])
    ):
        return j

    if desired != q_from[i]:
        for u in neighbors(grid, q_from[i]):
            k = occupied_now.get(u)
            if (
                k is not None
                and desired != q_from[k]
                and is_swap_required(
                    grid,
                    dist_tables,
                    agents,
                    occupied_now,
                    k,
                    i,
                    q_from[i],
                    desired,
                )
                and is_swap_possible(grid, agents, occupied_now, desired, q_from[i])
            ):
                return k
    return None


def func_pibt(
    grid: Grid,
    dist_tables: list[dict[Coord, int]],
    agents: list[Agent],
    q_from: list[Coord],
    q_to: list[Optional[Coord]],
    occupied_now: dict[Coord, int],
    occupied_next: dict[Coord, int],
    i: int,
    rng: random.Random,
) -> bool:
    candidate_coords = neighbors(grid, q_from[i]) + [q_from[i]]
    tie_breakers = {v: rng.random() for v in candidate_coords}
    candidate_coords.sort(key=lambda v: d_get(dist_tables, i, v) + tie_breakers[v])

    swap_agent = is_swap_required_and_possible(
        grid, dist_tables, agents, q_from, q_to, occupied_now, i, candidate_coords
    )
    if swap_agent is not None:
        candidate_coords.reverse()

    for coord_idx, coord in enumerate(candidate_coords):
        if coord in occupied_next:
            continue

        j = occupied_now.get(coord)
        if j is not None and q_to[j] == q_from[i]:
            continue

        occupied_next[coord] = i
        q_to[i] = coord

        if (
            j is not None
            and coord != q_from[i]
            and q_to[j] is None
            and not func_pibt(
                grid,
                dist_tables,
                agents,
                q_from,
                q_to,
                occupied_now,
                occupied_next,
                j,
                rng,
            )
        ):
            if occupied_next.get(coord) == i:
                del occupied_next[coord]
            q_to[i] = None
            continue

        if (
            coord_idx == 0
            and swap_agent is not None
            and q_to[swap_agent] is None
            and q_from[i] not in occupied_next
        ):
            occupied_next[q_from[i]] = swap_agent
            q_to[swap_agent] = q_from[i]

        return True

    occupied_next[q_from[i]] = i
    q_to[i] = q_from[i]
    return False


def set_new_config(
    grid: Grid,
    dist_tables: list[dict[Coord, int]],
    agents: list[Agent],
    q_from: list[Coord],
    order: list[int],
    rng: random.Random,
) -> Optional[list[Coord]]:
    n_agents = len(agents)
    q_to: list[Optional[Coord]] = [None] * n_agents
    occupied_now: dict[Coord, int] = {q_from[i]: i for i in range(n_agents)}
    occupied_next: dict[Coord, int] = {}

    for i in order:
        if q_to[i] is None and not func_pibt(
            grid,
            dist_tables,
            agents,
            q_from,
            q_to,
            occupied_now,
            occupied_next,
            i,
            rng,
        ):
            return None

    decided = [pos for pos in q_to if pos is not None]
    if len(decided) != n_agents:
        return None
    return decided  # type: ignore


def solve(grid: Grid, agents: list[Agent]) -> Optional[Plan]:
    n_agents = len(agents)
    rng = random.Random(0)
    print('Computing dist table')
    dist_tables = [compute_dist_table(grid, agent.target) for agent in agents]
    print('Done computing dist table')

    plan: Plan = {agent: [] for agent in agents}
    current = [agent.start for agent in agents]

    rows, cols = grid.shape
    max_steps = min(rows * cols * 8, 80000)

    for step_itr in range(max_steps):
        if all(current[i] == agents[i].target for i in range(n_agents)):
            return plan

        order = sorted(
            range(n_agents),
            key=lambda i: (-d_get(dist_tables, i, current[i]), agents[i].id),
        )

        if step_itr > 182 and step_itr < 200:
            print(f'--- {step_itr} ---')
            bad_agents = [i for i in range(n_agents) if agents[i].target != current[i]]
            for agent_i in bad_agents:
                priority = order.index(agent_i)
                print(f'agent {agent_i} current {current[agent_i]} target {agents[agent_i].target} order: {priority}')

        nxt = set_new_config(grid, dist_tables, agents, current, order, rng)
        if nxt is None:
            return None

        for i, agent in enumerate(agents):
            dr = nxt[i][0] - current[i][0]
            dc = nxt[i][1] - current[i][1]
            plan[agent].append((dr, dc))
        current = nxt

    return None


if __name__ == "__main__":
    # grid: Grid = get_grid('./assets/empty-8-8.map')
    # scene = get_scenario('./assets/empty-8-8-random-1.scen', 32)
    grid: Grid = get_grid('./assets/ht_chantry.map')
    scene = get_scenario('./assets/ht_chantry.scen', 100)


    # grid: Grid = get_grid('./assets/two_rooms_narrow_door.map')
    # scene = get_scenario('./assets/two_rooms_narrow_door.scen')
    agents = to_agents(scene)
    SWAP_ENABLED = False


    result = solve(grid, agents)
    if not result:
        print('no solution')
        exit(1)
    else:
        plan: Plan = result
        # print_plan(grid, agents, plan or {})
        print('saved plan')
        save_configs_for_visualizer(to_configs(grid, plan), './output/out.txt')
