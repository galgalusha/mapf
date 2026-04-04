from collections import defaultdict
from copy import deepcopy
from itertools import count
from typing import NamedTuple, Optional, TypeAlias
import numpy as np
from numpy.typing import NDArray
import heapq
from astar import find_astar_path
from mapf_types import Conflict
from print import print_plan

Coord: TypeAlias = tuple[int, int]

class Agent(NamedTuple):
    id: int
    start: Coord
    target: Coord

Grid: TypeAlias = NDArray[np.bool_]

"""True is a free cell. False is a blocked cell"""
grid: Grid = np.array([
    [True, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
    ])

Action: TypeAlias = tuple[int, int]

action_names = ['up', 'down', 'left', 'right', 'pass']

agents = [ 
    Agent(id=1, start=(1, 0), target=(2, 3)),
    Agent(id=2, start=(0, 1), target=(3, 2)),
    ]



Plan: TypeAlias = dict[Agent, list[Action]]


def get_cost(plan: Plan):
    lists: list[list[Action]] = list(plan.values())
    return max(len(l) for l in lists) 


class Node(NamedTuple):
    cost: int
    constraints: dict[Agent, set[Conflict]]


tie_breakre = count()

def get_priotity(node: Node):
    return (node.cost, next(tie_breakre))


def create_plan(grid: Grid, agents: list[Agent], constraints: dict[Agent, set[Conflict]]) -> Plan:
    plan: Plan = {}
    for agent in agents:
        list_of_actions = find_astar_path(grid, agent.start, agent.target, constraints.get(agent, set([])))
        if not list_of_actions:
            return {}
        plan[agent] = list_of_actions
    return plan    


def find_conflict(agents: list[Agent], plan: Plan) -> Optional[tuple[Conflict, list[Agent]]]:
    horizon = max(len(action_list) for action_list in plan.values())
    positions: dict[Agent, Coord] = {agent: agent.start for agent in agents}
    for time in range(0, horizon):
        coords_to_agents: dict[Coord, list[Agent]] = defaultdict(list)
        for agent, coord in positions.items():
            coords_to_agents[coord].append(agent)
        for coord, c_agents in coords_to_agents.items():
            if len(c_agents) > 1:
                return (Conflict(time=time, coord=coord), c_agents)
        for agent in agents:
            if time >= len(plan[agent]): continue
            action = plan[agent][time]
            pos = positions[agent]
            positions[agent] = (pos[0] + action[0], pos[1] + action[1])
    return None


def solve(grid: Grid, agents: list[Agent]) -> Optional[Plan]:
    best_plan: Optional[Plan] = None
    best_cost = float('inf')
    node = Node(cost=0, constraints={})
    OPEN = [(get_priotity(node), node)]

    while len(OPEN) > 0:
        priotity, node = heapq.heappop(OPEN)
        
        plan = create_plan(grid, agents, constraints=node.constraints)
        if not plan:
            continue
        
        cost = get_cost(plan)

        if cost >= best_cost:
            continue

        conflict_tup: Optional[tuple[Conflict, list[Agent]]] = find_conflict(agents, plan)

        if not conflict_tup:
            best_plan = plan
            best_cost = cost
            continue

        conflict, conflicting_agents = conflict_tup

        for agent in conflicting_agents:
            node_constraints = deepcopy(node.constraints)
            agent_constraints = node_constraints.get(agent, set([]))
            agent_constraints.add(conflict)
            node_constraints[agent] = agent_constraints
            child = Node(cost=cost, constraints=node_constraints)
            heapq.heappush(OPEN, (get_priotity(child), child))

    return best_plan



plan = solve(grid, agents)

print_plan(grid, agents, plan or {})