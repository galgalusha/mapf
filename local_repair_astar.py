from typing import Optional

from astar import find_astar_path
from mapf_types import Action, Agent, Conflict, Coord, Grid, Plan
from utils import *


def solve(grid: Grid, agents: list[Agent]) -> Optional[Plan]:
    """
    Solves the Multi-Agent Path Finding (MAPF) problem using Local Repair A* (LRA*).

    LRA* is a simple and often effective algorithm. Each agent follows a
    pre-computed path. When an agent is about to collide with another agent,
    it triggers a "local repair" by replanning a new path from its current
    location to its goal, treating other agents' positions as temporary
    obstacles.

    The algorithm proceeds in timesteps. In each timestep, agents are processed
    sequentially. Each agent attempts to take the next step on its path. If that
    step leads to a conflict with an agent that has already moved or is stationary,
    it replans. If replanning is not possible or still results in a conflict,
    the agent waits for the timestep.
    """
    n_agents = len(agents)
    # Use a stable, predictable order for agents based on their ID
    agent_list = sorted(agents, key=lambda a: a.id)

    # 1. Initial plan for each agent using A* in an empty world
    paths: dict[Agent, list[Action]] = {}
    for agent in agent_list:
        path = find_astar_path(grid, agent.start, agent.target, set())
        if not path:
            # No solution if any agent can't reach its goal initially
            return None
        paths[agent] = path

    final_plan: Plan = {agent: [] for agent in agent_list}
    current_positions: list[Coord] = [agent.start for agent in agent_list]

    # A generous timeout to prevent infinite loops in case of deadlocks
    max_steps = grid.size * n_agents

    for _ in range(max_steps):
        if all(current_positions[i] == agent_list[i].target for i in range(n_agents)):
            return final_plan

        next_positions_this_step: list[Optional[Coord]] = [None] * n_agents

        # Process agents sequentially for this timestep
        for i in range(n_agents):
            agent = agent_list[i]
            current_pos = current_positions[i]

            if current_pos == agent.target:
                next_positions_this_step[i] = current_pos  # Wait at goal
                continue

            # 1. Peek at the next step in the current path
            path = paths[agent]
            action = path[0] if path else (0, 0)
            target_cell = (current_pos[0] + action[0], current_pos[1] + action[1])

            # 2. Check for conflict and repair
            obstacles = {p for p in next_positions_this_step[:i] if p} | {p for p in current_positions[i + 1 :]}

            if target_cell in obstacles:
                # REPAIR: Re-run A* with current obstacles
                constraints = {Conflict(time=1, coord=pos) for pos in obstacles}
                new_path = find_astar_path(grid, current_pos, agent.target, constraints)
                paths[agent], path = new_path, new_path
                action = path[0] if path else (0, 0)
                target_cell = (current_pos[0] + action[0], current_pos[1] + action[1])

            if target_cell in obstacles:
                action, target_cell = (0, 0), current_pos

            # 3. Commit action and next position
            next_positions_this_step[i] = target_cell
            if paths[agent] and action == paths[agent][0]:
                paths[agent].pop(0)
            final_plan[agent].append(action)

        current_positions = next_positions_this_step  # type: ignore

    return None  # Timed out


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
