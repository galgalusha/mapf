"""Multi-Agent Path Finding (MAPF) utility functions.

This module provides utilities for loading MAPF problem instances from
standard benchmark files and validating MAPF solutions.
"""
import random
import os
import re
from typing import TypeAlias

import numpy as np
from mapf_types import *


def get_grid(map_file: str) -> Grid:
    """Load grid map from a MAPF benchmark file.

    Parses a .map file from the MAPF benchmarks (movingai.com format) and
    returns a 2D boolean array representing the grid.

    Args:
        map_file: Path to the .map file.

    Returns:
        2D boolean array where True indicates free space and False indicates
        an obstacle. Shape is (height, width) with indexing grid[y, x].

    Raises:
        AssertionError: If the map file format is invalid.
    """
    width, height = 0, 0
    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break

        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid


def get_scenario(scen_file: str, N: int | None = None) -> tuple[Config, Config]:
    """Load start and goal configurations from a MAPF scenario file.

    Parses a .scen file from the MAPF benchmarks (movingai.com format) and
    extracts start and goal positions for agents.

    Args:
        scen_file: Path to the .scen file.
        N: Maximum number of agents to load. If None, loads all agents.

    Returns:
        A tuple (starts, goals) where each is a list of (y, x) coordinates.
    """
    with open(scen_file, "r") as f:
        starts, goals = [], []
        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s, y_s, x_g, y_g = [int(res.group(k)) for k in range(1, 5)]
                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break

    return starts, goals


def save_scene(
    grid: Grid,
    scenario: tuple[Config, Config],
    scen_file: str,
    map_file: str = 'bla.map',
) -> None:
    """Saves a scenario to a .scen file in the movingai.com format.

    Args:
        grid: The grid map, used to get dimensions.
        scenario: A tuple (starts, goals) where each is a list of (y, x) coordinates.
        scen_file: The path to the output .scen file.
        map_file: The path to the map file, used to get the map name.
    """
    starts, goals = scenario
    assert len(starts) == len(goals), "Number of starts and goals must be equal."

    height, width = grid.shape
    map_name = os.path.basename(map_file)

    dirname = os.path.dirname(scen_file)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)

    with open(scen_file, "w") as f:
        f.write("version 1\n")
        for start, goal in zip(starts, goals):
            start_y, start_x = start
            goal_y, goal_x = goal
            # Format: bucket, map, width, height, start_x, start_y, goal_x, goal_y, optimal_dist
            line = f"1\t{map_name}\t{width}\t{height}\t{start_x}\t{start_y}\t{goal_x}\t{goal_y}\t0.0\n"
            f.write(line)


def to_agents(starts_goals: tuple[Config, Config]) -> list[Agent]:
    """Converts start and goal configurations into a list of Agent objects.

    Args:
        starts_goals: A tuple containing two lists:
            - A list of start coordinates (y, x) for each agent.
            - A list of goal coordinates (y, x) for each agent.

    Returns:
        A list of Agent objects, where each agent has an ID (1-based index),
        a start position, and a target position.

    Raises:
        AssertionError: If the number of starts and goals do not match.
    """
    starts, goals = starts_goals
    assert len(starts) == len(goals), "Number of starts and goals must be equal."
    return [
        Agent(id=i + 1, start=start, target=goal)
        for i, (start, goal) in enumerate(zip(starts, goals))
    ]


def manhattan_distance(p1: Coord, p2: Coord) -> int:
    """Calculates the Manhattan distance between two coordinates."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def add_more_agents(
    grid: Grid,
    scenario: tuple[Config, Config],
    num_new_agents: int,
) -> tuple[Config, Config]:
    """Adds new agents to a scenario with random start and goal positions.

    This function adds a specified number of new agents to an existing scenario.
    For each new agent, it randomly selects an available start position and an
    available goal position from the free cells on the grid.

    A start or goal position is "available" if it is not already assigned to
    another agent in the same role (start/goal). It is permissible for one
    agent's start to be another's goal.

    Args:
        grid: The grid map.
        scenario: A tuple (starts, goals) for existing agents.
        num_new_agents: The number of new agents to add.

    Returns:
        A new tuple (starts, goals) with additional agents.
    """
    starts, goals = scenario

    # 1. Find all free cells
    free_cells = list(zip(*np.where(grid)))

    # 2. Identify available positions for new starts and goals
    available_starts = list(set(free_cells) - set(starts))
    available_goals = list(set(free_cells) - set(goals))

    random.shuffle(available_starts)
    random.shuffle(available_goals)

    # 3. Determine how many agents we can actually add
    num_to_add = min(num_new_agents, len(available_starts), len(available_goals))

    # 4. Take the first `num_to_add` from the shuffled lists
    newly_added_starts = available_starts[:num_to_add]
    newly_added_goals = available_goals[:num_to_add]

    # 5. Combine with existing agents
    return starts + newly_added_starts, goals + newly_added_goals


def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    """Check if a coordinate is valid and free on the grid.

    Args:
        grid: 2D boolean array representing the map.
        coord: Position (y, x) to check.

    Returns:
        True if coordinate is within bounds and not an obstacle, False otherwise.
    """
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    """Get valid neighboring coordinates (4-connected grid).

    Args:
        grid: 2D boolean array representing the map.
        coord: Center position (y, x).

    Returns:
        List of valid neighboring coordinates in 4 directions (left, right,
        up, down). Empty list if coord is invalid.
    """
    # coord: y, x
    neigh: list[Coord] = []

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord

    if x > 0 and grid[y, x - 1]:
        neigh.append((y, x - 1))

    if x < grid.shape[1] - 1 and grid[y, x + 1]:
        neigh.append((y, x + 1))

    if y > 0 and grid[y - 1, x]:
        neigh.append((y - 1, x))

    if y < grid.shape[0] - 1 and grid[y + 1, x]:
        neigh.append((y + 1, x))

    return neigh


def save_configs_for_visualizer(configs: Configs, filename: str) -> None:
    """Save solution configurations for visualization.

    Exports the solution in a format compatible with mapf-visualizer tool.

    Args:
        configs: List of configurations, where each configuration is a list
            of agent positions (y, x) at a timestep.
        filename: Output file path.

    Example:
        >>> configs = [[(0, 0), (1, 1)], [(0, 1), (1, 2)]]
        >>> save_configs_for_visualizer(configs, "output.txt")
    """
    dirname = os.path.dirname(filename)
    if len(dirname) > 0:
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)


def to_configs(grid: Grid, plan: Plan) -> Configs:
    """Converts a plan into a sequence of configurations.

    A plan is a mapping from an agent to a sequence of actions. A configuration
    is a list of coordinates for all agents at a single timestep. This function
    simulates the plan to generate the configuration for each timestep.

    Args:
        grid: The grid map. (Currently unused, but kept for API consistency
              in case validation against the grid is added in the future).
        plan: A dictionary mapping each agent to its list of actions.

    Returns:
        A list of configurations. The first configuration is the start
        positions. Each subsequent configuration is the result of applying
        one action from each agent's plan.
    """
    if not plan:
        return []

    # Sort agents by ID to have a consistent order in the output configs
    agents = sorted(plan.keys(), key=lambda a: a.id)
    if not agents:
        return []

    # Determine the total number of timesteps in the plan
    horizon = max((len(p) for p in plan.values()), default=0)

    # Initialize configurations with the starting positions
    configs: Configs = []
    initial_config = [agent.start for agent in agents]
    configs.append(initial_config)

    # Simulate the plan timestep by timestep
    for t in range(horizon):
        last_config = configs[-1]
        next_config: Config = []
        for i, agent in enumerate(agents):
            current_pos = last_config[i]
            agent_plan = plan[agent]
            if t < len(agent_plan):
                action = agent_plan[t]
                next_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
                next_config.append(next_pos)
            else:
                # If agent's plan is finished, it stays at its last position
                next_config.append(current_pos)
        configs.append(next_config)

    return configs


def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    """Validate a MAPF solution for correctness.

    Checks that the solution:
    - Starts at the specified start positions
    - Ends at the specified goal positions
    - Has valid transitions (agents move to adjacent cells or stay)
    - Has no vertex collisions (two agents at same position)
    - Has no edge collisions (two agents swap positions)

    Args:
        grid: 2D boolean array representing the map.
        starts: Initial positions of all agents.
        goals: Goal positions of all agents.
        solution: Sequence of configurations over time.

    Raises:
        AssertionError: If the solution violates any MAPF constraint.
    """
    # starts
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # goals
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"


def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    """Check if a MAPF solution is valid.

    Wrapper around validate_mapf_solution that returns a boolean instead
    of raising exceptions.

    Args:
        grid: 2D boolean array representing the map.
        starts: Initial positions of all agents.
        goals: Goal positions of all agents.
        solution: Sequence of configurations over time.

    Returns:
        True if solution is valid, False otherwise.

    Example:
        >>> grid = get_grid("map.map")
        >>> starts, goals = get_scenario("scenario.scen", N=10)
        >>> pibt = PIBT(grid, starts, goals)
        >>> solution = pibt.run()
        >>> is_valid = is_valid_mapf_solution(grid, starts, goals, solution)
    """
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False
