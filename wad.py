"""
Wait and Drive (WAD) Solver for MAPF


## Agent priorities

We will have a function create_agent_priorities where priority is basically
sorting. We can sort them by distance from goals using decending order.
There is no need for a tie breaking because once ordering is set, each
priority is unique.


## Cells

This solver will split the grid into cells of height CELL_SIZE
and width CELL_SIZE.
For each cell, we will find a passable coordinate to be entry or exit
point to roads coming in and out of this cell.

## Roads and Segments

We will have directed roads between cells coming in and out of their
entry points.
Roads will be represented by a list of directed horizontal or vertical
segments. Each segment will have a direction which is one of: left,right,up,down.
We will create roads with a new function create_road that will use A* that 
will satisfy the follow constraints:

[1] It will maximize usage of existing road segments given the segments
    share the same direction.
[2] It will minimize crossing with other roads if its not for the sake
    of sharing segments.
[3] It should completely avoid creating a segment of length greater than 1
    that is included in another segment that has the opposite direction.
    
If a road that fulfills all the above conditions cannot be found, the function
that creates the road shall rise an error.
Entry and exit points will not be counter for the above constraints, so if,
for example, for two roads from opposite directions, one ends where the other begins,
it won't be considered as a violation. The entry/exit points don't have
to be part of the segments.


## Selecting cells for road creation

We will create roads from every cell to every other cell in both
directions (cell_1 to cell_2 and from cell_2 to cell_1).
The order for choosing the cells could be arbitrary.


## Agent stages

After roads are constructed, we can start moving the agents.
Every agent will be in one of 3 stages as follows:

[1] GETTING_ON: Getting the agent from start position to the road.
    Given the agents start point and goal point, we know the start
    cell and goal cell. These cells will tell us which road to take
    from cell_start to cell_goal.
    The stage is about moving the agent from its start position to the
    closest point on that road.

[2] DRIVING: Moving the agent along the road.
    If there is a collision with another agent and the other agent
    is in the GETTING_ON stage, then the other agent will wait a turn.
    If there is a collision with another agent and the other agent
    is in the DRIVING stage, that means that we are at a point where 2
    segments from different diretions intersects. The agent with the
    lower priotity will wait and the other will continue driving.

[3] DROPPING_OFF: Getting the agent from the road to the goal position.
    The agents dropping point will be the point on the road that is closest
    to its goal position. The DROPPING_OFF stage begins once the agents reach
    the dropping point until the agent gets to its goal.

Agent whose start cell and goal cell are the same, will start immediately 
in the DROPPING_OFF stage without going through the previous stages.


## Debugging

For ease of debugging, it will be valuable to store the agent stage
and plan information on the 'data' property of the Agent instance as follows:

Agent.data[AGENT_STAGE] will be an enum representing one of the 3 stages the agent is currently in.
Agent.data[GETTING_ON_POINT] - the position on the road calculated on the GETTING_ON stage.
Agent.data[DROPPING_OFF_POINT] - the position on the road calculated on the DROPPING_OFF stage.
Agent.data[ROAD] - reference to the road object that the agent needs to take.
Agent.data[PRIOTITY] - a unique number we set up upon priority initialization.

All this data can be constructed in a function called create_agent_plan.


## Time iterations

On the main loop of the solve function, we will iterate through time steps
from 1 to MAX_TIME, or until all agents are in their goal positions.

Agents in the GETTING_ON or GETTING_OFF stage will perform a PIBT step. Meaning,
An agent in the GETTING_ON stage will use PIBT to move from its current position
to the GETTING_ON_POINT.
An agent in the GETTING_OFF stage will use PIBT to move from its DROPPING_OFF_POINT
to the goal position.
Because each of these 2 stages are limited to a single cell, each PIBT will use this
cell as its grid. 
Also, the agents using PIBT are allowed to push each other, but they are not
allowed to push agents that are on the DRIVING stage.
It only means that when calculating the next move using PIBT, the current
positions of agents in the DRIVING stage are considered as blocked on the grid.

Agents in the DRIVING stage will advance as described in the DRIVING stage descriptionsection.
When a DRIVING agent collide with a non-driving agent, the non-driving agent
needs to be considered as pushed when it comes to its PIBT move (highest priority in PIBT
and avoid the next position of the DRIVING agent).

## KNOWN LIMITATIONS (will be considered in future refinement)

* It is assumed that within a cell, all passable points are connected.
* It is assumed there are no rooms with only one way in/out.

"""

from typing import Optional
import itertools
from dist_table import compute_dist_table
from wad_model import *
from utils import *
from mapf_types import Agent, Grid, Plan
from wad_road import create_road_system, print_road_system


def get_cell_for_coord(road_system: RoadSystem, coord: Coord) -> Cell:
    r, c = coord
    r_idx = r // CELL_SIZE
    c_idx = c // CELL_SIZE
    for cell in road_system.cells:
        if cell.r_idx == r_idx and cell.c_idx == c_idx:
            return cell
    raise LookupError(f'no cell found for coord {coord}')


def create_agent_priorities(agents: list[Agent]):
    sorted_agents = sorted(
        agents,
        key=lambda agent: manhattan_distance(agent.start, agent.target),
        reverse=True,
    )
    for i, agent in enumerate(sorted_agents):
        agent.data[PRIORITY] = i


def initialize_agents_data(
    grid: Grid,
    agents: list[Agent],
    road_system: RoadSystem,
    cropped_grids: dict[Cell, Grid],
):
    create_agent_priorities(agents)

    for agent in agents:
        agent.data[DIST_TABLE_TO_ROAD_ON] = {}
        start_cell = get_cell_for_coord(road_system, agent.start)
        goal_cell = get_cell_for_coord(road_system, agent.target)

        if start_cell == goal_cell:
            agent.data[AGENT_STAGE] = AgentStage.DROPPING_OFF
            agent.data[ROAD] = None
            agent.data[GETTING_ON_POINT] = None
            agent.data[DROPPING_OFF_POINT] = None

            # Dist table for start/goal cell
            local_goal = (agent.target[0] - start_cell.pos[0], agent.target[1] - start_cell.pos[1])
            dist_table = compute_dist_table(cropped_grids[start_cell], local_goal)
            agent.data[DIST_TABLE_TO_ROAD_ON][start_cell] = dist_table
        else:
            road = road_system.find_road(start_cell, goal_cell)
            if not road:
                raise Exception(f"No road found for agent {agent.id}")

            agent.data[ROAD] = road
            agent.data[AGENT_STAGE] = AgentStage.GETTING_ON

            all_road_coords = list(
                itertools.chain.from_iterable(s.coords for s in road.segments)
            )

            # Calculate getting_on_point
            local_agent_start = (agent.start[0] - start_cell.pos[0], agent.start[1] - start_cell.pos[1])
            start_dist_table = compute_dist_table(cropped_grids[start_cell], local_agent_start)
            
            road_coords_in_start_cell = [c for c in all_road_coords if get_cell_for_coord(road_system, c) == start_cell]

            getting_on_point = min(
                road_coords_in_start_cell,
                key=lambda c: start_dist_table.get((c[0] - start_cell.pos[0], c[1] - start_cell.pos[1]), float('inf')),
                default=None
            )
            
            if getting_on_point is None:
                raise Exception(f"Could not find getting_on_point for agent {agent.id}")

            # Calculate dropping_off_point
            local_agent_target = (agent.target[0] - goal_cell.pos[0], agent.target[1] - goal_cell.pos[1])
            goal_dist_table = compute_dist_table(cropped_grids[goal_cell], local_agent_target)

            road_coords_in_goal_cell = [c for c in all_road_coords if get_cell_for_coord(road_system, c) == goal_cell]
            
            dropping_off_point = min(
                road_coords_in_goal_cell,
                key=lambda c: goal_dist_table.get((c[0] - goal_cell.pos[0], c[1] - goal_cell.pos[1]), float('inf')),
                default=None
            )

            if dropping_off_point is None:
                raise Exception(f"Could not find dropping_off_point for agent {agent.id}")

            agent.data[GETTING_ON_POINT] = getting_on_point
            agent.data[DROPPING_OFF_POINT] = dropping_off_point

            # Getting on dist table
            gop_cell = get_cell_for_coord(road_system, getting_on_point)
            local_goal = (getting_on_point[0] - gop_cell.pos[0], getting_on_point[1] - gop_cell.pos[1])
            dist_table = compute_dist_table(cropped_grids[gop_cell], local_goal)
            agent.data[DIST_TABLE_TO_ROAD_ON] = dist_table

            # Dropping off dist table
            dop_cell = get_cell_for_coord(road_system, dropping_off_point)
            local_goal = (dropping_off_point[0] - dop_cell.pos[0], dropping_off_point[1] - dop_cell.pos[1])
            dist_table = compute_dist_table(cropped_grids[dop_cell], local_goal)
            agent.data[DIST_TABLE_TO_ROAD_OFF] = dist_table


def d_get(dist_tables: list[dict[Coord, int]], agent_idx: int, v: Coord) -> int:
    return dist_tables[agent_idx].get(v, 10**9)

def crop_grid_to_cells(grid, road_system) -> dict[Cell, Grid]:
    return ({
        cell: grid[
            cell.pos[0] : cell.pos[0] + cell.height,
            cell.pos[1] : cell.pos[1] + cell.width,
        ]
        for cell in road_system.cells
    })


def solve(grid: Grid, agents: list[Agent]) -> Optional[Plan]:
    print("Creating road system...")
    road_system = create_road_system(grid)
    print("Road system created.")
    # print_road_system(grid, road_system)

    cropped_grids = crop_grid_to_cells(grid, road_system)

    print("Initializing agents data...")
    initialize_agents_data(grid, agents, road_system, cropped_grids)
    print("Agents data initialized.")

    return None


if __name__ == "__main__":
    grid: Grid = get_grid("./assets/random-32-32-20.map")
    scene = get_scenario("./assets/random-32-32-20.scen", 20)
    agents = to_agents(scene)

    # road_system = create_road_system(grid)
    # print_road_system(grid, road_system)


    result = solve(grid, agents)
    if not result:
        print("no solution")
        exit(1)
    else:
        plan: Plan = result
        print("saved plan")
        save_configs_for_visualizer(to_configs(grid, plan), "./output/out.txt")
