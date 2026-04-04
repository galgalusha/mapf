from mapf_types import Agent, Coord, Grid, Plan


def print_plan(grid: Grid, agents: list[Agent], plan: Plan):
    if not plan:
        print("No plan found")
        return

    horizon = max(len(action_list) for action_list in plan.values())
    positions: dict[Agent, Coord] = {agent: agent.start for agent in agents}

    for time in range(0, horizon + 1):
        if time > 0:
            print(f"--- Frame {time} ---")

        cell_contents = [[None for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not grid[i, j]:
                    cell_contents[i][j] = "##\n##"
                else:
                    cell_contents[i][j] = "  \n  "

        for agent in agents:
            goal_r, goal_c = agent.target
            if cell_contents[goal_r][goal_c] != "##\n##":
                cell_contents[goal_r][goal_c] = f"g{agent.id}\n  "

        for agent, coord in positions.items():
            r, c = coord
            cell_contents[r][c] = f"{agent.id} \n  "

        print("+" + "+".join(["-" * 2 for _ in range(grid.shape[1])]) + "+")

        for i in range(grid.shape[0]):
            for line in range(2):
                print("|", end="")
                for j in range(grid.shape[1]):
                    cell_lines = cell_contents[i][j].split("\n")
                    print(cell_lines[line] + "|", end="")
                print()

            print("+" + "+".join(["-" * 2 for _ in range(grid.shape[1])]) + "+")

        if time < horizon:
            for agent in agents:
                if time < len(plan[agent]):
                    action = plan[agent][time]
                    pos = positions[agent]
                    positions[agent] = (pos[0] + action[0], pos[1] + action[1])
