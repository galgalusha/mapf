import sys

content = """from collections import deque
import random
from typing import Optional
import numpy as np
from utils import *
from dist_table import compute_dist_table, is_valid, neighbors
from mapf_types import Action, Agent, Coord, Grid, Plan, Config, Configs

def path_dist(dist_tables, agent_idx, v):
    return dist_tables[agent_idx].get(v, 10**9)

def get_path(grid: Grid, start: Coord, goal: Coord, obs: set[Coord] = None) -> list[Coord]:
    if start == goal: return [start]
    obs = obs or set()
    q = deque([(start, [start])])
    visited = {start} | obs
    while q:
        curr, path = q.popleft()
        for nxt in neighbors(grid, curr):
            if nxt == goal: return path + [nxt]
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
    return []

def get_shortest_path(agent_idx: int, start: Coord, grid: Grid, agents: list[Agent], dist_tables, occupied_now) -> list[Coord]:
    p = [start]
    goal = agents[agent_idx].target
    while p[-1] != goal:
        v = p[-1]
        best_nxt = None
        best_dist = float('inf')
        best_occ = False
        
        for nxt in neighbors(grid, v):
            d = path_dist(dist_tables, agent_idx, nxt)
            occ = nxt in occupied_now
            
            if d < best_dist or (d == best_dist and not occ and best_occ):
                best_dist, best_occ, best_nxt = d, occ, nxt
        if best_nxt is None:
            break
        p.append(best_nxt)
    return p

def update_plan(agent_idx: int, next_node: Coord, current_config: Config, occupied_now: dict[Coord, int], history: Configs):
    prev_node = current_config[agent_idx]
    if prev_node in occupied_now and occupied_now[prev_node] == agent_idx:
        del occupied_now[prev_node]
    occupied_now[next_node] = agent_idx
    current_config[agent_idx] = next_node
    history.append(list(current_config))

def get_nearest_empty(v_start: Coord, obs: set[Coord], grid: Grid, occupied_now: dict[Coord, int], agent_idx: int, dist_tables) -> Optional[Coord]:
    q = deque([v_start])
    visited = set(obs)
    visited.add(v_start)
    while q:
        curr = q.popleft()
        if curr not in occupied_now:
            return curr
        
        nxts = [n for n in neighbors(grid, curr) if n not in visited]
        nxts.sort(key=lambda x: path_dist(dist_tables, agent_idx, x))
        for nxt in nxts:
            visited.add(nxt)
            q.append(nxt)
    return None

def push_toward_empty(v_current: Coord, obs: set[Coord], grid: Grid, occupied_now: dict[Coord, int], current_config: Config, history: Configs, agent_idx: int, dist_tables) -> bool:
    v_empty = get_nearest_empty(v_current, obs, grid, occupied_now, agent_idx, dist_tables)
    if not v_empty: return False
    path = get_path(grid, v_current, v_empty, obs)
    if not path: return False
    for i in range(len(path) - 1, 0, -1):
        mover_coord = path[i-1]
        target = path[i]
        if mover_coord in occupied_now:
            update_plan(occupied_now[mover_coord], target, current_config, occupied_now, history)
    return True

def push(agent_idx: int, grid: Grid, agents: list[Agent], dist_tables: list[dict[Coord, int]], U: set[Coord], current_config: Config, occupied_now: dict[Coord, int], history: Configs) -> bool:
    if current_config[agent_idx] == agents[agent_idx].target:
        return True
    
    p_star = get_shortest_path(agent_idx, current_config[agent_idx], grid, agents, dist_tables, occupied_now)
    p_star.pop(0)
    if not p_star:
        return True
        
    v = p_star[0]
    while current_config[agent_idx] != agents[agent_idx].target:
        while v not in occupied_now:
            update_plan(agent_idx, v, current_config, occupied_now, history)
            p_star.pop(0)
            if not p_star: return True
            v = p_star[0]
            
        obs = set(U)
        obs.add(current_config[agent_idx])
        if not push_toward_empty(v, obs, grid, occupied_now, current_config, history, agent_idx, dist_tables):
            return False
            
    return True

def execute_swap_maneuver(v: Coord, r_idx: int, s_idx: int, grid: Grid, current_config: Config, occupied_now: dict[Coord, int], history: Configs):
    v_neighbors = neighbors(grid, v)
    empties = [n for n in v_neighbors if n not in occupied_now][:2]
    if len(empties) < 2: return
    
    e1, e2 = empties
    last_s = current_config[s_idx]
    
    update_plan(r_idx, e1, current_config, occupied_now, history)
    update_plan(s_idx, v, current_config, occupied_now, history)
    update_plan(s_idx, e2, current_config, occupied_now, history)
    update_plan(r_idx, v, current_config, occupied_now, history)
    update_plan(r_idx, last_s, current_config, occupied_now, history)
    update_plan(s_idx, v, current_config, occupied_now, history)

def resolve(r_idx: int, s_idx: int, grid: Grid, agents: list[Agent], U: set[Coord], current_config: Config, occupied_now: dict[Coord, int], history: Configs, dist_tables: list[dict[Coord, int]], junctions: list[Coord]) -> bool:
    ideal_s = current_config[r_idx]

    while ideal_s in occupied_now:
        blocker_idx = occupied_now[ideal_s]
        p = get_shortest_path(blocker_idx, ideal_s, grid, agents, dist_tables, occupied_now)
        if len(p) < 2: return False
        
        target = p[1]
        if target in occupied_now:
            obs = set(U) | {current_config[s_idx], current_config[blocker_idx]}
            if not push_toward_empty(target, obs, grid, occupied_now, current_config, history, blocker_idx, dist_tables):
                if not swap_op(blocker_idx, grid, agents, junctions, U, current_config, occupied_now, history, dist_tables):
                    return False
            else:
                update_plan(blocker_idx, target, current_config, occupied_now, history)
        else:
            update_plan(blocker_idx, target, current_config, occupied_now, history)

    update_plan(s_idx, ideal_s, current_config, occupied_now, history)
    return True

def multiPush(r_idx: int, s_idx: int, path: list[Coord], grid: Grid, current_config: Config, occupied_now: dict[Coord, int], history: Configs, dist_tables) -> bool:
    if len(path) == 0: return False
    
    if current_config[s_idx] != path[1]:
        for i in range(1, len(path)):
            if path[i] in occupied_now:
                if not push_toward_empty(path[i], {current_config[s_idx]}, grid, occupied_now, current_config, history, r_idx, dist_tables):
                    return False
            update_plan(r_idx, path[i], current_config, occupied_now, history)
            update_plan(s_idx, path[i-1], current_config, occupied_now, history)
    else:
        for i in range(2, len(path)):
            v = path[i]
            if v in occupied_now:
                if not push_toward_empty(v, {current_config[r_idx]}, grid, occupied_now, current_config, history, s_idx, dist_tables):
                    return False
            update_plan(s_idx, path[i], current_config, occupied_now, history)
            update_plan(r_idx, path[i-1], current_config, occupied_now, history)
            
        if not push_toward_empty(path[-1], {current_config[r_idx]}, grid, occupied_now, current_config, history, r_idx, dist_tables):
            return False
        update_plan(r_idx, path[-1], current_config, occupied_now, history)
    return True

def clear(v: Coord, r_idx: int, s_idx: int, grid: Grid, current_config: Config, occupied_now: dict[Coord, int], history: Configs, dist_tables) -> bool:
    def get_unoccupied_nodes():
        return [n for n in neighbors(grid, v) if n not in occupied_now]
    
    unoccupied_nodes = get_unoccupied_nodes()
    if len(unoccupied_nodes) >= 2: return True
    
    for u in neighbors(grid, v):
        unoccupied_nodes = get_unoccupied_nodes()
        if u in unoccupied_nodes: continue
        obs = set(unoccupied_nodes) | {current_config[r_idx], current_config[s_idx]}
        if push_toward_empty(u, obs, grid, occupied_now, current_config, history, r_idx, dist_tables):
            if len(get_unoccupied_nodes()) >= 2: return True
            
    last_loc_s = current_config[s_idx]
    for u in neighbors(grid, v):
        unoccupied_nodes = get_unoccupied_nodes()
        if u in unoccupied_nodes: continue
        disturbing_agent = occupied_now.get(u)
        if disturbing_agent is None: continue
        
        for w in unoccupied_nodes:
            obs = set(get_unoccupied_nodes()) | {u, v, w}
            if push_toward_empty(last_loc_s, obs, grid, occupied_now, current_config, history, s_idx, dist_tables):
                update_plan(r_idx, last_loc_s, current_config, occupied_now, history)
                update_plan(disturbing_agent, v, current_config, occupied_now, history)
                update_plan(disturbing_agent, w, current_config, occupied_now, history)
                update_plan(r_idx, v, current_config, occupied_now, history)
                update_plan(s_idx, last_loc_s, current_config, occupied_now, history)
                
                obs2 = set(get_unoccupied_nodes()) | {v, last_loc_s}
                if push_toward_empty(w, obs2, grid, occupied_now, current_config, history, r_idx, dist_tables):
                    if len(get_unoccupied_nodes()) >= 2: return True
                    break
    return False

def swap_op(r_idx: int, grid: Grid, agents: list[Agent], junctions: list[Coord], U: set[Coord], current_config: Config, occupied_now: dict[Coord, int], history: Configs, dist_tables: list[dict[Coord, int]]) -> bool:
    p_star = get_shortest_path(r_idx, current_config[r_idx], grid, agents, dist_tables, occupied_now)
    if len(p_star) <= 1: return True
    
    s_idx = occupied_now.get(p_star[1])
    if s_idx is None: return True

    c_before = list(current_config)
    
    v_star = p_star[0]
    sorted_junctions = sorted(junctions, key=lambda j: abs(j[0]-v_star[0]) + abs(j[1]-v_star[1]))
    
    success = False
    tmp_history = []
    
    for v in sorted_junctions:
        # Check path to v
        p_to_v = get_path(grid, current_config[r_idx], v, set())
        if not p_to_v: continue
        
        snapshot = list(current_config)
        snapshot_occ = dict(occupied_now)
        local_history = [list(current_config)]
        
        if v == current_config[r_idx] or multiPush(r_idx, s_idx, p_to_v, grid, current_config, occupied_now, local_history, dist_tables):
            if clear(v, r_idx, s_idx, grid, current_config, occupied_now, local_history, dist_tables):
                success = True
                tmp_history = local_history[1:]
                break
                
        # rollback
        current_config = list(snapshot)
        occupied_now.clear()
        occupied_now.update(snapshot_occ)
        
    if not success: return False
    
    # Commit multiPush and clear
    history.extend(tmp_history)
    
    # execute swap at v
    execute_swap_maneuver(v, r_idx, s_idx, grid, current_config, occupied_now, history)
    
    # reverse temporary history
    reversed_tmp = []
    for step in reversed(tmp_history):
        rev_step = list(step)
        rev_step[r_idx], rev_step[s_idx] = rev_step[s_idx], rev_step[r_idx]
        reversed_tmp.append(rev_step)
        
    history.extend(reversed_tmp)
    current_config = list(history[-1])
    occupied_now.clear()
    for i, c in enumerate(current_config):
        occupied_now[c] = i
        
    if agents[s_idx].target in U:
        return resolve(r_idx, s_idx, grid, agents, U, current_config, occupied_now, history, dist_tables, junctions)
        
    return True

def compress_solution(grid: Grid, agents: list[Agent], history: Configs) -> Configs:
    if not history: return []
    num_agents = len(agents)
    makespan = len(history) - 1
    
    temp_orders = { (r,c): deque() for r in range(grid.shape[0]) for c in range(grid.shape[1]) }
    
    for t in range(makespan + 1):
        for i in range(num_agents):
            v = history[t][i]
            if not temp_orders[v] or history[t-1][i] != v:
                temp_orders[v].append(i)
                
    new_plan = [list(history[0])]
    internal_clocks = [0]*num_agents
    
    goal_config = [a.target for a in agents]
    
    while new_plan[-1] != goal_config:
        config = []
        for i in range(num_agents):
            t = internal_clocks[i]
            
            if t == makespan:
                config.append(new_plan[-1][i])
                continue
                
            v_current = history[t][i]
            while t < makespan and v_current == history[t+1][i]:
                t += 1
            internal_clocks[i] = t
            
            if t == makespan:
                config.append(new_plan[-1][i])
                continue
                
            v_next = history[t+1][i]
            if temp_orders[v_next] and temp_orders[v_next][0] == i:
                config.append(v_next)
                temp_orders[v_current].popleft()
                internal_clocks[i] = t + 1
            else:
                config.append(new_plan[-1][i])
        new_plan.append(config)
        
    return new_plan

def solve(grid: Grid, agents: list[Agent]) -> Optional[Plan]:
    num_agents = len(agents)
    dist_tables = [compute_dist_table(grid, a.target) for a in agents]
    
    junctions = [ (c, r) for c in range(grid.shape[0]) for r in range(grid.shape[1]) 
                 if grid[c, r] and len(neighbors(grid, (c, r))) >= 3 ]
    
    occupied_now: dict[Coord, int] = {a.start: i for i, a in enumerate(agents)}
    current_config: Config = [a.start for a in agents]
    history: Configs = [list(current_config)]
    U: set[Coord] = set()

    for i in range(num_agents):
        while current_config[i] != agents[i].target:
            if not push(i, grid, agents, dist_tables, U, current_config, occupied_now, history):
                if not swap_op(i, grid, agents, junctions, U, current_config, occupied_now, history, dist_tables):
                    return None
        U.add(current_config[i])

    # Compress solution like C++ Implementation
    history = compress_solution(grid, agents, history)

    plan: Plan = {agent: [] for agent in agents}
    agent_indices = {agent: i for i, agent in enumerate(agents)}

    for t in range(len(history) - 1):
        for agent in agents:
            idx = agent_indices[agent]
            pos_now = history[t][idx]
            pos_next = history[t+1][idx]
            action = (pos_next[0] - pos_now[0], pos_next[1] - pos_now[1])
            plan[agent].append(action)

    return plan

if __name__ == "__main__":
    grid: Grid = get_grid('./assets/empty-4-4-agents-2.scen')
    scene = get_scenario('./assets/empty-4-4-agents-2.scen')
    agents = to_agents(scene)

    result = solve(grid, agents)
    if not result:
        print('no solution')
        exit(1)
    else:
        plan: Plan = result
        print('saved plan')
        # save_configs_for_visualizer(to_configs(grid, plan), './output/out.txt')
"""

with open('push_and_swap.py', 'w') as f:
    f.write(content)
