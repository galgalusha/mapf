import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lns2'))

from loadscen import *
from prioritizedPlanning import *
from Utils import *
from LNSUtil import *
from ReplanCBSSIPPS import LNS2CBS
from ReplanPPSIPPS import LNS2PP
from mapf_types import Coord, Config, Configs
from utils import save_configs_for_visualizer

# SCENE_FILE = './assets/empty-8-8-agents-60.scen'

SCENE_FILE = './assets/2_rooms_2_doors_5x5.scen'
OUT_FILE = './output/out_lns2.txt'
NUM_OF_AGENTS = 2
NUM_OF_NEIGHBORS = 1

instanceMap, instanceStarts, instanceGoals = loadScen(SCENE_FILE, NUM_OF_AGENTS)

map_width = len(instanceMap[0])
map_height = len(instanceMap)


paths, num_replan = LNS2PP(NUM_OF_NEIGHBORS, map_width, map_height, instanceMap, instanceStarts, instanceGoals)

max_len = max(len(p) for p in paths) if paths else 0
configs = []
for t in range(max_len):
    step_config = []
    for agent_path in paths:
        # If agent path ended, it stays at the last position
        pos = agent_path[t] if t < len(agent_path) else agent_path[-1]
        step_config.append((pos[1], pos[0]) if isinstance(pos, (tuple, list)) else pos)
    configs.append(step_config)

save_configs_for_visualizer(configs, OUT_FILE)
print(f"Configs saved to {OUT_FILE}")
    