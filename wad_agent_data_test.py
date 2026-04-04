import unittest
import numpy as np
from mapf_types import Agent
from wad_model import (
    initialize_agents_data,
    GETTING_ON_POINT,
    DROPPING_OFF_POINT,
    ROAD,
    AGENT_STAGE,
    DIST_TABLE_TO_ROAD_ON,
    DIST_TABLE_TO_ROAD_OFF,
    AgentStage,
    Road,
    Cell,
    Segment,
    RoadSystem
)
from dist_table import compute_dist_table
import wad_model
import wad_road

class WadAgentDataTest(unittest.TestCase):
    def test_initialize_agents_data(self):
        grid = np.array([
            [True,  True,  True,  False, False, True,  True,  True],
            [True,  False, False, False, False, False, False, True],
            [True,  False, False, False, False, False, False, True],
            [True,  True,  True,  True,  True,  True,  True,  True]
        ])
        
        wad_model.CELL_SIZE = 4

        agents = [Agent(id=0, start=(0, 2), target=(0, 5))]

        road_system = RoadSystem()
        start_cell = Cell(r_idx=0, c_idx=0, width=4, height=4, pos=(0, 0))
        # Corrected the end_cell to be in a different location as per the road
        end_cell = Cell(r_idx=0, c_idx=1, width=4, height=4, pos=(0, 4))
        
        entry_point = (0, 2)
        exit_point = (0, 5)
        
        segments = [
            Segment(coords=[(0, 2), (0, 1), (0, 0)], direction=(0, -1)),
            Segment(coords=[(0, 0), (1, 0), (2, 0), (3, 0)], direction=(1, 0)),
            Segment(coords=[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)], direction=(0, 1)),
            Segment(coords=[(3, 7), (2, 7), (1, 7), (0, 7)], direction=(-1, 0)),
            Segment(coords=[(0, 7), (0, 6), (0, 5)], direction=(0, -1)),
        ]

        road = Road(
            start_cell=start_cell,
            end_cell=end_cell,
            segments=segments,
            entry_point=entry_point,
            exit_point=exit_point
        )
        road_system.add_road(road)

        # Mock get_cells to control the cells returned for the test
        original_get_cells = wad_road.get_cells
        wad_road.get_cells = lambda grid: [start_cell, end_cell]

        initialize_agents_data(grid, agents, road_system)
        
        # Restore original functions
        wad_road.get_cells = original_get_cells

        agent_data = agents[0].data

        self.assertEqual(agent_data[GETTING_ON_POINT], entry_point)
        self.assertEqual(agent_data[DROPPING_OFF_POINT], exit_point)
        self.assertEqual(agent_data[ROAD], road)
        self.assertEqual(agent_data[AGENT_STAGE], AgentStage.GETTING_ON)

        dist_table_on = compute_dist_table(grid, entry_point)
        self.assertEqual(agent_data[DIST_TABLE_TO_ROAD_ON], dist_table_on)

        dist_table_off = compute_dist_table(grid, exit_point)
        self.assertEqual(agent_data[DIST_TABLE_TO_ROAD_OFF], dist_table_off)

if __name__ == '__main__':
    unittest.main()
