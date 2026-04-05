import unittest
import numpy as np
from unittest.mock import patch
from wad_model import Agent, AgentStage, ROAD, AGENT_STAGE, GETTING_ON_POINT, DROPPING_OFF_POINT, DIST_TABLE_TO_ROAD_ON, DIST_TABLE_TO_ROAD_OFF
from wad_road import create_road_system
from wad import crop_grid_to_cells, initialize_agents_data


class WadAgentDataTest(unittest.TestCase):
    @patch('wad_model.CELL_SIZE', 4)
    @patch('wad.CELL_SIZE', 4)
    @patch('wad_road.CELL_SIZE', 4)
    def test_initialize_agents_data(self):
        """Tests initialize_agents_data assigns correct properties to the agent."""

        grid = self.parse_grid(
            "...##...\n"
            ".######.\n"
            ".######.\n"
            "........")

        agent = Agent(id=1, start=(0, 2), target=(0, 5))
        road_system = create_road_system(grid)
        cropped_grids = crop_grid_to_cells(grid, road_system)
        initialize_agents_data(grid, [agent], road_system, cropped_grids)

        self.assertEqual(agent.data[AGENT_STAGE], AgentStage.GETTING_ON)
        self.assertEqual(agent.data[GETTING_ON_POINT], (3, 2))
        self.assertEqual(agent.data[DROPPING_OFF_POINT], (2, 7))

        road = agent.data[ROAD]
        self.assertIsNotNone(road)
        self.assertEqual(road.entry_point, (3, 2))
        self.assertEqual(road.exit_point, (2, 7))

        # Verify dist tables. We verify checking exact expected distance paths
        dist_on = agent.data[DIST_TABLE_TO_ROAD_ON]
        self.assertEqual(dist_on.get((0, 2)), 7) # path: (0,2)->(0,1)->(0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2) -> distance 7

        dist_off = agent.data[DIST_TABLE_TO_ROAD_OFF]
        self.assertEqual(dist_off.get((0, 1)), 4) 


    def parse_grid(self, grid_str):
        lines = grid_str.split("\n")
        grid = np.array([[c == '.' for c in row] for row in lines], dtype=bool)
        return grid

if __name__ == '__main__':
    unittest.main()
