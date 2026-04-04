import unittest
import numpy as np
from unittest.mock import patch

from wad_model import Cell
from wad_road import find_cell_passable_point, get_cells


class WadRoadTest(unittest.TestCase):
    @patch('wad_road.CELL_SIZE', 3)
    def test_get_cells(self):
        """Tests the cell creation logic."""
        grid = np.ones((9, 9), dtype=bool)
        cells = get_cells(grid)
        self.assertEqual(len(cells), 9)
        # Check some expected cells
        self.assertIn(Cell(r_idx=0, c_idx=0), cells)
        self.assertIn(Cell(r_idx=1, c_idx=1), cells)
        self.assertIn(Cell(r_idx=2, c_idx=2), cells)
        # Check an out-of-bounds cell index
        self.assertNotIn(Cell(r_idx=3, c_idx=0), cells)

    @patch('wad_road.CELL_SIZE', 3)
    def test_find_cell_passable_point_clear(self):
        """Tests finding the center in a fully passable cell."""
        grid = np.ones((9, 9), dtype=bool)

        # Cell(0,0) covers grid area [0:3, 0:3]. Geometric center is (1,1).
        cell_0_0 = Cell(r_idx=0, c_idx=0)
        center_0_0 = find_cell_passable_point(grid, cell_0_0)
        self.assertEqual(center_0_0, (1, 1))

        # Cell(1,2) covers grid area [3:6, 6:9]. Geometric center is (4,7).
        cell_1_2 = Cell(r_idx=1, c_idx=2)
        center_1_2 = find_cell_passable_point(grid, cell_1_2)
        self.assertEqual(center_1_2, (4, 7))

    @patch('wad_road.CELL_SIZE', 3)
    def test_find_cell_passable_point_with_obstacle(self):
        """Tests finding the center when the geometric center is blocked."""
        grid = np.ones((9, 9), dtype=bool)

        # For Cell(0,0), block the geometric center (1,1)
        grid[1, 1] = False

        cell_0_0 = Cell(r_idx=0, c_idx=0)
        center_0_0 = find_cell_passable_point(grid, cell_0_0)
        self.assertEqual(center_0_0, (0, 1))

if __name__ == '__main__':
    unittest.main()