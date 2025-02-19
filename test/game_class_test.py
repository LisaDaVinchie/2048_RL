import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.game_class import Game2048_env

class TestGame2048Env(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        # Mocking the reward function and parameters path for testing
        self.env = Game2048_env()
    
    def test_reset(self):
        """Test if the reset method correctly initializes the grid."""
        grid = self.env.reset()
        self.assertEqual(grid.shape, (4, 4))  # Check if the grid size is correct
        self.assertTrue(np.any(grid == 2) or np.any(grid == 4))  # Check if there's at least one new tile (2 or 4)
        self.assertEqual(np.sum(grid == 0), 15)  # There should be exactly 15 zeros (one tile should be added)
    
    def test_step_move_up(self):
        """Test if the step method works correctly with an up move."""
        old_grid = np.array([[0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 4, 0], [0, 0, 0, 0]])
        expected_new_grid = np.array([[0, 0, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        new_grid = self.env._move(old_grid, 0)  # Move up
        
        self.assertTrue(np.array_equal(new_grid, expected_new_grid))  # Check if the grid is as expected
    
    def test_step_move_down(self):
        """Test if the step method works correctly with a down move."""
        old_grid = np.array([[0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 4, 0], [0, 0, 0, 0]])
        expected_new_grid = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 4]])
        new_grid = self.env._move(old_grid, 1)  # Move down
        self.assertTrue(np.array_equal(new_grid, expected_new_grid))

    def test_step_move_left(self):
        """Test if the step method works correctly with a left move."""
        old_grid = np.array([[0, 2, 0, 2], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0]])
        expected_new_grid = np.array([[4, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        new_grid = self.env._move(old_grid, 2)  # Move left
        self.assertTrue(np.array_equal(new_grid, expected_new_grid))  # Check if the grid is as expected
    
    def test_step_move_right(self):
        """Test if the step method works correctly with a right move."""
        old_grid = np.array([[0, 2, 0, 2],
                             [0, 0, 0, 4],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])
        expected_new_grid = np.array([[0, 0, 0, 4],
                                      [0, 0, 0, 4],
                                      [0, 0, 0, 0],
                                      [0, 0, 0, 0]])
        new_grid = self.env._move(old_grid, 3)  # Move right
        self.assertTrue(np.array_equal(new_grid, expected_new_grid))  # Check if the grid is as expected

    def test_step_game_over(self):
        """Test if the step method handles the game over condition."""
        # Set a grid where the game is over (no empty cells and no adjacent merges)
        old_grid = np.array([[2, 4, 8, 16],
                             [32, 64, 128, 256],
                             [512, 1024, 2048, 4096],
                             [8192, 16384, 32768, 65536]])
        _, game_over = self.env.step(old_grid, 0)  # Try any move (no empty cells to merge)
        self.assertEqual(game_over, True)  # The game should be over

    def test_merge_tiles(self):
        """Test if the _merge_tiles method correctly merges tiles."""
        line = np.array([2, 2, 0, 4])
        expected_merged_line = np.array([4, 4, 0, 0])  # The first two 2's should merge into 4
        
        merged_line = self.env._merge_tiles(line)
        self.assertTrue(np.array_equal(merged_line, expected_merged_line))  # Should merge the first two 2's into 4 and the 4 stays

    def test_is_game_over_no_moves(self):
        """Test if the game over condition works properly (no moves left)."""
        grid = np.array([[2, 4, 8, 16],
                         [32, 64, 128, 256],
                         [512, 1024, 2048, 4096],
                         [8192, 16384, 32768, 65536]])
        game_over = self.env.is_game_over(grid)
        self.assertTrue(game_over)  # The game should be over

    def test_add_new_tile(self):
        """Test if a new tile is added correctly to the grid."""
        grid = np.array([[0, 3, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        grid_with_tile = self.env._add_new_tile(grid)
        self.assertTrue(np.any(grid_with_tile == 2) or np.any(grid_with_tile == 4))  # Check if a tile is added
        self.assertEqual(np.sum(grid_with_tile == 0), 14)  # One empty cell should be replaced with a new tile

if __name__ == "__main__":
    unittest.main()
