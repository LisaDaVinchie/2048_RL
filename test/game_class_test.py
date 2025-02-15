import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.game_class import Game2048_env

class TestGame2048Env(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        # Mocking the reward function and parameters path for testing
        self.mock_params_path = Path("/mock/path/to/params.json")
        self.mock_reward_function = MagicMock(return_value=10)  # Mock reward value
        self.env = Game2048_env(self.mock_params_path, self.mock_reward_function)
    
    def test_reset(self):
        """Test if the reset method correctly initializes the grid."""
        grid = self.env.reset()
        self.assertEqual(grid.shape, (4, 4))  # Check if the grid size is correct
        self.assertTrue(np.any(grid == 2) or np.any(grid == 4))  # Check if there's at least one new tile (2 or 4)
        self.assertEqual(np.sum(grid == 0), 15)  # There should be exactly 15 zeros (one tile should be added)

    def test_step_move_left(self):
        """Test if the step method works correctly with a left move."""
        self.env.reset()
        old_grid = self.env.grid.copy()
        new_grid, reward, game_over = self.env.step(2)  # Move left
        self.assertNotEqual(np.sum(old_grid), np.sum(new_grid))  # The grid should change after the move
        self.assertEqual(game_over, False)  # The game should not be over after one move

    def test_step_game_over(self):
        """Test if the step method handles the game over condition."""
        # Set a grid where the game is over (no empty cells and no adjacent merges)
        self.env.grid = np.array([[2, 4, 8, 16],
                                  [32, 64, 128, 256],
                                  [512, 1024, 2048, 4096],
                                  [8192, 16384, 32768, 65536]])
        new_grid, reward, game_over = self.env.step(0)  # Try any move (no empty cells to merge)
        self.assertEqual(game_over, True)  # The game should be over

    def test_step_move_right(self):
        """Test if the step method works correctly with a right move."""
        self.env.reset()
        old_grid = self.env.grid.copy()
        new_grid, reward, game_over = self.env.step(3)  # Move right
        self.assertNotEqual(np.sum(old_grid), np.sum(new_grid))  # The grid should change after the move
        self.assertEqual(game_over, False)  # The game should not be over after one move

    def test_merge_tiles(self):
        """Test if the _merge_tiles method correctly merges tiles."""
        line = np.array([2, 2, 0, 4])
        merged_line = self.env._merge_tiles(line)
        self.assertTrue(np.array_equal(merged_line, np.array([4, 4, 0, 0])))  # Should merge the first two 2's into 4 and the 4 stays

    def test_is_game_over_no_moves(self):
        """Test if the game over condition works properly (no moves left)."""
        grid = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]])
        game_over = self.env.is_game_over(grid)
        self.assertTrue(game_over)  # The game should be over

    def test_add_new_tile(self):
        """Test if a new tile is added correctly to the grid."""
        grid = np.zeros((4, 4), dtype=int)
        grid_with_tile = self.env._add_new_tile(grid)
        self.assertTrue(np.any(grid_with_tile == 2) or np.any(grid_with_tile == 4))  # Check if a tile is added
        self.assertEqual(np.sum(grid_with_tile == 0), 15)  # One empty cell should be replaced with a new tile

if __name__ == "__main__":
    unittest.main()
