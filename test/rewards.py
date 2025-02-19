import unittest
import numpy as np
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from rewards import maxN_emptycells_reward

class TestMaxNEmptyCellsReward(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.reward_params = {
            "rewards": {
                "max_tile_reward": 10,
                "empty_cells_reward": 2,
                "game_over_penalty": 1000
            }
        }
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.reward_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        print(f"Temporary file created at: {self.params_path}")  # For debugging

    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()

    def test_reward_with_json_params(self):
        """Test if the function correctly loads parameters from the JSON file."""
        old_grid = np.array([[2, 0, 2, 4], [0, 4, 8, 2], [16, 8, 4, 2], [32, 16, 8, 4]])
        new_grid = np.array([[4, 0, 2, 4], [0, 4, 8, 2], [16, 8, 4, 2], [32, 16, 8, 4]])
        reward = maxN_emptycells_reward(old_grid, new_grid, False, self.params_path)
        expected_reward = 2  # Score increase by 2
        self.assertEqual(reward, expected_reward)
    
    def test_reward_with_manual_params(self):
        """Test if manually provided parameters override the JSON values."""
        old_grid = np.array([[2, 0, 2, 4], [0, 4, 8, 2], [16, 8, 4, 2], [32, 16, 8, 4]])
        new_grid = np.array([[4, 0, 2, 4], [0, 4, 8, 2], [16, 8, 4, 2], [32, 16, 8, 4]])
        reward = maxN_emptycells_reward(old_grid, new_grid, False, self.params_path, max_tile_reward=20)
        expected_reward = 2  # Should not change because max_tile_reward is not relevant here
        self.assertEqual(reward, expected_reward)

    def test_reward_with_game_over_penalty(self):
        """Test if game over penalty is correctly applied."""
        old_grid = np.array([[2, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]])
        new_grid = old_grid.copy()
        reward = maxN_emptycells_reward(old_grid, new_grid, True, self.params_path)
        expected_reward = -1000  # Only the game-over penalty should apply
        self.assertEqual(reward, expected_reward)
    
    def test_reward_with_more_empty_cells(self):
        """Test if the function correctly rewards empty cell increase."""
        old_grid = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
        new_grid = np.array([[2, 2, 2, 0], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
        reward = maxN_emptycells_reward(old_grid, new_grid, False, self.params_path)
        expected_reward = 0  # One new empty cell (2 points)
        self.assertEqual(reward, expected_reward)

if __name__ == "__main__":
    unittest.main()