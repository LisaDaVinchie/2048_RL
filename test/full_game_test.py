import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.game_class import Game2048_env

class TestGame2048(unittest.TestCase):
    
    def setUp(self):
        self.env = Game2048_env(size=4)
    
    def test_initial_state(self):
        """Test that the initial grid contains exactly one nonzero tile."""
        grid = self.env.reset()
        self.assertEqual(np.count_nonzero(grid), 1)
    
    def test_move_up(self):
        """Test the upward movement and merging behavior."""
        grid = np.array([
            [0, 0, 0, 2],
            [0, 0, 0, 2],
            [0, 0, 4, 0],
            [0, 0, 0, 0]
        ])
        expected_result = np.array([
            [0, 0, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        new_grid, _ = self.env._move(grid, action=0)
        np.testing.assert_array_equal(new_grid, expected_result)
    
    def test_move_down(self):
        """Test the downward movement and merging behavior."""
        grid = np.array([
            [0, 0, 0, 2],
            [0, 0, 0, 2],
            [0, 0, 4, 0],
            [0, 0, 0, 0]
        ])
        expected_result = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 4, 4]
        ])
        new_grid, _ = self.env._move(grid, action=1)
        np.testing.assert_array_equal(new_grid, expected_result)
    
    def test_game_over(self):
        """Test if the game correctly detects game over conditions."""
        grid = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        self.assertTrue(self.env.is_game_over(grid))
    
    def test_full_game(self):
        """Simulate a full random game to ensure it terminates correctly."""
        grid = self.env.reset()
        moves = 0
        max_tile = 0
        while not self.env.is_game_over(grid) and moves < 500:
            action = np.random.choice([0, 1, 2, 3])  # Random action
            grid, _, _ = self.env.step(grid, action)
            max_tile = max(max_tile, np.max(grid))
            moves += 1
        self.assertTrue(self.env.is_game_over(grid) or moves == 500)

if __name__ == '__main__':
    unittest.main()