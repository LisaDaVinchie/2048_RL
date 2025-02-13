import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(f"\n Imported path\n{sys.path}\n")

from src.game import game_step, move, is_game_over

class TestGameStep(unittest.TestCase):
    def test_move_left(self):
        # Test moving left
        grid = np.array([
            [2, 0, 2, 0],
            [0, 4, 0, 4],
            [8, 0, 8, 0],
            [0, 16, 0, 16]
        ])
        expected_grid = np.array([
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
            [32, 0, 0, 0]
        ])
        result = game_step(grid, 'left')
        # Check if the grid matches the expected grid (ignoring the new tile)
        self.assertTrue(np.array_equal(result[:, 0], expected_grid[:, 0]))

    def test_move_right(self):
        # Test moving right
        grid = np.array([
            [2, 0, 2, 0],
            [0, 4, 0, 4],
            [8, 0, 8, 0],
            [0, 16, 0, 16]
        ])
        expected_grid = np.array([
            [0, 0, 0, 4],
            [0, 0, 0, 8],
            [0, 0, 0, 16],
            [0, 0, 0, 32]
        ])
        result = game_step(grid, 'right')
        # Check if the grid matches the expected grid (ignoring the new tile)
        self.assertTrue(np.array_equal(result[:, 3], expected_grid[:, 3]))

    def test_move_up(self):
        # Test moving up
        grid = np.array([
            [2, 0, 2, 0],
            [0, 4, 0, 4],
            [8, 0, 8, 0],
            [0, 16, 0, 16]
        ])
        expected_grid = np.array([
            [2, 4, 2, 4],
            [8, 16, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        result = game_step(grid, 'up')
        # Check if the grid matches the expected grid (ignoring the new tile)
        self.assertTrue(np.array_equal(result[:2, :], expected_grid[:2, :]))

    def test_move_down(self):
        # Test moving down
        grid = np.array([
            [2, 0, 2, 0],
            [0, 4, 0, 4],
            [8, 0, 8, 0],
            [0, 16, 0, 16]
        ])
        expected_grid = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 4, 2, 4],
            [8, 16, 8, 16]
        ])
        result = game_step(grid, 'down')
        # Check if the grid matches the expected grid (ignoring the new tile)
        self.assertTrue(np.array_equal(result[2:, :], expected_grid[2:, :]))

    def test_no_move_possible(self):
        # Test when no move is possible
        grid = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        result = game_step(grid, 'left')
        # The grid should remain unchanged
        self.assertTrue(np.array_equal(result, grid))

    def test_new_tile_added(self):
        # Test if a new tile is added after a valid move
        grid = np.array([
            [2, 0, 2, 0],
            [0, 4, 0, 4],
            [8, 0, 8, 0],
            [0, 16, 0, 16]
        ])
        expected_grid = np.array([
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
            [32, 0, 0, 0]
        ])
        result = game_step(grid, 'left')

        # Check if the number of non-zero tiles has increased by 1
        self.assertEqual(np.count_nonzero(result), np.count_nonzero(expected_grid) + 1)

if __name__ == "__main__":
    unittest.main()