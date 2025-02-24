import torch as th
import numpy as np
import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.utils.one_hot_encode import to_one_hot as to_one_hot_torch, to_int

class TestOneHot(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_channels = 12
        sample_grid = np.zeros((self.batch_size, 1, 4, 4))

        possible_values = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        for grid in sample_grid:
            for i in range(4):
                for j in range(4):
                    grid[0, i, j] = np.random.choice(possible_values)

        self.sample_grid_th = th.tensor(sample_grid).float()
        
    def test_one_hot_functions(self):
        
        # Convert the grid to one-hot encoding
        one_hot_grid = to_one_hot_torch(self.sample_grid_th, self.n_channels)
        
        self.assertEqual(one_hot_grid.size(), (self.batch_size, self.n_channels, 4, 4))
        
        print("Test passed!")
    
    def test_reverse_one_hot(self):
        # Convert the grid to one-hot encoding
        one_hot_grid = to_one_hot_torch(self.sample_grid_th, self.n_channels)
        
        # Convert the one-hot grid back to the original grid
        grid = to_int(one_hot_grid)
        
        # Check that the arrays grid and self.sample_grid_th are the same
        self.assertTrue(th.equal(grid.float(), self.sample_grid_th.float()))

if __name__ == "__main__":
    unittest.main()
