import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.utils.representations import to_one_hot, from_one_hot

class Test2048Encoding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.test_grid = np.array([
            [[0, 2, 4, 8],
             [16, 32, 64, 128],
             [256, 512, 1024, 2048],
             [0, 0, 2, 4]],
            [[2, 4, 8, 16],
             [32, 64, 128, 256],
             [512, 1024, 2048, 0],
             [0, 2, 4, 8]]
        ])  # Shape: [2, 4, 4]
        
        self.n_channels = int(np.log2(self.test_grid.max())) + 2
    
    def test_to_one_hot_shape(self):
        one_hot_encoded = to_one_hot(self.test_grid, self.n_channels)
        expected_shape = (self.batch_size, self.n_channels, 4, 4)
        self.assertEqual(one_hot_encoded.shape, expected_shape)
    
    def test_from_one_hot(self):
        one_hot_encoded = to_one_hot(self.test_grid, self.n_channels)
        decoded_grid = from_one_hot(one_hot_encoded)
        np.testing.assert_array_equal(decoded_grid, self.test_grid)

if __name__ == "__main__":
    unittest.main()