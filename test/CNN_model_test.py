import unittest
import numpy as np
import sys
import os
import torch as th

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(f"\n Imported path\n{sys.path}\n")

from src.CNN_model import CNN_model

class TestCNNModel(unittest.TestCase):
    def test_forward(self):
        model = CNN_model(grid_size=4, action_size=4)
        sample_matrix = th.randn(1, 1, 4, 4)
        with th.no_grad():
            result = model.forward(sample_matrix)
        self.assertEqual(result.size(), (1, 4))