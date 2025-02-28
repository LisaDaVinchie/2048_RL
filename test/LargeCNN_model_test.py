import torch as th
import json
from tempfile import NamedTemporaryFile
from pathlib import Path
import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.models import Large_CNN

class TestLargeCNNModel(unittest.TestCase):
    def setUp(self):
        self.model_params = {
            "Large_CNN": {
                "middle_channels": [128, 128, 128, 128],
                "kernel_sizes": [1, 3, 5, 7]
            },
            "agent":{
                "action_size": 4,
                "grid_size": 4,
                "n_channels": 12
            },
            "training": {
                "representation_kind": "one_hot"
            }
        }
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.model = Large_CNN(params_path=self.params_path)
    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
        
    def test_model_initialization(self):
        self.assertEqual(self.model.grid_size, 4)
        self.assertEqual(self.model.action_size, 4)
        self.assertEqual(self.model.n_channels, 12)
        self.assertEqual(self.model.middle_channels, [128, 128, 128, 128])
        self.assertEqual(self.model.kernel_sizes, [1, 3, 5, 7])
    
    def create_test_tensor(self, batch_size, n_channels):
        tensor = th.zeros((batch_size, n_channels, 4, 4))
        for b in range(batch_size):
            for c in range(n_channels):
                idx = th.randint(0, 16, (1,))
                tensor[b, c, idx // 4, idx % 4] = 1
        return tensor
        
    
    def test_forward_1tensor(self):
        batch_size = 1
        input_tensor = self.create_test_tensor(batch_size, self.model.n_channels)
        output = self.model(input_tensor)
        self.assertEqual(output.size(), (batch_size, 4))
        
    def test_forward_many_tensor(self):
        batch_size = 10
        input_tensor = self.create_test_tensor(batch_size, self.model.n_channels)
        output = self.model(input_tensor)
        self.assertEqual(output.size(), (batch_size, 4))
    
if __name__ == "__main__":
    unittest.main()