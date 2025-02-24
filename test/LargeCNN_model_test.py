import torch as th
from torch import nn
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
                "middle_channels": [128, 128, 128, 128]
            },
            "agent":{
                "action_size": 4,
                "grid_size": 4,
                "n_channels": 11
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
        self.assertEqual(self.model.n_channels, 11)
        self.assertEqual(self.model.middle_channels, [128, 128, 128, 128])
    
    def create_test_tensor(self, batch_size):
        input_tensor = th.zeros((batch_size, self.model.n_channels, self.model.grid_size, self.model.grid_size))
        for i in range(batch_size):
            for row in range (self.model.grid_size):
                for col in range(self.model.grid_size):
                    random_position = th.randint(0, self.model.n_channels, (1,)).item()
                    input_tensor[i, random_position, row, col] = 1
        return input_tensor
    
    def test_forward_1tensor(self):
        batch_size = 1
        input_tensor = self.create_test_tensor(batch_size)
        output = self.model(input_tensor)
        self.assertEqual(output.size(), (batch_size, 4))
        
    def test_forward_many_tensor(self):
        batch_size = 10
        input_tensor = self.create_test_tensor(batch_size)
        output = self.model(input_tensor)
        self.assertEqual(output.size(), (batch_size, 4))
    
if __name__ == "__main__":
    unittest.main()