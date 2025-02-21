import torch as th
from torch import nn
import json
from tempfile import NamedTemporaryFile
from pathlib import Path
import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.models import LinearModel

class TestLinearModel(unittest.TestCase):
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.model_params = {
            "Linear_model": {
                "middle_channels": [64, 64, 64]
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
        
        self.model = LinearModel(params_path=self.params_path)
        
        self.batch_size = 1
        possible_values = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.input_tensor = th.zeros((self.batch_size, 1, self.model.grid_size, self.model.grid_size))
        for i in range(self.batch_size):
            self.input_tensor[i, 0, :, :] = th.tensor([[possible_values[i % len(possible_values)] for _ in range(self.model.grid_size)] for _ in range(self.model.grid_size)])
        
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
    
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.grid_size, 4)
        self.assertEqual(self.model.action_size, 4)
        self.assertEqual(self.model.n_channels, 11)
    
    def test_forward(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.size(), (self.batch_size, self.model.action_size))
    
if __name__ == "__main__":
    unittest.main()