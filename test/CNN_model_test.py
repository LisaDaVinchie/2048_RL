import unittest
import sys
import os
import torch as th
from torch import nn
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

# Add the parent directory to the path so that we can import the game module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.CNN_model import CNN_model

class TestCNNModel(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.model_params = {
            "CNN_model": {
                "middle_channels": [64, 64, 64],
                "kernel_sizes": [3, 4, 3],
                "padding": [1, 1, 1],
                "softmax": False
            },
            "agent":{
                "state_size": 16,
                "action_size": 4,
                "grid_size": 4,
                "gamma": 0.95,
                "epsilon": 1.0,
                "epsilon_decay": 0.00001,
                "epsilon_min": 0.3,
                "buffer_maxlen": 2000,
                "batch_size": 64,
                "target_update_freq": 10
            }      
        }
        
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.model_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        self.model = CNN_model(params_path=self.params_path)

        self.batch_size = 10
        self.input_tensor = th.rand((self.batch_size, 1, 4, 4))
        
    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
        
    def test_model_initialization(self):
        """Test if the model initializes correctly"""
        # Test if the model initializes correctly
        
        # Check if the model has the correct attributes
        self.assertEqual(self.model.grid_size, 4)
        self.assertEqual(self.model.action_size, 4)
        self.assertEqual(self.model.middle_channels, [64, 64, 64])
        self.assertEqual(self.model.kernel_sizes, [3, 4, 3])
        self.assertEqual(self.model.padding, [1, 1, 1])
        self.assertFalse(self.model.softmax)
        
    def test_forward_pass(self):
        """Test if the forward pass of the model works correctly"""
        # Test the forward pass of the model
        output = self.model(self.input_tensor)
        
        # Check if the output shape is correct
        self.assertEqual(output.shape, (self.batch_size, self.model.action_size))
        
        # Check if the output is a probability distribution (if softmax is True)
        if self.model.softmax:
            self.assertTrue(th.allclose(output.sum(dim=1), th.ones(self.batch_size)))
        
    def test_invalid_kernel_size(self):
        # Test if the model raises an error for invalid kernel sizes
        self.model = CNN_model(params_path=self.params_path)
        with self.assertRaises(ValueError):
            model1 = CNN_model(
                params_path=self.params_path,
                grid_size=self.model.grid_size,
                action_size=self.model.action_size,
                middle_channels=self.model.middle_channels,
                kernel_sizes=(10, 10, 10),  # Invalid kernel sizes
                padding=self.model.padding,
                softmax=self.model.softmax
            )

if __name__ == "__main__":
    unittest.main()