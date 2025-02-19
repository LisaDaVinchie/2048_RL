import unittest
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.agent_class import DQN_Agent

# Dummy model for testing
class DummyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = x.view(-1, 16)
        return self.fc(x)

# Define the test case class
class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        """Create a temporary JSON file with reward parameters."""
        self.agent_params = {
            "agent": {
                "state_size": 16,
                "action_size": 4,
                "batch_size": 32,
                "gamma": 0.99,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.2,
                "target_update_freq": 1,
                "buffer_maxlen": 10000
            }
        }
        
        # Create a temporary file
        self.temp_json = NamedTemporaryFile(delete=False, mode='w')
        json.dump(self.agent_params, self.temp_json)
        self.temp_json.close()  # Close the file to ensure it's written and available
        self.params_path = Path(self.temp_json.name).resolve()  # Use absolute path
        
        test_model = DummyModel(16, 4)
        
        self.agent = DQN_Agent(
            params_path=self.params_path,
            model=test_model,
            loss_function=nn.SmoothL1Loss(),
            optimizer=optim.SGD(test_model.parameters(), lr=0.01)
        )
    
    def tearDown(self):
        """Delete the temporary JSON file after tests."""
        Path(self.temp_json.name).unlink()
        
    def create_random_state(self) -> th.Tensor:
        """Create a random state tensor."""
        return th.randn((1, 1, 4, 4))

    def test_agent_initialization(self):
        """Test if the agent initializes correctly."""
        self.assertIsInstance(self.agent.model, nn.Module)
        self.assertEqual(self.agent.state_size, self.agent.state_size)
        self.assertEqual(self.agent.action_size, self.agent.action_size)
        self.assertIsInstance(self.agent.replay_buffer, deque)
        self.assertGreater(self.agent.epsilon, 0)
        
        # Check if the parameters are loaded correctly
        self.assertEqual(self.agent.state_size, 16)
        self.assertEqual(self.agent.action_size, 4)
        self.assertEqual(self.agent.batch_size, 32)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.epsilon, 0.1)
        self.assertEqual(self.agent.epsilon_decay, 0.995)
        self.assertEqual(self.agent.epsilon_min, 0.2)
        self.assertEqual(self.agent.target_update_freq, 1)
        self.assertEqual(self.agent.buffer_maxlen, 10000)

    def test_choose_action(self):
        """Test if the agent selects valid actions."""
        state = th.randn(1, self.agent.state_size)
        action, _ = self.agent.choose_action(state)
        self.assertIn(action, range(self.agent.action_size))

    def test_store_to_buffer(self):
        """Test if the replay buffer stores transitions correctly."""
        state = self.create_random_state()
        next_state = self.create_random_state()
        action = np.random.randint(0, self.agent.action_size)
        reward = np.random.rand()
        done = False
        
        self.agent.store_to_buffer(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.replay_buffer), 1)

    def test_train_step(self):
        """Test if training updates the model."""
        for _ in range(self.agent.batch_size):  # Fill buffer
            state = self.create_random_state()
            next_state = self.create_random_state()
            action = np.random.randint(0, self.agent.action_size)
            reward = np.random.rand()
            done = False
            self.agent.store_to_buffer(state, action, reward, next_state, done)

        initial_weights = self.agent.model.fc.weight.clone()
        self.agent.train_step(episode=1)
        self.assertFalse(th.equal(initial_weights, self.agent.model.fc.weight))

    def test_update_target_model(self):
        """Test if the target model is updated correctly."""
        initial_target_weights = self.agent.target_model.fc.weight.clone()
        for _ in range(self.agent.batch_size):  # Fill buffer
            state = self.create_random_state()
            action = np.random.randint(0, self.agent.action_size)
            reward = np.random.rand()
            next_state = self.create_random_state()
            done = False
            self.agent.store_to_buffer(state, action, reward, next_state, done)
        
        self.agent.train_step(episode=1)
        self.assertTrue(th.equal(self.agent.model.fc.weight, self.agent.target_model.fc.weight))
        self.assertFalse(th.equal(initial_target_weights, self.agent.target_model.fc.weight))

    def test_save_and_load(self):
        """Test if the model can be saved and loaded correctly."""
        save_path = "test_model.pth"
        self.agent.save(save_path)
        
        # Modify the model weights to ensure loading works
        with th.no_grad():
            self.agent.model.fc.weight.fill_(0.0)
        
        self.agent.load(save_path)
        self.assertTrue(th.equal(self.agent.model.fc.weight, self.agent.target_model.fc.weight))

if __name__ == "__main__":
    unittest.main()