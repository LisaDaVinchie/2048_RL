import unittest
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent_class import DQN_Agent

# Dummy model for testing
class DummyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        return self.fc(x)

# Define the test case class
class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        """Initialize the test environment and agent."""
        self.state_size = 16  # 2048 uses a 4x4 board
        self.action_size = 4  # Up, Down, Left, Right
        
        self.model = DummyModel(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        
        self.agent = DQN_Agent(
            model=self.model,
            loss_function=self.loss_function,
            optimizer=self.optimizer,
            state_size=self.state_size,
            action_size=self.action_size
        )

    def test_agent_initialization(self):
        """Test if the agent initializes correctly."""
        self.assertIsInstance(self.agent.model, nn.Module)
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertIsInstance(self.agent.replay_buffer, deque)
        self.assertGreater(self.agent.epsilon, 0)

    def test_choose_action(self):
        """Test if the agent selects valid actions."""
        state = th.randn(self.state_size)
        action, is_exploration = self.agent.choose_action(state)
        self.assertIn(action, range(self.action_size))

    def test_store_to_buffer(self):
        """Test if the replay buffer stores transitions correctly."""
        state = th.randn(self.state_size)
        next_state = th.randn(self.state_size)
        action = np.random.randint(0, self.action_size)
        reward = np.random.rand()
        done = False
        
        self.agent.store_to_buffer(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.replay_buffer), 1)

    def test_train_step(self):
        """Test if training updates the model."""
        for _ in range(self.agent.batch_size):  # Fill buffer
            self.agent.store_to_buffer(th.randn(self.state_size), np.random.randint(0, self.action_size), np.random.rand(), th.randn(self.state_size), False)

        initial_weights = self.agent.model.fc.weight.clone()
        self.agent.train_step(episode=1)
        self.assertFalse(th.equal(initial_weights, self.agent.model.fc.weight))

if __name__ == "__main__":
    unittest.main()