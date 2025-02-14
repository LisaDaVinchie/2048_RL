import numpy as np
from collections import deque
import random
import torch as th
import torch.nn as nn

class DQN_Agent:
    """Handles
    - decision making
    - replay buffer
    - training"""
    def __init__(self, model: nn.Module, loss_function, optimizer: th.optim.Optimizer,
                 state_size: int, action_size: int,
                 gamma: float=0.95, epsilon: float=1.0,
                 epsilon_decay: float=0.995, epsilon_min: float=0.01,
                 buffer_maxlen: int=2000, batch_size: int=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.target_model = model.clone()
        self.replay_buffer = deque(maxlen=buffer_maxlen)
        
        self.model = model #The nn model
        self.loss_function = loss_function
        self.optimizer = optimizer
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
    def _clone_model(self, model: nn.Module) -> nn.Module:
        clone = type(model)(model.state_dict())
        clone.load_state_dict(model.state_dict())
        return clone
        
    def store_to_buffer(self, state: th.tensor, action,
                        reward, next_state: th.tensor, done: bool):
        """Store the experience to the replay buffer as (state, action, reward, next_state, done)"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def choose_action_(self, state):
        "Choose an action based on the epsilon-greedy policy"
        if np.random.rand() <= self.epsilon:
            # return a random number between 0 and 4
            return np.random.choice(self.action_size)
        
        # Predict the Q values of the state
        with th.no_grad():
            q_values = self.model(state)
        return th.argmax(q_values).item()
    
    def train_step(self):
        """Train the model using the experiences from the replay buffer"""
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # Extract experiences from the minibatch
        states = th.FloatTensor([exp[0] for exp in minibatch])
        actions = th.LongTensor([exp[1] for exp in minibatch])
        rewards = th.FloatTensor([exp[2] for exp in minibatch])
        next_states = th.FloatTensor([exp[3] for exp in minibatch])
        dones = th.BoolTensor([exp[4] for exp in minibatch])
        
        # Compute the Q values of the states
        current_q_values = self.model(states)

        # Compute the Q values of the next states using the target network
        next_q_values = self.target_model(next_states)
        
        # Compute the target Q values
        target = rewards + self.gamma * th.max(next_q_values, dim=1).values * (1 - dones)
        
        # Compute the loss
        self.optimizer.zero_grad()
        self.loss_function.backward()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        """Update the target model"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def save(self, path):
        """Save the model weights"""
        th.save(self.model.state_dict(), path)
        
    def load (self, path):
        """Load the model weights"""
        self.model.load_state_dict(th.load(path))
        self.target_model.load_state_dict(self.model.state_dict())