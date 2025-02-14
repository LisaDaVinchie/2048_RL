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
        
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.target_model: nn.Module = model.clone()
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
    
    # Exploration step
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
        
        loss = self._compute_loss(states, actions, rewards, next_states, dones)
        
        # Compute the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def _compute_loss(self, states, actions, rewards, next_states, dones):
        """Compute the loss"""
        # Q pred
        current_q_values = self.model(states)
        
        # Q target
        with th.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * ~dones
            loss = self.loss_function(current_q_values, target_q_values)
        return loss
    
    def choose_action_(self, state: th.tensor, epsilon: float):
        """Choose an exploration or exploitation based on the epsilon-greedy policy
        and track if the action is exploration or exploitation"""
        if np.random.rand() <= epsilon:
            # Exploration: return a random number between 0 and 4
            # return a random number between 0 and 4
            action = np.random.choice(self.action_size)
            is_exploration = True
        else:
            # Exploitation: return the action with the highest Q value
            with th.no_grad():
                q_values = self.model(state)
            action = th.argmax(q_values).item()
            is_exploration = False
        return action, is_exploration
            
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