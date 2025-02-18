import numpy as np
from collections import deque
import random
import torch as th
import torch.nn as nn
import copy
import math
from pathlib import Path
from utils.import_params_json import load_config

class DQN_Agent:
    """Handles
    - decision making
    - replay buffer
    - training"""
    def __init__(self, params_path: Path, model: nn.Module, loss_function,
                 optimizer: th.optim.Optimizer, state_size: int = None,
                 action_size: int = None, gamma: float = None,
                 epsilon: float = None, epsilon_decay: float = None,
                 epsilon_min: float = None, buffer_maxlen: int = None,
                 batch_size: int = None, target_update_freq: int = None):
        
        agent_params = load_config(params_path, ["agent"]).get("agent", {})
        state_size = state_size if state_size is not None else agent_params.get("state_size", 16)
        action_size = action_size if action_size is not None else agent_params.get("action_size", 4)
        gamma = gamma if gamma is not None else agent_params.get("gamma", 0.95)
        epsilon = epsilon if epsilon is not None else agent_params.get("epsilon", 1.0)
        epsilon_decay = epsilon_decay if epsilon_decay is not None else agent_params.get("epsilon_decay", 0.995)
        epsilon_min = epsilon_min if epsilon_min is not None else agent_params.get("epsilon_min", 0.01)
        buffer_maxlen = buffer_maxlen if buffer_maxlen is not None else agent_params.get("buffer_maxlen", 2000)
        batch_size = batch_size if batch_size is not None else agent_params.get("batch_size", 32)
        target_update_freq = target_update_freq if target_update_freq is not None else agent_params.get("target_update_freq", 10)
        
        print("Agent parameters:")
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Gamma: {gamma}")
        print(f"Epsilon: {epsilon}")
        print(f"Epsilon decay: {epsilon_decay}")
        print(f"Epsilon min: {epsilon_min}")
        print(f"Buffer maxlen: {buffer_maxlen}")
        print(f"Batch size: {batch_size}")
        print(f"Target update frequency: {target_update_freq}")
        
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.replay_buffer = deque(maxlen=buffer_maxlen)
        
        self.model = model #The nn model
        self.target_model= self._clone_model(model)
        self.loss_function = loss_function
        self.optimizer = optimizer
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.loss = None
        self.current_q_values = None

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Properly clones a PyTorch model."""
        model_clone = copy.deepcopy(model)  # Deep copy ensures all parameters are new
        model_clone.load_state_dict(model.state_dict())  # Copy weights
        model_clone.eval()  # Set to eval mode (optional but recommended)
        return model_clone
        
    def store_to_buffer(self, state: th.tensor, action: int, reward: int, next_state: th.tensor, done: bool):
        """Store the experience to the replay buffer as (state, action, reward, next_state, done)"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    # Exploration step
    def train_step(self, episode: int):
        """Train the model using the experiences from the replay buffer"""
        
        if len(self.replay_buffer) < self.batch_size:
            return
        
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # Extract experiences from the minibatch
        states = th.stack([exp[0] for exp in minibatch])
        actions = th.LongTensor([exp[1] for exp in minibatch])
        rewards = th.FloatTensor([exp[2] for exp in minibatch])
        next_states = th.stack([exp[3] for exp in minibatch])
        dones = th.BoolTensor([exp[4] for exp in minibatch])
        
        self.current_q_values, target_q_values = self.compute_Q_values(states, actions, rewards, next_states, dones)
        self.loss = self.loss_function(self.current_q_values, target_q_values)
        
        # Compute the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)
        
        if episode % self.target_update_freq == 0:
            self.update_target_model()

    def compute_Q_values(self, states: th.tensor, actions: th.tensor, rewards: th.tensor, next_states: th.tensor, dones: th.tensor):
        """Compute the Q values for the current state and the target Q values"""
        # Q pred
        current_q_values = self.model(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        
        # Q target
        with th.no_grad():
            next_q_values = (self.target_model(next_states).max(dim=1)[0]) * (1 - dones.float())
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())
        return current_q_values, target_q_values
    
    def choose_action(self, state: th.tensor, training: bool=True):
        """Choose an exploration or exploitation based on the epsilon-greedy policy
        and track if the action is exploration or exploitation"""
        
        if training and np.random.rand() <= self.epsilon:
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
        
    def load(self, path):
        """Load the model weights"""
        self.model.load_state_dict(th.load(path))
        self.target_model.load_state_dict(self.model.state_dict())