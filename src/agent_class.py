import numpy as np
from collections import deque
import torch as th
import torch.nn as nn

class DQN_Agent:
    """Handles
    - decision making
    - replay buffer
    - training"""
    def __init__(self, model: nn.Module,
                 state_size: int, action_size: int,
                 gamma: float=0.95, epsilon: float=1.0,
                 epsilon_decay: float=0.995, epsilon_min: float=0.01,
                 buffer_maxlen: int=2000, batch_size: int=32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.model = model #The nn model
        
        self.target_model = model.clone()
        
        
        self.replay_buffer = deque(maxlen=buffer_maxlen)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
    def store_to_buffer(self, state: th.tensor, action,
                        reward, next_state: th.tensor, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def choose_action_(self, state):
        "Choose an action based on the epsilon-greedy policy"
        if np.random.rand() <= self.epsilon:
            # return a random number between 0 and 4
            return np.random.choice(self.action_size)
        
        # Predict the Q values of the state
        q_values = self.model(state)
        return th.argmax(q_values).item()