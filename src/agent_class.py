import numpy as np
from collections import deque
import random
import torch as th
import torch.nn as nn
import copy
import math
from pathlib import Path
from utils.import_params_json import load_config
from epsilon_update import exponential, multiply

class DQN_Agent:
    """Handles
    - decision making
    - replay buffer
    - training"""
    def __init__(self, params_path: Path, model: nn.Module, loss_function,
                 optimizer: th.optim.Optimizer, grid_size: int = None,
                 action_size: int = None, gamma: float = None,
                 epsilon: float = None, epsilon_decay: float = None,
                 epsilon_min: float = None, buffer_maxlen: int = None,
                 batch_size: int = None, target_update_freq: int = None,
                 learn_iterations: int = None, explore_for: int = None,
                 steps_ahead: int = None, n_channels: int = None,
                 epsilon_decay_kind: str = None, tau: float = None):
        """Initialise the agent with the parameters"""
        
        agent_params = load_config(params_path, ["agent"]).get("agent", {})
        self.grid_size = grid_size if grid_size is not None else agent_params.get("grid_size", 4)
        self.action_size = action_size if action_size is not None else agent_params.get("action_size", 4)
        self.gamma = gamma if gamma is not None else agent_params.get("gamma", 0.95)
        self.epsilon = epsilon if epsilon is not None else agent_params.get("epsilon", 1.0)
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else agent_params.get("epsilon_decay", 0.995)
        self.epsilon_min = epsilon_min if epsilon_min is not None else agent_params.get("epsilon_min", 0.01)
        self.buffer_maxlen = buffer_maxlen if buffer_maxlen is not None else agent_params.get("buffer_maxlen", 2000)
        self.batch_size = batch_size if batch_size is not None else agent_params.get("batch_size", 32)
        self.target_update_freq = target_update_freq if target_update_freq is not None else agent_params.get("target_update_freq", 10)
        self.learn_iterations = learn_iterations if learn_iterations is not None else agent_params.get("learn_iterations", 1)
        self.n_channels = n_channels if n_channels is not None else agent_params.get("n_channels", 11)
        self.epsilon_decay_kind = epsilon_decay_kind if epsilon_decay_kind is not None else agent_params.get("epsilon_decay_kind", "multiply")
        self.explore_for = explore_for if explore_for is not None else agent_params.get("explore_for", 500)
        self.steps_ahead = steps_ahead if steps_ahead is not None else agent_params.get("steps_ahead", 1)
        self.tau = tau if tau is not None else agent_params.get("tau", 0.001)
        
        if self.epsilon_decay_kind == "exponential":
            self.epsilon_update_class = exponential(params_path)
        elif self.epsilon_decay_kind == "multiply":
            self.epsilon_update_class = multiply(params_path)
        else:
            raise ValueError("Invalid epsilon decay kind")
        
        self.replay_buffer = deque(maxlen=self.buffer_maxlen)
        self.n_step_buffer = deque(maxlen=self.steps_ahead)
        self.state_size = self.grid_size * self.grid_size
        self.model = model
        self.target_model= self._clone_model(self.model)
        self.loss_function = loss_function
        self.optimizer = optimizer
        
        self.loss = None
        self.current_q_values = None

    def _clone_model(self, model: nn.Module) -> nn.Module:
        """Properly clones a PyTorch model."""
        model_clone = copy.deepcopy(model)  # Deep copy ensures all parameters are new
        model_clone.load_state_dict(model.state_dict())  # Copy weights
        model_clone.eval()  # Set to eval mode (optional but recommended)
        return model_clone
        
    def store_to_buffer(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool):
        """Store the experience to the replay buffer as (state, action, reward, next_state, done)"""
        
        if state is None:
            raise ValueError("State cannot be None")
        if next_state is None:
            raise ValueError("Next state cannot be None")
        
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we are using n-step learning, we need to wait until we have n steps
        if done or len(self.n_step_buffer) == self.steps_ahead:
            # Compute n-step reward
            R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
            state, action, _, next_state, done = self.n_step_buffer[0]  # Use the first transition
            self.replay_buffer.append((state, action, R, next_state, done))
            self.n_step_buffer.clear()
    
    def _compute_Q_values(self, states: th.tensor, actions: th.tensor, rewards: th.tensor, next_states: th.tensor, dones: th.tensor):
        """Compute the Q values for the current state and the target Q values"""
        
        # Q pred
        current_q_values = self.model(states).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        
        non_terminal_mask = 1 - dones.float()
        # Q target
        with th.no_grad():
            next_q_values = (self.target_model(next_states).max(dim=1)[0]) * non_terminal_mask
            # Compute the target Q values, discounting the future rewards
            target_q_values = rewards + self.gamma * next_q_values * non_terminal_mask
            
        return current_q_values, target_q_values
    
    # Exploration step
    def train_step(self, episode: int):
        """Train the model using the experiences from the replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            print(f"Episode: {episode}, Not enough samples in the replay buffer")
            return
        
        self.model.train()
        for epoch in range(self.learn_iterations):
            
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            
            # Extract experiences from the minibatch
            states = th.tensor(np.array([exp[0] for exp in minibatch]), dtype=th.float32)
            actions = th.LongTensor([exp[1] for exp in minibatch])
            rewards = th.FloatTensor([exp[2] for exp in minibatch])
            next_states = th.tensor(np.array([exp[3] for exp in minibatch]), dtype=th.float32)
            dones = th.BoolTensor([exp[4] for exp in minibatch])
            
            self.current_q_values, target_q_values = self._compute_Q_values(states.unsqueeze(1), actions, rewards, next_states.unsqueeze(1), dones)
            self.loss = self.loss_function(self.current_q_values, target_q_values)
            
            # Compute the loss
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        
        # Update the epsilon value
        self.epsilon = self.epsilon_update_class.update(self.epsilon, episode)
        
        if episode % self.target_update_freq == 0:
            self._update_target_model_weights()
    
    def choose_action(self, state: np.ndarray, training: bool=True):
        """Choose an exploration or exploitation based on the epsilon-greedy policy
        and track if the action is exploration or exploitation"""
        
        if training and np.random.rand() <= self.epsilon:
            # Exploration: return a random number between 0 and 4
            # return a random number between 0 and 4
            action = np.random.choice(self.action_size)
            is_exploration = True
        else:
            # Exploitation: return the action with the highest Q value
            q_values = self.model(th.tensor(state, dtype=th.float32).unsqueeze(0).unsqueeze(0))
            action = np.argmax(q_values.detach().numpy())
            is_exploration = False
        return action, is_exploration
            
    def _update_target_model_weights(self):
        """Update the target model"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def save(self, path):
        """Save the model weights"""
        th.save(self.model.state_dict(), path)
        
    def load(self, path):
        """Load the model weights"""
        self.model.load_state_dict(th.load(path))
        self.target_model.load_state_dict(self.model.state_dict())