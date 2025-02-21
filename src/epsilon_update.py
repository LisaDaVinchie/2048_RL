import math
from pathlib import Path
from utils.import_params_json import load_config

class exponential():
    def __init__(self, params_path: Path, epsilon_decay: float = None, epsilon_min: float = None):
        agent_params = load_config(params_path, ["agent"]).get("agent", {})
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else agent_params.get("epsilon_decay", 0.995)
        self.epsilon_min = epsilon_min if epsilon_min is not None else agent_params.get("epsilon_min", 0.01)
        
    def update(self, epsilon: float, episode: int):
        
        if epsilon > self.epsilon_min:
            new_epsilon = self.epsilon_min + (epsilon - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)
        
        return new_epsilon
    
class multiply():
    def __init__(self, params_path: Path, epsilon_decay: float = None, epsilon_min: float = None, explore_for: int = None):
        agent_params = load_config(params_path, ["agent"]).get("agent", {})
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else agent_params.get("epsilon_decay", 0.995)
        self.epsilon_min = epsilon_min if epsilon_min is not None else agent_params.get("epsilon_min", 0.01)
        self.explore_for = explore_for if explore_for is not None else agent_params.get("explore_for", 500)
    
    def update(self, epsilon: float, episode: int):
        new_epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
        return new_epsilon