import math
from pathlib import Path
from utils.import_params_json import load_config

def exponential(epsilon: float, episode: int, params_path: Path, epsilon_decay: float = None, epsilon_min: float = None):
    agent_params = load_config(params_path, ["agent"]).get("agent", {})
    eps_decay = epsilon_decay if epsilon_decay is not None else agent_params.get("epsilon_decay", 0.995)
    eps_min = epsilon_min if epsilon_min is not None else agent_params.get("epsilon_min", 0.01)
    
    if epsilon > eps_min:
        new_epsilon = eps_min + (epsilon - eps_min) * math.exp(-eps_decay * episode)
    
    return new_epsilon
        
def multiply(epsilon: float, episode: int, params_path: Path, epsilon_decay: float = None, epsilon_min: float = None, explore_for: int = None):
    agent_params = load_config(params_path, ["agent"]).get("agent", {})
    eps_decay = epsilon_decay if epsilon_decay is not None else agent_params.get("epsilon_decay", 0.995)
    eps_min = epsilon_min if epsilon_min is not None else agent_params.get("epsilon_min", 0.01)
    exp_for = explore_for if explore_for is not None else agent_params.get("explore_for", 5000)
    
    if episode % exp_for == 0:
            new_epsilon = new_epsilon * 2
    else:
        new_epsilon = max(eps_min, epsilon * eps_decay)
    
    return new_epsilon 