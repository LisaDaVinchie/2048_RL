import numpy as np
from pathlib import Path


from utils.import_params_json import load_config

def maxN_emptycells_reward(old_grid: np.ndarray, new_grid: np.ndarray, is_game_over: bool, params_path: Path, max_tile_reward: int = None, empty_cells_reward: int = None, game_over_penalty: int = None) -> int:
    """
    Calculate the reward based on the differences between the old grid and the new grid.
    
    Parameters:
        old_grid (np.ndarray): The grid before the move.
        new_grid (np.ndarray): The grid after the move.
        is_game_over (bool): Whether the game is over after the move.
        max_tile_reward (int): The reward for increasing the maximum tile value.
        empty_cells_reward (int): The reward for increasing the number of empty cells.
        game_over_penalty (int): The penalty for losing the game (positive value, will be subtracted).
    
    Returns:
        int: The calculated reward.
    """
    
    # Load the reward parameters from the configuration file
    reward_params = load_config(params_path, ["rewards"]).get("rewards", {})
    
    max_tile_reward = max_tile_reward if max_tile_reward is not None else reward_params.get("max_tile_reward", 10)
    empty_cells_reward = empty_cells_reward if empty_cells_reward is not None else reward_params.get("empty_cells_reward", 2)
    game_over_penalty = game_over_penalty if game_over_penalty is not None else reward_params.get("game_over_penalty", 1000)
    
    reward = 0
    
    # Reward for score increase
    old_sum = np.sum(old_grid)
    new_sum = np.sum(new_grid)
    reward += new_sum - old_sum
    
    # print("Score reward: ", new_sum - old_sum, flush=True)
    
    # Bonus reward for increasing the maximum tile value
    old_max = np.max(old_grid)
    new_max = np.max(new_grid)
    if new_max > old_max:
        reward += new_max * max_tile_reward  # Bonus reward for increasing the max tile
        
    # print("Max tile reward: ", new_max * max_tile_reward, flush=True)
    
    # Small reward for increasing the number of empty cells
    old_empty = np.sum(old_grid == 0)
    new_empty = np.sum(new_grid == 0)
    reward += (new_empty - old_empty) * empty_cells_reward  # Small reward for more empty cells
    
    # print("Empty tiles reward: ", (new_empty - old_empty) * empty_cells_reward, flush=True)
    
    # Penalty for game over
    if is_game_over:
        reward -= game_over_penalty  # Heavy penalty for losing the game
    
    # print("Game over penalty: ", -game_over_penalty, flush=True)
    
    return reward