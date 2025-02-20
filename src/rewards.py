import numpy as np
from pathlib import Path
from utils.import_params_json import load_config

def original_reward(old_grid: np.ndarray, new_grid: np.ndarray, is_game_over: bool, params_path: Path, game_over_penalty: int = None):
    """
    Compute the score gained in a move based on merges.
    
    The score is the sum of all newly created tiles during merging.
    
    Parameters:
        old_grid (np.ndarray): The grid before the move.
        new_grid (np.ndarray): The grid after the move.

    Returns:
        int: The score gained in this move.
    """
    # Load the reward parameters from the configuration file
    reward_params = load_config(params_path, ["rewards"]).get("rewards", {})
    game_over_penalty = game_over_penalty if game_over_penalty is not None else reward_params.get("game_over_penalty", 1000)
    
    score = 0
    for i in range(4):
        for j in range(4):
            if new_grid[i, j] > old_grid[i, j]:  # Tile grew -> merge happened
                score += new_grid[i, j]  # Add merged tile value
                
    if is_game_over:
        score -= game_over_penalty
    

    return score

def log2_merge_reward(old_grid: np.ndarray, new_grid: np.ndarray, is_game_over: bool, params_path: Path, game_over_penalty: int = None):
    
    # Load the reward parameters from the configuration file
    reward_params = load_config(params_path, ["rewards"]).get("rewards", {})
    game_over_penalty = game_over_penalty if game_over_penalty is not None else reward_params.get("game_over_penalty", 1000)
    no_changes_penalty = reward_params.get("no_changes_penalty", 10)

    merged_tiles = new_grid > old_grid
    merged_values = new_grid[merged_tiles]
    
    if merged_values.numel() > 0:
        score = np.sum(np.log2(merged_values))
        
    if np.array_equal(old_grid, new_grid):
        score -= no_changes_penalty
    
    return score
    
    
def maxN_emptycells_reward(old_grid: np.ndarray, new_grid: np.ndarray, is_game_over: bool, params_path: Path, max_tile_reward: int = None, empty_cells_reward: int = None, game_over_penalty: int = None, no_changes_penalty: int = None) -> int:
    """
    Calculate the reward based on the differences between the old grid and the new grid.
    
    Parameters:
        old_grid (np.ndarray): The grid before the move.
        new_grid (np.ndarray): The grid after the move.
        is_game_over (bool): Whether the game is over after the move.
        max_tile_reward (int): The reward for increasing the maximum tile value.
        empty_cells_reward (int): The reward for increasing the number of empty cells.
        game_over_penalty (int): The penalty for losing the game (positive value, will be subtracted).
        no_changes_penalty (int): The penalty for making a move that does not change the grid.
    
    Returns:
        int: The calculated reward.
    """
    
    # Load the reward parameters from the configuration file
    reward_params = load_config(params_path, ["rewards"]).get("rewards", {})
    
    max_tile_reward = max_tile_reward if max_tile_reward is not None else reward_params.get("max_tile_reward", 10)
    empty_cells_reward = empty_cells_reward if empty_cells_reward is not None else reward_params.get("empty_cells_reward", 2)
    game_over_penalty = game_over_penalty if game_over_penalty is not None else reward_params.get("game_over_penalty", 1000)
    no_changes_penalty = no_changes_penalty if no_changes_penalty is not None else reward_params.get("no_changes_penalty", 10)
    
    reward = 0
    
    # Penalty for making a move that does not change the grid
    if np.array_equal(old_grid, new_grid):
        reward -= no_changes_penalty
    
    old_grid = np.log2(old_grid + 1) / 11
    new_grid = np.log2(new_grid + 1) / 11
    
    # Reward for score increase
    old_sum = np.sum(old_grid)
    new_sum = np.sum(new_grid)
    reward += new_sum - old_sum
    
    # print("Score reward: ", new_sum - old_sum, flush=True)
    
    # Bonus reward for increasing the maximum tile value
    old_max = np.max(old_grid)
    new_max = np.max(new_grid)
    if new_max >= old_max:
        reward += new_max * max_tile_reward  # Bonus reward for increasing the max tile
        
    # print("Max tile reward: ", new_max * max_tile_reward, flush=True)
    
    # Small reward for increasing the number of empty cells
    old_empty = np.sum(old_grid == 0)
    new_empty = np.sum(new_grid == 0)
    if new_empty > old_empty:
        reward += np.log2(new_empty - old_empty) * empty_cells_reward  # Small reward for more empty cells
    
    # print("Empty tiles reward: ", (new_empty - old_empty) * empty_cells_reward, flush=True)
    
    # Penalty for game over
    if is_game_over:
        reward -= game_over_penalty  # Heavy penalty for losing the game
    
    # print("Game over penalty: ", -game_over_penalty, flush=True)
    
    return reward

    
def maxN_emptycells_merge_reward(old_grid: np.ndarray, new_grid: np.ndarray, is_game_over: bool, params_path: Path, max_tile_reward: int = None, empty_cells_reward: int = None, game_over_penalty: int = None, no_changes_penalty: int = None) -> int:
    """
    Calculate the reward based on the differences between the old grid and the new grid.
    
    Parameters:
        old_grid (np.ndarray): The grid before the move.
        new_grid (np.ndarray): The grid after the move.
        is_game_over (bool): Whether the game is over after the move.
        max_tile_reward (int): The reward for increasing the maximum tile value.
        empty_cells_reward (int): The reward for increasing the number of empty cells.
        game_over_penalty (int): The penalty for losing the game (positive value, will be subtracted).
        no_changes_penalty (int): The penalty for making a move that does not change the grid.
    
    Returns:
        int: The calculated reward.
    """
    
    # Load the reward parameters from the configuration file
    reward_params = load_config(params_path, ["rewards"]).get("rewards", {})
    
    max_tile_reward = max_tile_reward if max_tile_reward is not None else reward_params.get("max_tile_reward", 10)
    empty_cells_reward = empty_cells_reward if empty_cells_reward is not None else reward_params.get("empty_cells_reward", 2)
    game_over_penalty = game_over_penalty if game_over_penalty is not None else reward_params.get("game_over_penalty", 1000)
    no_changes_penalty = no_changes_penalty if no_changes_penalty is not None else reward_params.get("no_changes_penalty", 10)
    
    reward = 0
    
    
    # Penalty for making a move that does not change the grid
    if np.array_equal(old_grid, new_grid):
        return -no_changes_penalty
        
    old_grid = np.log2(old_grid + 1) / 11
    new_grid = np.log2(new_grid + 1) / 11
    
    # Reward for merging tiles
    for i in range(new_grid.shape[0]):
        for j in range(new_grid.shape[1]):
            if new_grid[i, j] > old_grid[i, j]:  # Tile grew -> merge happened
                reward += new_grid[i, j]  # Add merged tile value
    
    # # Reward for score increase
    # old_sum = np.sum(old_grid)
    # new_sum = np.sum(new_grid)
    # reward += new_sum - old_sum
    
    # Bonus reward for increasing the maximum tile value
    old_max = np.max(old_grid)
    new_max = np.max(new_grid)
    if new_max > old_max:
        reward += new_max * max_tile_reward  # Bonus reward for increasing the max tile
    
    # Small reward/penalty for increasing/decreasing the number of empty cells
    old_empty = np.sum(old_grid == 0)
    new_empty = np.sum(new_grid == 0)
    reward += (new_empty - old_empty) * empty_cells_reward  # Small reward for more empty cells
    
    # Penalty for game over
    if is_game_over:
        reward -= game_over_penalty  # Penalty for losing the game
    
    return int(reward)  # Return as integer for compatibility with other reward functions