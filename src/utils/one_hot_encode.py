import torch as th
import numpy as np

def to_one_hot(grid: np.ndarray, n_channels: int) -> th.Tensor:
    """
    Convert the grid into a one-hot encoded tensor.
    
    Args:
        grid (np.ndarray): The 4x4 grid of tile values.
        max_tile_value (int): The maximum tile value to encode (e.g., 2048).
    
    Returns:
        th.Tensor: A tensor of shape (1, num_channels, grid_size, grid_size),
                   where num_channels is the number of unique tile values.
    """
    # Create a list of all possible tile values (powers of 2)
    tile_values = [2 ** i for i in range(1, int(n_channels) + 1)]
    num_channels = len(tile_values)
    
    # Initialize an empty tensor for the one-hot encoded grid
    one_hot_grid = th.zeros((1, num_channels, grid.shape[0], grid.shape[1]))
    
    # Fill the one-hot grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] > 0:
                # Find the index of the tile value in the list
                tile_index = tile_values.index(grid[i, j])
                one_hot_grid[0, tile_index, i, j] = 1
    
    return one_hot_grid