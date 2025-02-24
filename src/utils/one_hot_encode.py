import torch as th

def to_one_hot(grid: th.Tensor, n_channels: int) -> th.Tensor:
    """
    Convert a batch of 2048 grids into a one-hot encoded tensor using PyTorch.
    
    Args:
        grid (torch.Tensor): A batch of 4x4 grids of shape (batch_size, 4, 4).
        n_channels (int): The number of channels (log2 of max tile + 1).
    
    Returns:
        th.Tensor: One-hot encoded tensor of shape (batch_size, n_channels, 4, 4).
    """
    exponents = th.where(grid > 0, grid.log2().to(th.int64), th.zeros_like(grid, dtype=th.int64))
    one_hot_grid = th.nn.functional.one_hot(exponents.squeeze(1), num_classes=n_channels)  # (batch_size, 4, 4, n_channels)
    
    return one_hot_grid.permute(0, 3, 1, 2).float() # (batch_size, n_channels, 4, 4)

def to_int(one_hot_grid: th.tensor) -> th.tensor:
    """
    Reverse the one-hot encoded tensor back to the original grid values.
    
    Args:
        one_hot_grid (th.Tensor): A one-hot encoded tensor of shape (batch_size, n_channels, 4, 4).
    
    Returns:
        th.Tensor: The original grid tensor of shape (batch_size, 4, 4).
    """
    # Get the index of the maximum value along the channel dimension (n_channels)
    
    exponents = th.argmax(one_hot_grid, dim=1).to(th.float32)
    
    grid = 2 ** exponents.unsqueeze(1)
    
    grid[grid == 1] -= 1
    
    return grid.float()