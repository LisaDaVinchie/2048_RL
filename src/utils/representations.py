# import torch as th
import numpy as np

def to_one_hot(grid: np.ndarray, n_channels: int) -> np.ndarray:
    # exponents = np.log2(grid, where=(grid > 0), out=np.zeros_like(grid, dtype=float)).astype(int)
    # one_hot_grid = np.eye(n_channels, dtype=np.float32)[exponents]
    """Transform a batch of 2048 grids into a one-hot encoded tensor.
    Expects grid of shape (batch_size, width, height).
    Returns tensor of shape (batch_size, n_channels, width, height).
    """
    exponents = np.log2(grid, where=(grid > 0), out=np.zeros_like(grid, dtype=float)).astype(int)
    one_hot_grid = np.eye(n_channels, dtype=np.float32)[exponents]

    return np.moveaxis(one_hot_grid, -1, 0)


def from_one_hot(one_hot_grid: np.ndarray) -> np.ndarray:
    """Reverse the one-hot encoded tensor back to the original grid values.
    Takes as input a tensor of shape (n_channels, width, height).
    """
    exponents = np.argmax(one_hot_grid, axis=0)
    grid = 2 ** exponents
    grid[grid == 1] -= 1
    return grid

def to_log2(grid: np.ndarray, n_channels: int) -> np.ndarray:
    return np.log2(grid, where=(grid > 0), out=np.zeros_like(grid, dtype=float)).astype(np.float32) / (n_channels - 1)

def from_log2(log2_grid: np.ndarray, n_channels: int) -> np.ndarray:
    grid =  2 ** (log2_grid * (n_channels - 1))
    grid[grid == 1] -= 1
    return grid

# def to_one_hot(grid: th.Tensor, n_channels: int) -> th.Tensor:
#     """
#     Convert a batch of 2048 grids into a one-hot encoded tensor using PyTorch.
    
#     Args:
#         grid (torch.Tensor): A batch of 4x4 grids of shape (batch_size, 4, 4).
#         n_channels (int): The number of channels (log2 of max tile + 1).
    
#     Returns:
#         th.Tensor: One-hot encoded tensor of shape (batch_size, n_channels, 4, 4).
#     """
#     with th.no_grad():
#         exponents = th.where(grid > 0, grid.log2().to(th.int64), th.zeros_like(grid, dtype=th.int64))
#         one_hot_grid = th.nn.functional.one_hot(exponents.squeeze(1), num_classes=n_channels)  # (batch_size, 4, 4, n_channels)
    
#     return one_hot_grid.permute(0, 3, 1, 2).float() # (batch_size, n_channels, 4, 4)

# def to_int(one_hot_grid: th.tensor) -> th.tensor:
#     """
#     Reverse the one-hot encoded tensor back to the original grid values.
    
#     Args:
#         one_hot_grid (th.Tensor): A one-hot encoded tensor of shape (batch_size, n_channels, 4, 4).
    
#     Returns:
#         th.Tensor: The original grid tensor of shape (batch_size, 4, 4).
#     """
#     # Get the index of the maximum value along the channel dimension (n_channels)
    
#     exponents = th.argmax(one_hot_grid, dim=1).to(th.float32)
    
#     grid = 2 ** exponents.unsqueeze(1)
    
#     grid[grid == 1] -= 1
    
#     return grid.float()

# def switch_representation(x: np.ndarray, representation_kind: str, n_channels: int) -> th.Tensor:
#     if representation_kind == "raw":
#         return x
#     elif representation_kind == "log2":
#         return np.log2(x, where=(x > 0), out=np.zeros_like(x, dtype=float)).astype(np.float32)
#     elif representation_kind == "one_hot":
#         return to_one_hot_np(x, n_channels)
#     else:
#         raise ValueError("Invalid representation kind")