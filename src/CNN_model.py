import torch as th
from torch import nn
from torch.nn import functional as F
from typing import Tuple


class CNN_model(nn.Module):
    def __init__(self, grid_size: int, action_size: int,
                 middle_channels: Tuple[int, int, int] = (64, 64, 64),
                 kernel_sizes: Tuple[int, int, int] = (3, 3, 3),
                 padding: Tuple[int, int, int] = (1, 1, 1)):
        super(CNN_model, self).__init__()
        
        final_grid_size = grid_size - sum(kernel_sizes) + len(kernel_sizes) + 2 * sum(padding)
        
        if final_grid_size < 1:
            raise ValueError("The kernel sizes are too large for the grid size")
        
        strides = [1 for _ in range(len(kernel_sizes))]
        in_channels = 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=middle_channels[0],
                               kernel_size=kernel_sizes[0],
                               stride=strides[0],
                               padding=padding[0])
        self.conv2 = nn.Conv2d(in_channels=middle_channels[0],
                               out_channels=middle_channels[1],
                               kernel_size=kernel_sizes[1],
                               stride=strides[1],
                               padding=padding[1])
        self.conv3 = nn.Conv2d(in_channels=middle_channels[1],
                               out_channels=middle_channels[2],
                               kernel_size=kernel_sizes[2],
                               stride=strides[2],
                               padding=padding[2])
        
        self.fc1 = nn.Linear(middle_channels[2] * final_grid_size * final_grid_size, action_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x: th.Tensor, softmax: bool = True) -> th.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        if softmax:
            return F.softmax(x, dim=1)
        else:
            return x