import torch as th
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from pathlib import Path
from utils.import_params_json import load_config


class CNN_model(nn.Module):
    def __init__(self, params_path: Path, grid_size: int = None, action_size: int = None,
                 middle_channels: Tuple[int, int, int] = None, kernel_sizes: Tuple[int, int, int] = None,
                 padding: Tuple[int, int, int] = None, softmax: bool = None):
        super(CNN_model, self).__init__()
        
        model_params = load_config(params_path, ["agent"]).get("agent", {})
        self.grid_size = grid_size if grid_size is not None else model_params.get("grid_size", 4)
        self.action_size = action_size if action_size is not None else model_params.get("action_size", 4)
        
        model_params = load_config(params_path, ["CNN_model"]).get("CNN_model", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", (16, 32, 64))
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else model_params.get("kernel_sizes", (2, 2, 2))
        self.padding = padding if padding is not None else model_params.get("padding", (1, 1, 1))
        self.softmax = softmax if softmax is not None else model_params.get("softmax", True)
        
        # print("CNN_model params:")
        # print(f"grid_size: {self.grid_size}")
        # print(f"action_size: {self.action_size}")
        # print(f"middle_channels: {self.middle_channels}")
        # print(f"kernel_sizes: {self.kernel_sizes}")
        # print(f"padding: {self.padding}")
        # print(f"softmax: {self.softmax}")
        # print()
        
        
        final_grid_size = self.grid_size - sum(self.kernel_sizes) + len(self.kernel_sizes) + 2 * sum(self.padding)
        
        if final_grid_size < 1:
            raise ValueError("The kernel sizes are too large for the grid size")
        
        self.strides = [1 for _ in range(len(self.kernel_sizes))]
        in_channels = 1
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.middle_channels[0],
                               kernel_size=self.kernel_sizes[0],
                               stride=self.strides[0],
                               padding=self.padding[0])
        self.bn1 = nn.BatchNorm2d(self.middle_channels[0])
        self.conv2 = nn.Conv2d(in_channels=self.middle_channels[0],
                               out_channels=self.middle_channels[1],
                               kernel_size=self.kernel_sizes[1],
                               stride=self.strides[1],
                               padding=self.padding[1])
        self.bn2 = nn.BatchNorm2d(self.middle_channels[1])
        self.conv3 = nn.Conv2d(in_channels=self.middle_channels[1],
                               out_channels=self.middle_channels[2],
                               kernel_size=self.kernel_sizes[2],
                               stride=self.strides[2],
                               padding=self.padding[2])
        self.bn3 = nn.BatchNorm2d(self.middle_channels[2])
        self.fc1 = nn.Linear(self.middle_channels[2] * final_grid_size * final_grid_size, 128)
        self.fc2 = nn.Linear(128, self.action_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = th.log2(x + 1)/11
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        if self.softmax:
            return F.softmax(x, dim=1)
        else:
            return x