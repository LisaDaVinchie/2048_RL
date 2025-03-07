import torch as th
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from pathlib import Path
from utils.import_params_json import load_config

def get_representation_kind(params_path: Path, representation_kind: str = None) -> str:
    config = load_config(params_path, ["training"]).get("training", {})
    return representation_kind if representation_kind is not None else config.get("representation_kind", "log2")

class CNN_model(nn.Module):
    def __init__(self, params_path: Path, grid_size: int = None, action_size: int = None,
                 n_channels: int = None, middle_channels: Tuple[int, int, int, int] = None, kernel_sizes: Tuple[int, int, int] = None,
                 padding: Tuple[int, int, int] = None, representation_kind: str = None):
        super(CNN_model, self).__init__()
        
        model_params = load_config(params_path, ["agent"]).get("agent", {})
        self.grid_size = grid_size if grid_size is not None else model_params.get("grid_size", 4)
        self.action_size = action_size if action_size is not None else model_params.get("action_size", 4)
        self.n_channels = n_channels if n_channels is not None else model_params.get("n_channels", 11)
        
        model_params = load_config(params_path, ["CNN_model"]).get("CNN_model", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", (16, 32, 64, 128))
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else model_params.get("kernel_sizes", (2, 2, 2))
        self.padding = padding if padding is not None else model_params.get("padding", (1, 1, 1))
        self.representation_kind = get_representation_kind(params_path, representation_kind)
        
        self.in_channels = 1
        
        if self.representation_kind == "one_hot":
            self.in_channels *= self.n_channels
            
        final_grid_size = self.grid_size - sum(self.kernel_sizes) + len(self.kernel_sizes) + 2 * sum(self.padding)
        
        if final_grid_size < 1:
            raise ValueError("The kernel sizes are too large for the grid size")
        
        self.strides = [1 for _ in range(len(self.kernel_sizes))]
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
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
        self.fc1 = nn.Linear(self.middle_channels[2] * final_grid_size * final_grid_size, self.middle_channels[3])
        self.fc2 = nn.Linear(self.middle_channels[3], self.action_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
        
class LinearModel(nn.Module):
    def __init__(self, params_path: Path, grid_size: int = None, action_size: int = None, middle_channels: Tuple[int, int, int] = None, n_channels: int = None, representation_kind: str = None):
        super(LinearModel, self).__init__()
        
        model_params = load_config(params_path, ["agent"]).get("agent", {})
        self.grid_size = grid_size if grid_size is not None else model_params.get("grid_size", 4)
        self.action_size = action_size if action_size is not None else model_params.get("action_size", 4)
        self.n_channels = n_channels if n_channels is not None else model_params.get("n_channels", 11)
        self.representation_kind = get_representation_kind(params_path, representation_kind)
        
        model_params = load_config(params_path, ["Linear_model"]).get("Linear_model", {})
        self.middle_channels = middle_channels if middle_channels is not None else model_params.get("middle_channels", (16, 32, 64))
        
        input_size = self.grid_size * self.grid_size
        
        if self.representation_kind == "one_hot":
            input_size *= self.n_channels
        
        self.fc1 = nn.Linear(input_size, self.middle_channels[0])
        self.bn1 = nn.BatchNorm1d(self.middle_channels[0])
        self.fc2 = nn.Linear(self.middle_channels[0], self.middle_channels[1])
        self.bn2 = nn.BatchNorm1d(self.middle_channels[1])
        self.fc3 = nn.Linear(self.middle_channels[1], self.middle_channels[2])
        self.bn3 = nn.BatchNorm1d(self.middle_channels[2])
        self.fc4 = nn.Linear(self.middle_channels[2], self.action_size)
        
        self.activation = nn.ReLU()
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x.reshape(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.bn1(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        # x = self.bn2(x)
        
        x = self.fc3(x)
        x = self.activation(x)
        # x = self.bn3(x)
        x = self.fc4(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_sizes: Tuple[int, int, int, int] = None):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, kernel_sizes[0], padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, kernel_sizes[1], padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, kernel_sizes[2], padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, kernel_sizes[3], padding='same')

    def forward(self, x):
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return th.cat((output1, output2, output3, output4), dim=1)

class Large_CNN(nn.Module):

    def __init__(self, params_path: Path, n_channels: int = None, action_size: int = None, grid_size: int = None, middle_channels: Tuple[int, int, int, int] = None, kernel_sizes: Tuple[int, int, int, int] = None):
        super(Large_CNN, self).__init__()
        
        config = load_config(params_path, ["agent"]).get("agent", {})
        self.n_channels = n_channels if n_channels is not None else config.get("n_channels", 3)
        self.grid_size = grid_size if grid_size is not None else config.get("grid_size", 2)
        self.action_size = action_size if action_size is not None else config.get("action_size", 2)
        
        config = load_config(params_path, ["Large_CNN"]).get("Large_CNN", {})
        self.middle_channels = middle_channels if middle_channels is not None else config.get("middle_channels", (12, 12, 12, 12))
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else config.get("kernel_sizes", (2, 2, 2, 2))
        
        self.representation_kind = get_representation_kind(params_path)
        input_size = 1
        
        if self.representation_kind == "one_hot":
            input_size *= self.n_channels
        
        self.conv1 = ConvBlock(input_size, self.middle_channels[0], self.kernel_sizes)
        self.conv2 = ConvBlock(self.middle_channels[0], self.middle_channels[1], self.kernel_sizes)
        self.conv3 = ConvBlock(self.middle_channels[1], self.middle_channels[2], self.kernel_sizes)
        self.dense1 = nn.Linear(self.middle_channels[2] * self.grid_size * self.grid_size, self.middle_channels[3])
        self.dense2 = nn.Linear(self.middle_channels[3], self.action_size)
    
    def forward(self, x: th.tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense2(x)