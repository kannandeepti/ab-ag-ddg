""" Create an abstract PyTorch model class that can be used to train and evaluate models. """

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader



class ModelJointChain_MLP(nn.Module):

    def __init__(
        self,
        num_layers: int,  # total number of layers, including input and output
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for i in range(num_layers - 2):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class SingleChain_MLP(nn.Module):

    def __init__(
        self,
        num_layers: int,  # total number of layers, including input and output
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for i in range(num_layers - 2):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
