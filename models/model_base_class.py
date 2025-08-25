""" Create an abstract PyTorch model class that can be used to train and evaluate models. """

from abc import ABC, abstractmethod
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import esm
from esm.modules import TransformerLayer, ESM1bLayerNorm


class ModelJointChain_MLP(nn.Module):

    def __init__(
        self,
        num_layers: int,  # total number of layers, including input and output
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.layers = nn.ModuleList([nn.Dropout(dropout_rate)])
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 2):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


class Residue_Transformer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,  # frozen PLM embedding dimension
        hidden_dim: int = 128,  # dimension after convolutional layer
        attention_heads: int = 8,
    ):
        super().__init__()

        # convolutional layer that takes in (sequence_length, embedding_dim) and outputs (sequence_length, hidden_dim)
        self.conv1d = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=1, padding=0)

        # ESM transformer layer
        self.transformer = TransformerLayer(
            hidden_dim,  # embedding dimension of transformer
            4 * hidden_dim,  # ffn embed dimension
            attention_heads,
            add_bias_kv=False,
            use_esm1b_layer_norm=True,
            use_rotary_embeddings=True,
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(hidden_dim)
        self.linear_ff = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # reshape for convolutional layer
        x = x.permute(0, 2, 1)  # (B, L, d) -> (B, d, L)
        x = self.conv1d(x)  # (B, m, L)

        # (B, m, L) => (L, B, m)
        x = x.permute(2, 0, 1)
        if not padding_mask.any():
            padding_mask = None
        x, attn = self.transformer(x, self_attn_mask=padding_mask)
        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (L, B, m) => (B, L, m)

        # global average pooling to get (B, m)
        x = x.mean(dim=1)

        # linear feed forward layer to get (B, 1)
        x = self.linear_ff(x)

        return x.squeeze(-1)
