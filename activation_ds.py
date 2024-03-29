import itertools
import json
from typing import Optional
from attrs import define
from abc import ABC, abstractmethod
from transformers import GPT2Tokenizer, PreTrainedTokenizerBase
import os
from pathlib import Path

import torch

from utils import flatten_activations


class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        super().__init__()
        self.x_data = x
        self.y_data = y

    @classmethod
    def from_data(cls, data, layers: list, device: str = "cpu"):
        data_x = []
        data_y = []
        for i, category in enumerate(data.keys()):
            x = torch.cat(tuple(flatten_activations(data[category][layer]) for layer in layers))
            data_x.append(x)
            y = torch.zeros(x.shape[0], len(data.keys()), dtype=torch.int32)
            y[:, i] = 1
            data_y.append(y)
        return cls(torch.cat(data_x).to(device), torch.cat(data_y).to(device))

    def project(self, dir: torch.Tensor):
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        new_x_data = self.x_data - torch.outer((self.x_data @ dir_norm), dir_norm)
        return ActivationsDataset(new_x_data, self.y_data)

    def project_(self, dir: torch.Tensor):
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        self.x_data -= torch.outer((self.x_data @ dir_norm), dir_norm)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx, :]
        sample = (x, y)
        return sample
