import torch.nn as nn
import torch

from dlex import MainConfig
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.models import ClassificationModel


class DeepSet(ClassificationModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)
        self.linear = nn.Linear(2, 64)
        self.proj = nn.Linear(64, dataset.num_classes)

    def forward(self, batch: Batch):
        x = self.linear(batch.X)
        x = torch.sum(x, dim=-2)
        x = self.proj(x)
        return x