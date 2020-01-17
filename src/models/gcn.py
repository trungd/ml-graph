import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.configs import MainConfig
from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from torch.nn.parameter import Parameter

from ..datasets import PytorchSingleGraphDataset


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.FloatTensor(input_size, output_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, adj):
        support = torch.mm(X, self.weight)
        output = torch.spmm(adj, support)
        return output if self.bias is None else output + self.bias


class GCN(BaseModel):
    def __init__(self, params: MainConfig, dataset: PytorchSingleGraphDataset):
        super().__init__(params, dataset)
        self.dataset = dataset

        cfg = params.model
        hidden_dim = cfg.hidden_dim if isinstance(cfg.hidden_dim, list) else cfg.hidden_dim
        gc_layers = []
        for n_in, n_out in zip(
                [dataset.input_dim] + hidden_dim,
                hidden_dim + [dataset.num_classes]):
            gc_layers.append(GraphConvolution(n_in, n_out))
        self.gc_layers = nn.ModuleList(gc_layers)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        X = self.dataset.node_embeddings
        for layer in self.gc_layers:
            X = layer(X, self.dataset.adj_matrix)
            if layer != self.gc_layers[-1]:
                X = F.relu(X)
                X = F.dropout(X, self.configs.dropout)
        return F.log_softmax(X, dim=1)

    def get_loss(self, batch, output):
        indices = batch.X.tolist()
        return self.criterion(output[indices], batch.Y)

    def infer(self, batch: Batch):
        logits = self.forward(batch)
        indices = batch.X.tolist()
        return torch.max(logits[indices], 1)[1].tolist(), batch.Y.tolist(), logits[indices], None