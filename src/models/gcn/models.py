import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from dlex.configs import AttrDict
from dlex.torch import Batch
from dlex.torch.models.base import BaseModel


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
    def __init__(self, params: AttrDict, dataset):
        super().__init__(params, dataset)

        cfg = params.model
        self.gc1 = GraphConvolution(dataset.input_size, cfg.hidden_size)
        self.gc2 = GraphConvolution(cfg.hidden_size, dataset.num_classes)
        self.dropout = params.model.dropout
        self.criterion = nn.MultiLabelMarginLoss()

    def forward(self, batch: Batch):
        X = F.relu(self.gc1(self.dataset.features, self.dataset.adj))
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.gc2(X, self.dataset.adj)
        return F.log_softmax(X, dim=1)

    def get_loss(self, batch, output):
        indices = batch.X.tolist()
        return self.criterion(output[indices], batch.Y)

    def infer(self, batch: Batch):
        logits = self.forward(batch)
        indices = batch.X.tolist()
        return torch.max(logits[indices], 1)[1].tolist(), [ls[0] for ls in batch.Y.tolist()], logits[indices], None