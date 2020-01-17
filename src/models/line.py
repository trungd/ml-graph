from dlex.torch import BaseModel, Batch
import torch
from torch import nn
from torch.nn import functional as F


class LINE(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.n1 = dataset.num_nodes
        self.dim = self.configs.dim
        self.order = self.configs.order

        self.nodes_embed = nn.Embedding(self.n1, self.dim)
        self.nodes_embed.weight.data.uniform_(-1, 1)

        if self.order == 2:
            self.context_nodes_embed = nn.Embedding(self.n1, self.dim)
            self.context_nodes_embed.weight.data.uniform_(-1, 1)

    def forward(self, batch: Batch):
        """
        :param source_node: list of [i,i,i,i,i, ...] of source nodes: 
            each source node repeat K + 1 time: one for target node, K times for K negative nodes
        :param target_node: list of [j,j1,j2,..,jK, ...] of target nodes: 
            j is target node, j1 -> jK is negative nodes
        :param label: FloatTensor([1, -1, -1, -1, -1, -1, 1, ....]) label 
            to indicate which is target node, which is negative nodes
        :return:
        """

        source_node, target_node, label = batch
        source_embed = self.nodes_embed(source_node)

        if self.order == 1:
            target_embed = self.nodes_embed(target_node)
        elif self.order == 2:
            target_embed = self.context_nodes_embed(target_node)
        else:
            raise ValueError

        inner_product = torch.sum(source_embed * target_embed, dim=-1)
        return inner_product

    def get_loss(self, batch, output):
        pos_neg = batch[-1].float() * output
        loss = F.logsigmoid(pos_neg)
        return -torch.mean(loss)

    def infer(self, batch):
        source_node, _, _ = batch
        return self.nodes_embed(source_node)
