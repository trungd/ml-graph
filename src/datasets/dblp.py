import os

import networkx as nx
import numpy as np
import torch

from dlex.datasets import DatasetBuilder
from dlex.datasets.torch import Dataset
from dlex.torch.utils.ops_utils import maybe_cuda


class DBLP(DatasetBuilder):
    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self._download_and_extract("https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz")
        self._download_and_extract("https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz")
        self._download_and_extract("https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz")

    def get_pytorch_wrapper(self, mode: str):
        return PytorchDBLP(self, mode)


class PytorchDBLP(Dataset):
    def __init__(self, builder: DatasetBuilder, mode: str):
        super().__init__(builder, mode)

        with open(os.path.join(builder.get_raw_data_dir(), "com-dblp.ungraph.txt")) as f:
            lines = f.read().split("\n")[4:]

        G = nx.Graph()
        G.add_edges_from([[int(x) for x in line.split('\t')] for line in lines if line != ""], weight=1.)

        self.g = G
        self.edges_raw = G.edges(data=True)
        self.nodes_raw = G.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    @property
    def num_nodes(self):
        return self.g.number_of_nodes()

    @property
    def num_edges(self):
        return self.g.number_of_edges()
        
    def __getitem__(self, i):
        if self.configs.edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_edges, p=self.edge_distribution)
        elif self.configs.edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(1)
        elif self.configs.edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_edges, size=1)
        u_i = []
        u_j = []
        label = []
        
        edge = self.edges[edge_batch_index]
        if self.g.__class__ == nx.Graph:
            if np.random.rand() > 0.5:      # important: second-order proximity is for directed edge
                edge = (edge[1], edge[0])
        u_i.append(edge[0])
        u_j.append(edge[1])
        label.append(1)
        for i in range(self.configs.K):
            while True:
                if self.configs.node_sampling == 'numpy':
                    negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                elif self.configs.node_sampling == 'atlas':
                    negative_node = self.node_sampling.sampling()
                elif self.configs.node_sampling == 'uniform':
                    negative_node = np.random.randint(0, self.num_of_nodes)
                if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                    break
            u_i.append(edge[0])
            u_j.append(negative_node)
            label.append(-1)
        return maybe_cuda(torch.tensor(u_i)), maybe_cuda(torch.tensor(u_j)), maybe_cuda(torch.tensor(label))
    
    def __len__(self):
        return self.configs.num_batches

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}

    def collate_fn(self, batch):
        return super().collate_fn(batch)

    def evaluate(self, y_pred, y_ref, metric: str, output_path: str):
        true_list = list()
        prediction_list = list()
        for edge in true_edges:
            true_list.append(1)
            prediction_list.append(get_score(embs, edge[0], edge[1]))

        for edge in false_edges:
            true_list.append(0)
            prediction_list.append(get_score(embs, edge[0], edge[1]))

        sorted_pred = prediction_list[:]
        sorted_pred.sort()
        threshold = sorted_pred[-len(true_edges)]

        y_pred = np.zeros(len(prediction_list), dtype=np.int32)
        for i in range(len(prediction_list)):
            if prediction_list[i] >= threshold:
                y_pred[i] = 1

        y_true = np.array(true_list)
        y_scores = np.array(prediction_list)
        ps, rs, _ = precision_recall_curve(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)


class AliasSampling:
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

