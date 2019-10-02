import os

import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.sklearn import SklearnDataset
from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda
from ..utils.utils import get_adj_sparse_matrix
from ..models.node2vec import Node2Vec


class BlogCatalog(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(self.get_raw_data_dir()):
            self.download_and_extract(
                "http://socialcomputing.asu.edu/uploads/1283153973/BlogCatalog-dataset.zip",
                self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_sklearn_wrapper(self, mode: str):
        return SklearnBlogCatalog(self, mode)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchBlogCatalog(self, mode)


class SklearnBlogCatalog(SklearnDataset):
    def __init__(self, builder: BlogCatalog, mode):
        super().__init__(builder)

        G = nx.Graph()
        raw_root = os.path.join(builder.get_raw_data_dir(), "BlogCatalog-dataset", "data")
        with open(os.path.join(raw_root, "nodes.csv")) as f:
            G.add_nodes_from(f.read().split())

        with open(os.path.join(raw_root, "edges.csv")) as f:
            G.add_edges_from([line.split(',') for line in f.read().split()])

        with open(os.path.join(raw_root, "groups.csv")) as f:
            labels = f.read().split()
            label2idx = {label: idx for idx, label in enumerate(labels)}

        for node in G.nodes:
            G.nodes[node]['labels'] = []
        with open(os.path.join(raw_root, "group-edges.csv")) as f:
            for line in f.read().split('\n'):
                node, group = line.split(',', 1)
                G.node[node]['labels'].append(group)

        for edge in G.edges:
            G[edge[0]][edge[1]]['weight'] = 1

        self.graph = G
        label_list = list(nx.get_node_attributes(G, "labels").values())
        label_list = [[label2idx[label] for label in ls] for ls in label_list]

        node_list = list(G)
        self.init_dataset(node_list, label_list)

        self.labels = labels
        self.node_labels = label_list

    @property
    def num_classes(self):
        return len(self.labels)


class PytorchBlogCatalog(PytorchDataset):
    def __init__(self, builder: BlogCatalog, mode):
        super().__init__(builder, mode)

        G = nx.Graph()
        raw_root = os.path.join(builder.get_raw_data_dir(), "BlogCatalog-dataset", "data")
        with open(os.path.join(raw_root, "nodes.csv")) as f:
            G.add_nodes_from(f.read().split())

        with open(os.path.join(raw_root, "edges.csv")) as f:
            G.add_edges_from([line.split(',') for line in f.read().split()])

        with open(os.path.join(raw_root, "groups.csv")) as f:
            labels = f.read().split()

        for node in G.nodes:
            G.nodes[node]['labels'] = []
        with open(os.path.join(raw_root, "group-edges.csv")) as f:
            for line in f.read().split('\n'):
                node, group = line.split(',', 1)
                G.node[node]['labels'].append(group)

        for edge in G.edges:
            G[edge[0]][edge[1]]['weight'] = 1

        label_list = list(nx.get_node_attributes(G, "labels").values())
        label2idx = {l: idx for idx, l in enumerate(labels)}
        label_list = [[ls[0]] for ls in label_list]
        label_list = [[label2idx[label] for label in ls] + [-1] * (len(labels) - len(ls)) for ls in label_list]

        # mlb = MultiLabelBinarizer(labels)
        # Y = mlb.fit_transform(label_list)
        Y = torch.LongTensor(label_list)
        X = torch.LongTensor(list(range(len(G.nodes))))

        self.graph = G
        emb = Node2Vec(builder.params)
        self.features = maybe_cuda(torch.FloatTensor(emb.fit_transform(G)))
        adj = nx.to_scipy_sparse_matrix(G, dtype=np.float32)
        self.adj = maybe_cuda(get_adj_sparse_matrix(adj))
        self._data = list(zip(X, Y))
        split_size = len(self._data) // 10
        if mode == "train":
            self._data = self._data[split_size:]
        elif mode == "test":
            self._data = self._data[:split_size]

    @property
    def num_labels(self):
        return len(self.labels)

    @property
    def input_size(self):
        return self.params.model.dimensions

    @property
    def num_classes(self):
        return 39

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))