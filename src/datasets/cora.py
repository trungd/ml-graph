import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from comptopo import pd_transform
from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.torch.utils.variable_length_tensor import pad_sequence
from numpy.linalg import norm

from .base import SklearnSingleGraphDataset, PytorchSingleGraphDataset
from ..utils.utils import get_adj_sparse_matrix, normalize_sparse_matrix


def encode_one_hot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_one_hot = np.array(
        list(map(classes_dict.get, labels)),
        dtype=np.int32)
    return labels_one_hot


class Cora(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self._download_and_extract(
            "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
            self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCora(self, mode)

    def get_sklearn_wrapper(self, mode: str):
        return SklearnCora(self, mode)


class SklearnCora(SklearnSingleGraphDataset):
    def __init__(self, builder):
        super().__init__(builder)
        self._load_data()

    def get_networkx_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from((node, dict(
            label=self.labels[node]
        )) for node in range(len(self.labels)))
        g.add_edges_from([(u, v) for u, v in self.edges])
        for u, v in g.edges:
            g.edges[(u, v)]['weight'] = norm(self.node_embeddings[u] - self.node_embeddings[v])
        return g

    def _load_data(self):
        path = os.path.join(self.builder.get_raw_data_dir(), "cora")
        idx_features_labels = np.genfromtxt(os.path.join(path, "cora.content"), dtype=np.dtype(str))
        self.labels = idx_features_labels[:, -1]
        self.node_embeddings = idx_features_labels[:, 1:-1].astype(float)

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        edges_unordered = np.genfromtxt(os.path.join(path, "cora.cites"), dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(
            [idx_map[node] for node in edges_unordered.flatten()],
            dtype=np.int32
        ).reshape(edges_unordered.shape)

        self.nodes = list(range(len(idx)))
        self.edges = edges

        self.adj_matrix = sp.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(len(idx), len(idx)),
            dtype=np.float32)

        # Extract features
        node_features = [dict(embedding=self.node_embeddings[node]) for node in self.nodes]
        if self.builder.params.dataset.graph_features:
            for feat in self.builder.params.dataset.graph_features:
                if feat == 'embedding':
                    for node in self.nodes:
                        node_features[node][feat] = self.node_embeddings[node]
                if feat == 'persistence_diagram':
                    dgms = self._get_persistence_diagrams()
                    for node in self.nodes:
                        node_features[node][feat] = dgms[node]
        self.node_features = node_features

    @property
    def num_classes(self):
        return 7

    @property
    def input_dim(self):
        return 1433


class PytorchCora(PytorchSingleGraphDataset):
    def __init__(self, builder: DatasetBuilder, mode: str):
        super().__init__(builder, mode)
        sk_dataset = SklearnCora(builder)
        self.sk_dataset = sk_dataset

        adj_matrix = get_adj_sparse_matrix(sk_dataset.adj_matrix)

        node_embeddings = sp.csr_matrix(sk_dataset.node_embeddings, dtype=np.float32)
        node_embeddings = normalize_sparse_matrix(node_embeddings)
        node_embeddings = torch.FloatTensor(np.array(node_embeddings.todense()))

        labels = torch.LongTensor(np.where(encode_one_hot(sk_dataset.labels))[1])

        self._data = list(zip(range(len(labels)), labels))

        # dataset splits as in Yang et al. (2016)
        yang_split = True
        if yang_split:
            if self.mode == "train":
                self._data = self._data[:140]
            elif self.mode == "dev":
                self._data = self._data[200:500]
            elif self.mode == "test":
                self._data = self._data[500:1500]
        else:
            if self.mode == "train":
                self._data = self._data[:1500]
            elif self.mode == "dev":
                self._data = self._data[1500:]
            elif self.mode == "test":
                self._data = self._data[1500:]

        self.node_features = sk_dataset.node_features
        self.node_embeddings = maybe_cuda(node_embeddings)
        self.adj_matrix = maybe_cuda(adj_matrix)

    @property
    def num_classes(self):
        return self.sk_dataset.num_classes

    @property
    def input_dim(self):
        if self.configs.graph_features:
            return None
        else:
            return self.sk_dataset.input_dim

    @property
    def embedding_dim(self):
        return self.node_embeddings.shape[1]

    def collate_fn(self, batch) -> Batch:
        if self.input_dim is not None:
            ret = super().collate_fn(batch)
            return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))
        elif self.configs.graph_features:
            X = dict()
            X_len = dict()
            for feat_name in self.configs.graph_features:
                if feat_name == 'embedding':
                    X[feat_name] = maybe_cuda(torch.FloatTensor([self.node_features[b[0]]['embedding'] for b in batch]))
                if feat_name == 'persistence_diagram':
                    pds = [self.sk_dataset.node_features[b[0]]['persistence_diagram'] for b in batch]
                    _X = {key: [pd[key] for pd in pds] for key in self.configs.graph_features.persistence_diagram.diagrams}
                    dim = {'h0_non_essential': 2, 'h0_essential': 1, 'h1_essential': 1}
                    for key in _X:
                        for transformer in self.configs.graph_features.persistence_diagram.transformers[key]:
                            _X[key] = [pd_transform(transformer, np.array(dgm)).tolist() for dgm in _X[key]]

                    for key, x in _X.items():
                        X[feat_name][key] = pad_sequence(x, [0.] if dim[key] == 1 else [0., 0.])

            return Batch(
                X=X,
                Y=maybe_cuda(torch.LongTensor([b[1] for b in batch]))
            )
