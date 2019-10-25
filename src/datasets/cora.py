import os

import numpy as np
import scipy.sparse as sp
import torch

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda
from ..utils.utils import get_adj_sparse_matrix, normalize_sparse_matrix


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class Cora(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self.download_and_extract(
            "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
            self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCora(self, mode)


class PytorchCora(Dataset):
    def __init__(self, builder: DatasetBuilder, mode: str):
        super().__init__(builder, mode)
        self._load_data()

    def _load_data(self):
        path = os.path.join(self.builder.get_raw_data_dir(), "cora")
        idx_features_labels = np.genfromtxt(os.path.join(path, "cora.content"), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(os.path.join(path, "cora.cites"), dtype=np.int32)
        edges = np.array(
            [idx_map[node] for node in edges_unordered.flatten()],
            dtype=np.int32
        ).reshape(edges_unordered.shape)

        adj = sp.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(len(idx), len(idx)),
            dtype=np.float32)
        adj = get_adj_sparse_matrix(adj)

        features = normalize_sparse_matrix(features)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        self._data = list(zip(range(len(features)), labels))
        if self.mode == "train":
            self._data = self._data[:140]
        elif self.mode == "dev":
            self._data = self._data[140:500]
        elif self.mode == "test":
            self._data = self._data[500:]

        self.features = maybe_cuda(features)
        self.adj = maybe_cuda(adj)

    @property
    def num_classes(self):
        return 7

    @property
    def input_size(self):
        return 1433

    def collate_fn(self, batch) -> Batch:
        ret = super().collate_fn(batch)
        return Batch(X=maybe_cuda(ret[0]), Y=maybe_cuda(ret[1]))