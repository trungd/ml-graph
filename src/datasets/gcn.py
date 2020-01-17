# Load pre-processed dataset from https://github.com/tkipf/gcn/tree/master/gcn/data
# Reference: https://github.com/tkipf/gcn/blob/master/gcn/utils.py

import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import networkx as nx
from dlex.datasets.builder import DatasetBuilder
from dlex.configs import AttrDict
from dlex.datasets.torch import Dataset


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


class GCNDataset(DatasetBuilder):
    def __init__(self, params: AttrDict, dataset_name: str):
        super().__init__(params)
        self.dataset_name = dataset_name

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        for name in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']:
            self.download(
                f"https://github.com/tkipf/gcn/blob/master/gcn/data/ind.{self.dataset_name}.{name}?raw=true",
                filename=f"ind.{self.dataset_name}.{name}")

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchDataset(self, mode)


class Cora(GCNDataset):
    def __init__(self, params: AttrDict):
        super().__init__(params, "cora")


class PubMed(GCNDataset):
    def __init__(self, params: AttrDict):
        super().__init__(params, "pubmed")


class CiteSeer(GCNDataset):
    def __init__(self, params: AttrDict):
        super().__init__(params, "citeseer")


class PytorchDataset(Dataset):
    def __init__(self, builder: GCNDataset, mode: str):
        # TODO: torch compatible implementation
        super().__init__(builder, mode)

        objects = []
        for name in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(os.path.join(
                    builder.get_raw_data_dir(),
                    "ind.{}.{}".format(builder.dataset_name, name)), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pickle.load(f, encoding='latin1'))
                else:
                    objects.append(pickle.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(os.path.join(builder.get_raw_data_dir(), f"ind.{builder.dataset_name}.test.index"))
        test_idx_range = np.sort(test_idx_reorder)

        if builder.dataset_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        self.adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

    @property
    def num_classes(self):
        return 7

    @property
    def input_dim(self):
        return 1433