import os
from typing import Dict, Union

import networkx as nx
import numpy as np
import torch
from comptopo import pd_transform
from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.torch.utils.variable_length_tensor import pad_sequence
from dlex.utils import logger
from grakel import datasets

from .base.sklearn import MultiGraphsDataset
from ..utils.utils import get_networkx_graph

DATASETS_NO_NODE_LIST = ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'COLLAB']


class Grakel(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

        self._graphs = None
        self._labels = None
        self._networkx_graphs = None
        self._sklearn_dataset = None

    def load_dataset(self):
        dataset = datasets.fetch_dataset(
            self.configs.dataset_name,
            verbose=False,
            data_home=self.get_raw_data_dir(),
            download_if_missing=True
        )
        logger.info("Dataset loaded.")
        G = dataset.data
        y = dataset.target

        if self.configs.dataset_name in DATASETS_NO_NODE_LIST:
            logger.info(" - ".join([
                "No. graphs: %d" % len(y),
                "Avg. no. nodes: %.2f" % (sum([len(set([v[0] for v in g[0]]) | set([v[1] for v in g[1]])) for g in G]) / len(G)),
                "Avg. no. edges: %.2f" % (sum([len(g[0]) for g in G]) / len(G))
            ]))
        else:
            logger.info(" - ".join([
                "No. graphs: %d" % len(y),
                "Avg. no. nodes: %.2f" % (sum([len(g[1]) for g in G]) / len(G)),
                "Avg. no. edges: %.2f" % (sum([len(g[0]) for g in G]) / len(G))
            ]))

        return G, y

    @property
    def graphs(self):
        if not self._graphs:
            logger.info(f"Loading dataset {self.configs.dataset_name}...")
            self._graphs, self._labels = self.load_dataset()
        return self._graphs

    @property
    def labels(self):
        filepath = os.path.join(self.get_processed_data_dir(), self.configs.dataset_name + "_labels.txt")
        if self.configs.reuse and os.path.exists(filepath):
            if self._labels is None:
                with open(filepath, "r") as f:
                    logger.info(f"Loading labels from {filepath}...")
                    self._labels = [int(x) for x in f.read().split('\n') if x != ""]
        else:
            if self._labels is None:
                self._graphs, self._labels = self.load_dataset()
            logger.info(f"Writing labels to {filepath}...")
            with open(filepath, "w") as f:
                f.write("\n".join([str(s) for s in self._labels]))
        return self._labels

    def get_working_dir(self):
        return os.path.join(super().get_working_dir(), self.configs.dataset_name)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_sklearn_wrapper(self, mode: str = None):
        return SklearnGrakelDataset(self)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchGrakelDataset(self, mode)

    @property
    def networkx_graphs(self):
        if self._networkx_graphs:
            return self._networkx_graphs

        if self.configs.dataset_name in DATASETS_NO_NODE_LIST:
            self._networkx_graphs = [get_networkx_graph(
                {v: 0 for v in set([v[0] for v in g[0]]) | set([v[1] for v in g[1]])},
                g[0],
            ) for g in self.graphs]
        else:
            self._networkx_graphs = [get_networkx_graph(g[1], g[0], g[2]) for g in self.graphs]
        return self._networkx_graphs

    @property
    def sklearn_dataset(self):
        if not self._sklearn_dataset:
            self._sklearn_dataset = self.get_sklearn_wrapper()

        return self._sklearn_dataset


class SklearnGrakelDataset(MultiGraphsDataset):
    def __init__(self, builder):
        super().__init__(builder)
        self.extract_features()
        self.stats()
        self._num_node_labels = None

    @property
    def graphs(self):
        return self.builder.graphs

    @property
    def labels(self):
        return self.builder.labels

    @property
    def num_node_labels(self):
        if not self._num_node_labels:
            graphs = self.networkx_graphs
            node_labels = set.union(*[set(nx.get_node_attributes(g, "label").values()) for g in graphs])
            self._num_node_labels = len(node_labels)
        return self._num_node_labels
        
    @property
    def num_classes(self):
        return len(set(self.labels))

    def stats(self):
        pass

    @property
    def networkx_graphs(self):
        return self.builder.networkx_graphs


class PytorchGrakelDataset(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        self.sklearn_dataset = builder.sklearn_dataset
        X = self.sklearn_dataset.X_train if mode == "train" else self.sklearn_dataset.X_test
        y = self.sklearn_dataset.y_train if mode == "train" else self.sklearn_dataset.y_test

        if isinstance(X, dict):
            X = [{key: X[key][i] for key in X} for i in range(len(y))]

        logger.info("No. persistence diagrams: %d", sum(sum(len(p) for p in x.get('multi_1.2.3.4')) for x in X))
        logger.info("No. reduced persistence diagrams: %d", sum(sum(len(p) for p in x.get('freq_multi_1.2.3.4', **self.params.dataset.graph_features.persistence_diagram.to_dict())) for x in X))

        for feat_name in self.sklearn_dataset.graph_features:
            if feat_name == "persistence_diagram":
                cfg = self.params.dataset.graph_features.persistence_diagram
                keys = cfg.keys if type(cfg.keys) == list else [cfg.key]

                # weights = self.sklearn_dataset._get_vertex_weights(cfg.signature)

                X = [{key: x.get(key, **(cfg.to_dict())) for key in keys} for x in X]

                if cfg.transformers:
                    for key in cfg.transformers:
                        for transformer in cfg.transformers[key]:
                            X[key] = [pd_transform(transformer, np.array(x[key])).tolist() for x in X]

        if -1 in y:  # [-1, 1] -> [0, 1]
            y = [i if i != -1 else 0 for i in y]
        elif 0 not in y:  # [1, 2, 3] -> [0, 1, 2]
            y = [i - 1 for i in y]

        self._data = list(zip(X, y))
        logger.debug("%s size: %d" % (mode, len(self._data)))
        
    @property
    def num_classes(self):
        return self.sklearn_dataset.num_classes

    @property
    def output_shape(self):
        return [self.num_classes]

    @property
    def input_shape(self):
        return None
        # return [self.sklearn_dataset.input_size]

    def collate_fn(self, batch) -> Union[Batch, Dict[str, Batch]]:
        #if self.input_shape is not None:
        #    ret = super().collate_fn(batch)
        #    return Batch(
        #        X=maybe_cuda(ret[0].float()),
        #        Y=maybe_cuda(ret[1]))
        #else:
        ret = {}
        for feat_name in self.sklearn_dataset.graph_features:
            if feat_name == "persistence_diagram":
                cfg = self.params.dataset.graph_features.persistence_diagram
                keys = cfg.keys if type(cfg.keys) == list else [cfg.key]
                X = {key: [b[0][key] for b in batch] for key in keys}
                for key in keys:
                    if key == "persistence_image":
                        X[key] = maybe_cuda(torch.FloatTensor(X[key]))
                    else:
                        X[key] = pad_sequence(X[key], 0., output_tensor=True, dim=3)

                if len(X) == 1:
                    X = X[list(X.keys())[0]]
                ret[feat_name] = Batch(X=X, Y=maybe_cuda(torch.LongTensor([b[1] for b in batch])))
            else:
                raise ValueError

        if len(ret) == 1:
            return ret[list(ret.keys())[0]]
