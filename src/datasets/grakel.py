import os
from typing import Dict, Union

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
        
        dataset = datasets.fetch_dataset(
            params.dataset.dataset_name,
            verbose=False,
            data_home=".",
            download_if_missing=True)
        G, y = dataset.data, dataset.target

        if self.configs.dataset_name in DATASETS_NO_NODE_LIST:
            logger.debug("Dataset size: %d", len(y))
            logger.debug("Average node count: %.2f",
                         sum([len(set([v[0] for v in g[0]]) | set([v[1] for v in g[1]])) for g in G]) / len(G))
            logger.debug("Average edge count: %.2f", sum([len(g[0]) for g in G]) / len(G))
        else:
            logger.debug("Dataset size: %d", len(y))
            logger.debug("Average node count: %.2f", sum([len(g[1]) for g in G]) / len(G))
            logger.debug("Average edge count: %.2f", sum([len(g[0]) for g in G]) / len(G))
        self.G = G
        self.y = y

    def get_working_dir(self):
        return os.path.join(super().get_working_dir(), self.configs.dataset_name)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_sklearn_wrapper(self, mode: str):
        return SklearnGrakelDataset(self)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchGrakelDataset(self, mode)


class SklearnGrakelDataset(MultiGraphsDataset):
    def __init__(self, builder):
        super().__init__(builder)
        self.G = builder.G
        self.y = builder.y
        self.extract_features()
        
    @property
    def num_classes(self):
        return len(set(self.y))

    def get_networkx_graphs(self):
        if self.configs.dataset_name in DATASETS_NO_NODE_LIST:
            return [get_networkx_graph(
                {v: 0 for v in set([v[0] for v in g[0]]) | set([v[1] for v in g[1]])},
                g[0],
            ) for g in self.G]
        else:
            return [get_networkx_graph(g[1], g[0], g[2]) for g in self.G]


class PytorchGrakelDataset(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        self.sklearn_dataset = SklearnGrakelDataset(builder)
        X = self.sklearn_dataset.X_train if mode == "train" else self.sklearn_dataset.X_test
        y = self.sklearn_dataset.y_train if mode == "train" else self.sklearn_dataset.y_test
        if isinstance(X, dict):
            X = [{key: X[key][i] for key in X} for i in range(len(y))]

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
                if cfg.transformers:
                    for key in cfg.transformers:
                        for transformer in cfg.transformers[key]:
                            X[key] = [pd_transform(transformer, np.array(dgm)).tolist() for dgm in X[key]]
                for key in X:
                    if key == "persistence_image":
                        X[key] = maybe_cuda(torch.FloatTensor(X[key]))
                    else:
                        X[key] = pad_sequence(X[key], 0., output_tensor=True)

                if len(X) == 1:
                    X = X[list(X.keys())[0]]
                ret[feat_name] = Batch(X=X, Y=maybe_cuda(torch.LongTensor([b[1] for b in batch])))
            else:
                raise ValueError

        if len(ret) == 1:
            return ret[list(ret.keys())[0]]
