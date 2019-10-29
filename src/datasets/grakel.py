import os

import torch

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.datatypes import VariableLengthInputBatch, VariableLengthTensor
from dlex.utils import logger
from dlex.torch.utils.ops_utils import maybe_cuda
from grakel import datasets

from .base.sklearn import MultiGraphsDataset
from ..utils.utils import get_networkx_graph


class Grakel(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)
        
        dataset = datasets.fetch_dataset(params.dataset.dataset_name, verbose=False)
        G, y = dataset.data, dataset.target
        if self.configs.dataset_name in ['REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
            logger.info("Dataset size: %d", len(y))
            logger.info("Average node count: %.2f",
                        sum([len(set([v[0] for v in g[0]]) | set([v[1] for v in g[1]])) for g in G]) / len(G))
            logger.info("Average edge count: %.2f", sum([len(g[0]) for g in G]) / len(G))
        else:
            logger.info("Dataset size: %d", len(y))
            logger.info("Average node count: %.2f", sum([len(g[1]) for g in G]) / len(G))
            logger.info("Average edge count: %.2f", sum([len(g[2]) for g in G]) / len(G))
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
        # self.init_dataset(G, y)
        
    @property
    def num_classes(self):
        return len(set(self.y))

    def get_networkx_graphs(self):
        if self.configs.dataset_name in ['REDDIT-MULTI-5K']:
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
        y = [i - 1 for i in y]
        self._data = list(zip(X, y))
        logger.info("%s size: %d" % (mode, len(self._data)))
        
    @property
    def num_classes(self):
        return self.sklearn_dataset.num_classes

    @property
    def output_shape(self):
        return [self.num_classes]

    @property
    def input_shape(self):
        # return None
        return [self.sklearn_dataset.input_size]

    def collate_fn(self, batch):
        if self.input_shape is not None:
            ret = super().collate_fn(batch)
            return Batch(
                X=maybe_cuda(ret[0].float()),
                Y=maybe_cuda(ret[1]))
        else:
            X_dim_0 = [b[0][0].not_essential_points for b in batch]
            X_dim_0_essential = [b[0][0].essential_points for b in batch]
            X_dim_1_essential = [b[0][1].essential_points for b in batch]
            return Batch(
                X=(
                    VariableLengthTensor(X_dim_0, [0., 0.]).cuda(),
                    VariableLengthTensor(X_dim_0_essential, [0., 0.]).cuda(),
                    VariableLengthTensor(X_dim_1_essential, [0., 0.]).cuda()
                ),
                Y=maybe_cuda(torch.LongTensor([b[1] for b in batch]))
            )
