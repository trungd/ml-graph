import torch

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.datatypes import VariableLengthInputBatch, VariableLengthTensor
from dlex.utils import logger
from grakel import datasets

from .base.sklearn import MultiGraphsDataset
from ..utils.utils import get_networkx_graph


class Grakel(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

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
        dataset = datasets.fetch_dataset(builder.params.dataset.dataset_name, verbose=False)
        G, y = dataset.data, dataset.target
        if self.configs.dataset_name in ['REDDIT-MULTI-5K']:
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
        self.extract_features()
        # self.init_dataset(G, y)
        
    @property
    def num_classes(self):
        return 6

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
        self._data = list(zip(self.sklearn_dataset.X_train, self.sklearn_dataset.y_train))
        
    @property
    def num_classes(self):
        return self.sklearn_dataset.num_classes

    def collate_fn(self, batch):
        X_dim_0 = [b[0]['dim_0'].tolist() for b in batch]
        X_dim_0_essential = [b[0]['dim_0_essential'].tolist() for b in batch]
        X_dim_1_essential = [b[0]['dim_1_essential'].tolist() for b in batch]
        return Batch(
            X=(
                VariableLengthTensor(X_dim_0, [0., 0.]),
                VariableLengthTensor(X_dim_0_essential, [0., 0.]),
                VariableLengthTensor(X_dim_1_essential, [0., 0.])
            ),
            Y=torch.LongTensor([b[1] for b in batch])
        )
        # return super().collate_fn(batch)
