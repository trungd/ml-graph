from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
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
        return GrakelDataset(self)


class GrakelDataset(MultiGraphsDataset):
    def __init__(self, builder):
        super().__init__(builder)

        dataset = datasets.fetch_dataset(builder.params.dataset.dataset_name, verbose=False)
        G, y = dataset.data, dataset.target
        logger.info("Dataset size: %d", len(y))
        logger.info("Average node count: %.2f", sum([len(g[1]) for g in G]) / len(G))
        logger.info("Average edge count: %.2f", sum([len(g[2]) for g in G]) / len(G))
        self.G = G
        self.y = y
        self.extract_features()
        # self.init_dataset(G, y)

    def get_networkx_graphs(self):
        return [get_networkx_graph(g[1], g[0], g[2]) for g in self.G]