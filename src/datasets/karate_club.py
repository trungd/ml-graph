import networkx as nx
from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder

from src.datasets.base.sklearn import SingleGraphDataset


class KarateClub(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def get_sklearn_wrapper(self, mode: str):
        return SklearnKarateClub(self)


class SklearnKarateClub(SingleGraphDataset):
    def __init__(self, builder):
        super().__init__(builder)
        G = nx.karate_club_graph()
        nx.set_edge_attributes(G, 1, "weight")
        self.graph = G
        self.X_train = list(range(len(G.nodes)))
        self.X_test = self.y_train = self.y_test = None

    def get_networkx_graph(self):
        return self.graph