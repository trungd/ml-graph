import networkx as nx

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.sklearn import SklearnDataset
from ..utils.node2vec import Graph


class Dummy(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def get_sklearn_wrapper(self, mode: str):
        return SklearnDummy(self)


class SklearnDummy(SklearnDataset):
    def __init__(self, builder):
        super().__init__(builder)
        edge_data = "1 32\n1 22\n1 20\n1 18\n1 14\n1 13\n1 12\n1 11\n1 9\n1 8\n1 7\n1 6\n1 5\n1 4\n1 3\n1 2\n2 31\n2 " \
                    "22\n2 20\n2 18\n2 14\n2 8\n2 4\n2 3\n3 14\n3 9\n3 10\n3 33\n3 29\n3 28\n3 8\n3 4\n4 14\n4 13\n4 " \
                    "8\n5 11\n5 7\n6 17\n6 11\n6 7\n7 17\n9 34\n9 33\n9 33\n10 34\n14 34\n15 34\n15 33\n16 34\n16 " \
                    "33\n19 34\n19 33\n20 34\n21 34\n21 33\n23 34\n23 33\n24 30\n24 34\n24 33\n24 28\n24 26\n25 " \
                    "32\n25 28\n25 26\n26 32\n27 34\n27 30\n28 34\n29 34\n29 32\n30 34\n30 33\n31 34\n31 33\n32 " \
                    "34\n32 33\n33 34"
        edge_list = [[s for s in line.split(' ')] for line in edge_data.split('\n')]
        G = nx.Graph()
        G.add_nodes_from(map(str, range(1, 35)))
        G.add_edges_from(edge_list)
        for edge in edge_list:
            G[edge[0]][edge[1]]['weight'] = 1

        cfg = builder.params.model
        if not cfg.directed:
            G = G.to_undirected()
        self.graph = G
        self.X_train = list(range(len(G.nodes)))
        self.X_test = self.y_train = self.y_test = None