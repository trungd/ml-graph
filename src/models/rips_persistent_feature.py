from typing import List

from ripser import ripser
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def rips_persistent_diagrams(graph: nx.Graph):
    V = len(graph.nodes)
    X = np.zeros([V, V], dtype=np.float)
    node2idx = {}
    for node in graph.nodes:
        if node not in node2idx:
            node2idx[node] = len(node2idx)
    for edge in graph.edges:
        u, v = edge
        u, v = node2idx[u], node2idx[v]
        X[u, v] = graph.edges[edge]["weight"]
    max_edge_weight = np.max(X)
    for u in range(V):
        for v in range(V):
            if X[u][v] == 0 and u != v:
                X[u][v] = max_edge_weight + 1
    dgms = ripser(X, maxdim=1, thresh=max_edge_weight, distance_matrix=True)['dgms']
    return dgms


class RipsPersistentFeature(TransformerMixin, BaseEstimator):
    def __init__():
        super().__init__()

    def fit_transform(self, graphs: List[nx.Graph], y=None, **fit_params):
        dgms = [rips_persistent_diagrams(graph) for graph in graphs]
        return dgms