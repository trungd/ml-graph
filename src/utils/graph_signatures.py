from typing import Dict

from scipy.sparse import csgraph
from scipy.linalg import eigh
import networkx as nx
import numpy as np


def vertex_degree_signatures(graph: nx.Graph) -> Dict[any, float]:
    ret = {}
    for n in graph.nodes:
        ret[n] = graph.degree[n]
    return ret


def heat_kernel_signature(graph: nx.Graph, t):
    nodes = list(graph.nodes.keys())
    A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    L = csgraph.laplacian(A, normed=True)
    eigen_values, eigen_vectors = eigh(L)
    ret = np.square(eigen_vectors).dot(np.diag(np.exp(-t * eigen_values))).sum(axis=1)
    return {node: ret[i] for i, node in enumerate(nodes)}
