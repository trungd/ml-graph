from typing import Dict
import random

from scipy.sparse import csgraph
from scipy.linalg import eigh
import networkx as nx
import numpy as np


def vertex_degree_signatures(graph: nx.Graph) -> Dict[any, float]:
    ret = {}
    for n in graph.nodes:
        ret[n] = graph.degree[n]
    return ret


def heat_kernel_signature(graph: nx.Graph, t) -> Dict[any, float]:
    nodes = list(graph.nodes.keys())
    A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    L = csgraph.laplacian(A, normed=True)
    eigen_values, eigen_vectors = eigh(L)
    ret = np.square(eigen_vectors).dot(np.diag(np.exp(-t * eigen_values))).sum(axis=1)
    return {node: ret[i] for i, node in enumerate(nodes)}


def assign_vertex_weight(graph: nx.Graph, func='degree', **kwargs):
    """
    Assign a weight to the attribute of each node in the graph
    :param graph: 
    :param func: 
        One of
        - random
        - degree: vertex degree
        - hks: heat kernel signature
    """
    if func == 'vertex_label':
        for n in graph.nodes:
            graph.nodes[n]['weight'] = graph.nodes[n]['label'] if 'label' in graph.nodes[n] else 0.
    elif func == 'random':
        for n in graph.nodes:
            graph.nodes[n]['weight'] = random.random()
    elif func == 'degree':
        for n in graph.nodes:
            graph.nodes[n]['weight'] = graph.degree[n]
    elif func == 'hks':
        t = kwargs['t']
        nodes = list(graph.nodes.keys())
        A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
        L = csgraph.laplacian(A, normed=True)
        eigen_values, eigen_vectors = eigh(L)
        ret = np.square(eigen_vectors).dot(np.diag(np.exp(-t * eigen_values))).sum(axis=1)
        for i, n in enumerate(nodes):
            graph.nodes[n]['weight'] = ret[i]
    else:
        raise ValueError