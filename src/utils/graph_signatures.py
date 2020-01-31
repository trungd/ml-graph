import random
from typing import Dict, Any, List

import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csgraph
from sklearn.cluster import KMeans


def vertex_degree_signatures(graph: nx.Graph) -> Dict[any, float]:
    ret = {}
    for n in graph.nodes:
        ret[n] = graph.degree[n]
    return ret


def heat_kernel_signature(graph: nx.Graph, t) -> Dict[Any, float]:
    nodes = list(graph.nodes.keys())
    A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    L = csgraph.laplacian(A, normed=True)
    eigen_values, eigen_vectors = eigh(L)
    ret = np.square(eigen_vectors).dot(np.diag(np.exp(-t * eigen_values))).sum(axis=1)
    return {node: ret[i] for i, node in enumerate(nodes)}


def return_probability_signature(graph: nx.Graph, S: int, normalize: bool = False) -> Dict[Any, List[float]]:
    nodes = list(graph.nodes.keys())
    A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    D = np.diag([graph.degree[n] for n in nodes])
    P1 = np.matmul(np.linalg.pinv(D), A)
    ret = [[] for _ in nodes]

    P = P1
    for s in range(S):
        if normalize:
            reg = sum([P[i, i] for i in range(len(nodes))])
            if reg == 0.0:
                reg = 1.0
        for i in range(len(nodes)):
            ret[i].append(P[i, i] / reg if normalize else P[i, i])
        P = np.matmul(P, P1)
    return {node: ret[i] for i, node in enumerate(nodes)}


def random_walk_probability_edge_signature(graph: nx.Graph, S: int) -> Dict[Any, float]:
    nodes = list(graph.nodes.keys())
    node2idx = {n: i for i, n in enumerate(nodes)}
    A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
    D = np.diag([graph.degree[n] for n in nodes])
    P1 = np.matmul(np.linalg.pinv(D), A)
    ret = {(u, v): [] for u, v in graph.edges}

    P = P1
    for s in range(S):
        for i in range(len(nodes)):
            for j in graph.neighbors(nodes[i]):
                if (nodes[i], j) in ret:
                    ret[(nodes[i], j)].append(max(P[i, node2idx[j]], P[node2idx[j], i]))
        P = np.matmul(P, P1)
    return ret


def assign_vertex_weight(graph: nx.Graph, func='degree', weights=None, **kwargs):
    """
    Assign a weight to the attribute of each node in the graph
    :param graph: 
    :param func: 
        One of
        - random
        - degree: vertex degree
        - hks: heat kernel signature
        - distance: distance to a source node (source must be provided in kwargs)
        - rpf: return probability
        - rpf_rank: ranking of return probability over all nodes
        - ns: neighbor size
    """
    if weights is not None:
        for n in graph.nodes:
            graph.nodes[n]['weight'] = weights[n]
    elif func == 'vertex_label':
        for n in graph.nodes:
            graph.nodes[n]['weight'] = graph.nodes[n]['label'] if 'label' in graph.nodes[n] else 0.
    elif func == 'random':
        for n in graph.nodes:
            graph.nodes[n]['weight'] = random.random()
    elif func == 'degree':
        for n in graph.nodes:
            graph.nodes[n]['weight'] = graph.degree[n]
    elif func == 'hks':
        t = kwargs['t'] if 't' in kwargs else 0.1
        nodes = list(graph.nodes.keys())
        A = nx.adjacency_matrix(graph, nodelist=nodes).todense()
        L = csgraph.laplacian(A, normed=True)
        eigen_values, eigen_vectors = eigh(L)
        ret = np.square(eigen_vectors).dot(np.diag(np.exp(-t * eigen_values))).sum(axis=1)
        for i, n in enumerate(nodes):
            graph.nodes[n]['weight'] = ret[i]
    elif func == 'ns':
        K = kwargs.get('K', 20)
        dist = nx.all_pairs_shortest_path_length(graph, cutoff=K)
        for n, ds in dist:
            ds = np.array(list(ds.values()))
            graph.nodes[n]['weight'] = [np.count_nonzero(ds <= k) / len(graph.nodes) for k in range(K)]
    elif func == 'rpf' or func == 'norm_rpf':
        ret = return_probability_signature(graph, kwargs.get('K', 20), normalize=func[:5] == 'norm_')
        for n in graph.nodes:
            graph.nodes[n]['weight'] = ret[n]
    elif func == 'rpf_nn':
        K = kwargs.get('K', 20)
        ret = return_probability_signature(graph, K, normalize=func[:5] == 'norm_')
        for k in range(K):
            nodes = graph.nodes
            ws = np.array([ret[n][k] for n in nodes]).reshape([-1, 1])
            kmeans = KMeans(n_clusters=10, random_state=0).fit(ws)
            ws = [kmeans.cluster_centers_[kmeans.labels_[i]] for i in range(len(ws))]
            for i, n in enumerate(nodes):
                ret[n][k] = ws[i]
        for n in graph.nodes:
            graph.nodes[n]['weight'] = ret[n]
    elif func == 'rpf_rank':
        K = kwargs.get('K', 20)
        ret = return_probability_signature(graph, K)
        for k in range(K):
            rank = sorted([(ret[n][k], n) for n in graph.nodes])
            for i, (_, n) in enumerate(rank):
                ret[n][k] = i
        for n in graph.nodes:
            graph.nodes[n]['weight'] = ret[n]
    elif func == 'non-rpf':
        ret = return_probability_signature(graph, 20)
        for n in graph.nodes:
            graph.nodes[n]['weight'] = [1 - r for r in ret[n]]
    elif func == 'rpf_lbl' or func == 'norm_rpf_lbl':
        ret = return_probability_signature(graph, 20, normalize=func[:5] == 'norm_')
        assert "num_labels" in kwargs
        for n in graph.nodes:
            graph.nodes[n]['weight'] = [(r + graph.nodes[n]['label']) / kwargs['num_labels'] for r in ret[n]]
    else:
        raise ValueError("%s is not supported." % func)
    
    
def assign_edge_weight(graph: nx.Graph, func: str):
    if func == 'random':
        for e in graph.edges:
            graph.edges[e]['weight'] = random.random()
    elif func == 'ollivier_ricci':
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
        for edge in graph.edges:
            graph.edges[edge]['weight'] = 1
        orc = OllivierRicci(graph, alpha=0.5)
        orc.compute_ricci_curvature()
        for edge in orc.G.edges:
            graph.edges[edge]['weight'] = orc.G.edges[edge]['ricciCurvature']
    elif func == 'forman_ricci':
        from GraphRicciCurvature.FormanRicci import FormanRicci
        frc = FormanRicci(graph)
        frc.compute_ricci_curvature()
        for edge in frc.G.edges:
            graph.edges[edge]['weight'] = 1 - frc.G.edges[edge]['formanCurvature']
    elif func == 'rpf':
        ret = random_walk_probability_edge_signature(graph, 20)
        for e in graph.edges:
            graph.edges[e]['weight'] = ret[e]
    else:
        raise ValueError