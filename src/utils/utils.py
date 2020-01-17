from typing import Dict, Tuple, List

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch


def get_networkx_graph(
        node_labels: Dict[int, any],
        edges: List[Tuple[int, int]],
        edge_labels: Dict[Tuple[int, int], any] = None):
    g = nx.Graph()
    g.add_nodes_from(zip(node_labels.keys(), [dict(label=label) for label in node_labels.values()]))
    if edge_labels is not None and len(edges) == len(edge_labels):
        g.add_edges_from([(edge[0], edge[1], dict(label=edge_labels[edge])) for edge in edges])
    else:
        g.add_edges_from(edges)
    return g


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_adj_sparse_matrix(adj) -> torch.FloatTensor:
    # adj = sp.coo_matrix(
    #    (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
    #    shape=(num_nodes, num_nodes),
    #    dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_sparse_matrix(adj + sp.eye(adj.shape[0]))

    # convert to torch sparse tensor
    sparse_mx = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)