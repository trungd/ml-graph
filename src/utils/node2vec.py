import numpy as np
import networkx as nx
import random

from tqdm import tqdm


class Graph:
    def __init__(self, nx_G: nx.Graph, is_directed: bool, p: float, q: float):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length: int, start_node):
        """Simulate a random walk starting from start node."""
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_neighbors = sorted(self.G.neighbors(cur))
            if len(cur_neighbors) > 0:
                if len(walk) == 1:
                    walk.append(cur_neighbors[alias_draw(
                        self.alias_nodes[cur][0],
                        self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_neighbors[alias_draw(
                        self.alias_edges[(prev, cur)][0],
                        self.alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks: int, walk_length: int):
        """Repeatedly simulate random walks from each node."""
        walks = []
        nodes = list(self.G.nodes())
        for _ in tqdm(range(num_walks), desc="Walk iteration"):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """Get the alias edge setup lists for a given edge."""
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """Preprocessing of transition probabilities for guiding the random walks."""
        G = self.G

        alias_nodes = {}
        for node in tqdm(G.nodes(), desc="Calculating alias nodes"):
            probs = np.array([G[node][neighbor]['weight'] for neighbor in sorted(G.neighbors(node))])
            probs = probs / np.sum(probs)
            alias_nodes[node] = alias_setup(probs)

        alias_edges = {}
        if self.is_directed:
            for edge in tqdm(G.edges(), desc="Calculating alias edges"):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges(), desc="Calculating alias edges"):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def alias_setup(probs):
    """Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    kk = int(np.floor(np.random.rand() * len(J)))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
