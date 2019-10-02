from typing import Dict, List, Tuple

import networkx as nx
from grakel import datasets

from utils.kernels import weisfeiler_lehman_subtree_features


mutag = datasets.fetch_dataset("MUTAG", verbose=False)
G, y = mutag.data, mutag.target
G = [get_networkx_graph(g[1], g[0], g[2]) for g in G]






g1 = nx.Graph()
g1.add_nodes_from([
    (1, dict(label=0)),
    (2, dict(label=1)),
    (3, dict(label=1))
])
g1.add_edges_from([
    (1, 2),
    (1, 3)
])

g2 = nx.Graph()
g2.add_nodes_from([
    (1, dict(label=0)),
    (2, dict(label=1)),
    (3, dict(label=2))
])
g2.add_edges_from([
    (1, 2),
    (2, 3),
    (1, 3)
])

print(weisfeiler_lehman_subtree_features([g1], 2))