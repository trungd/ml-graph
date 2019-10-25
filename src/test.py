import random

import networkx as nx
from grakel import datasets
from src.models.vertex_weight_persistent_feature import vertex_degree_persistent_diagrams
from src.utils.utils import get_networkx_graph

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

g = G[10]
for u, v in g.edges:
    g.edges[u, v]['weight'] = random.random()

print(vertex_degree_persistent_diagrams(g))
