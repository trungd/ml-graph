import random

import networkx as nx
import numpy as np
from comptopo.filtrations.vertex_weight_extended import extended_vertex_weight_persistence_diagrams
from comptopo.filtrations import vertex_weight_persistence_diagrams, edge_weight_persistence_diagrams
from grakel import datasets
from src.utils.graph_signatures import assign_vertex_weight, assign_edge_weight, return_probability_signature
from src.utils.utils import get_networkx_graph
import matplotlib.pyplot as plt
from matplotlib import collections as mc

if False:
    dataset = datasets.fetch_dataset("REDDIT-MULTI-5K", verbose=False)
    G, y = dataset.data, dataset.target
    G = [get_networkx_graph(g[1], g[0], g[2]) for g in G]


def generate_graph(num_nodes, num_edges):
    g = nx.Graph()
    g.add_nodes_from([(i, dict(weight=i)) for i in range(1, num_nodes + 1)])
    g.add_edges_from(random.choices(
        [(i, j) for j in range(1, num_nodes + 1) for i in range(j + 1, num_nodes + 1)],
        k=num_edges))
    return g


# g = G[10]
g = generate_graph(5, 10)
print(return_probability_signature(g, 5))

# assign_edge_weight(g, 'random')
# assign_vertex_weight(g, 'random')
# assign_edge_weight(g, 'random')
# dgms = edge_weight_persistence_diagrams(g, tool='gudhi')
# dgms = vertex_weight_persistence_diagrams(g, use_clique=True)
# dgms.normalize()
# print(vertex_weight_persistence_diagrams(g, tool='dionysus'))
# assign_vertex_weight(g, 'hks', t=0.1)
# print(extended_vertex_weight_persistence_diagrams(g))


def visualize(g: nx.Graph):
    # nx.draw(g)
    random.seed(42)
    pts = {node: [random.random(), g.nodes[node]['weight']] for node in g.nodes}

    fig, ax = plt.subplots()
    ax.scatter([pt[0] for pt in pts.values()], [pt[1] for pt in pts.values()], s=8)
    for node, pt in pts.items():
        ax.annotate(node, pt, size=5)
    lc = mc.LineCollection([[pts[u], pts[v]] for u, v in g.edges], linewidths=1)

    ax.add_collection(lc)

    plt.savefig("test.png")


# visualize(g)
