import collections
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from dlex.utils import logger
from .wl_subtree import weisfeiler_lehman_labels


class UnionFind:
    """
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    """

    def __init__(self, nodes):
        self._parent = {x: x for x in nodes}

    def find(self, u):
        """Finds and returns the parent of u with respect to the hierarchy."""

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        for node, parent in self._parent.items():
            if node == parent:
                yield node


class PersistenceDiagram(collections.abc.Sequence):
    def __init__(self):
        self._pairs = []
        self._betti = None

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]

    def append(self, x, y, index=None):
        self._pairs.append((x, y, index))

    def total_persistence(self, p=1):
        return sum([abs(x - y) ** p for x, y, _ in self._pairs]) ** (1.0 / p)

    def infinity_norm(self, p=1):
        return max([abs(x - y) ** p for x, y, _ in self._pairs])

    def remove_diagonal(self):
        self._pairs = [(x, y, c) for x, y, c in self._pairs if x != y]

    @property
    def betti(self):
        return self._betti

    @betti.setter
    def betti(self, value):
        assert value <= len(self), "Betti number must be less than or equal to persistence diagram cardinality"
        self._betti = value

    def __repr__(self):
        return '\n'.join([f'{x} {y} [{c}]' for x, y, c in self._pairs])


def _persistent_diagrams(
        graph: nx.Graph,
        order='sublevel',
        unpaired_value=None,
        vertex_attribute=None):
    uf = UnionFind(list(graph.nodes.keys()))

    edge_weights = np.array(list(nx.get_edge_attributes(graph, 'weight').values()))  # All edge weights
    edge_indices = None  # Ordering for filtration
    edge_indices_cycles = []  # Edge indices of cycles

    assert order in ["sublevel", "superlevel"]
    if order == 'sublevel':
        edge_indices = np.argsort(edge_weights, kind='stable')
    elif order == 'superlevel':
        edge_indices = np.argsort(-edge_weights, kind='stable')

    pd = PersistenceDiagram()

    for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):
        u, v = list(graph.edges.keys())[edge_index]

        # Preliminary assignment of younger and older component. We
        # will check below whether this is actually correct, for it
        # is possible that u is actually the older one.
        younger = uf.find(u)
        older = uf.find(v)

        if younger == older:
            edge_indices_cycles.append(edge_index)
            continue

        # Ensures that the older component precedes the younger one
        # in terms of its vertex index
        if younger > older:
            u, v = v, u
            younger, older = older, younger

        if vertex_attribute:
            vertex_weight = graph.nodes[vertex_attribute][younger]
        else:
            vertex_weight = 0.0

        creation = vertex_weight  # x coordinate for persistence diagram
        destruction = edge_weight  # y coordinate for persistence diagram

        uf.merge(u, v)
        pd.append(creation, destruction, younger)

    # By default, use the largest (sublevel set) or lowest
    # (superlevel set) weight, unless the user specified a
    # different one.
    unpaired_value = unpaired_value or edge_weights[edge_indices[-1]]

    # Add tuples for every root component in the Union--Find data
    # structure. This ensures that multiple connected components
    # are handled correctly.
    for root in uf.roots():

        vertex_weight = 0.0

        # Vertex attributes have been set, so we use them for the creation of the root tuple.
        if vertex_attribute:
            vertex_weight = graph.nodes[vertex_attribute][root]

        creation = vertex_weight
        destruction = unpaired_value
        pd.append(creation, destruction, root)
        pd.betti = pd.betti + 1 if pd.betti else 1

    return pd, edge_indices_cycles


class PersistentWLSubtree(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        num_iterations: int,
        use_infinity_norm=False,
        use_total_persistence=False,
        use_label_persistence=False,
        use_cycle_persistence=False,
        use_original_features=True,
        metric='minkowski',
        p=2,
        smooth=False
    ):
        super().__init__()
        self.num_iterations = num_iterations or 2
        self.use_infinity_norm = use_infinity_norm
        self.use_total_persistence = use_total_persistence
        self.use_label_persistence = use_label_persistence
        self.use_cycle_persistence = use_cycle_persistence
        self.use_original_features = use_original_features
        self.metric = metric
        self.p = p
        self.smooth = smooth

    def _assign_edge_weights(self, graph: nx.Graph, tau=1.0):
        def _ensure_list(l):
            return [l] if type(l) is not list else l

        def _minkowski(A, B):
            a, b = _to_vectors(A, B)
            return np.linalg.norm(a - b, ord=self.p)

        def _to_vectors(A, B):
            """
            Transforms two sets of labels to their corresponding
            high-dimensional vectors. For example, a sequence of
            `{a, a, b}` and `{a, c, c}` will be transformed to a
            vector `(2, 1, 0)` and `(1, 0, 2)`, respectively.

            This function does not have to care about the global
            alphabet of labels because they will only yield zero
            values.

            :param A: First label sequence
            :param B: Second label sequence

            :return: Two transformed vectors
            """

            label_to_index = dict()
            index = 0
            for label in A + B:
                if label not in label_to_index:
                    label_to_index[label] = index
                    index += 1

            a = np.zeros(len(label_to_index))
            b = np.zeros(len(label_to_index))

            for label in A:
                a[label_to_index[label]] += 1

            for label in B:
                b[label_to_index[label]] += 1

            return a, b

        metric = _minkowski
        for edge in graph.edges:
            source, target = edge

            source_labels = _ensure_list(graph.nodes[source]['label'])
            target_labels = _ensure_list(graph.nodes[target]['label'])

            source_label = source_labels[0]
            target_label = target_labels[0]

            weight = metric(source_labels[1:], target_labels[1:])

            # For all non-uniform metrics, we want to take into account
            # the differences between the source and target label of an
            # edge.
            # if metric != uniform:
            weight = weight + (source_label != target_label) + tau

            # Update the edge weight if smoothing is required for the distances.
            if self.smooth:
                graph.edges[edge]['weight'] += weight
            else:
                graph.edges[edge]['weight'] = weight

        return graph

    def fit_transform(self, graphs: List[nx.Graph], y=None, **fit_params):
        label_dicts = weisfeiler_lehman_labels(graphs, self.num_iterations)

        X_per_iteration = []
        num_columns_per_iteration = {}

        PDs = defaultdict(list)

        # Stores the *original* labels in the original graph for
        # subsequent forward propagation.
        original_labels = defaultdict(dict)

        # It is sufficient to copy the graphs *once*; their edge weights
        # will be reset anyway (unless 'smoothed' distances are selected
        # by the client).
        weighted_graphs = [graph.copy() for graph in graphs]

        for it, label_dict in label_dicts.items():
            for idx in sorted(label_dict.keys()):
                graph = weighted_graphs[idx]
                labels_raw, labels_compressed = label_dict[idx]
                labels_raw = {list(graph.nodes.keys())[i]: label for i, label in enumerate(labels_raw)}
                labels_compressed = {list(graph.nodes.keys())[i]: label for i, label in enumerate(labels_compressed)}

                nx.set_node_attributes(graph, labels_raw, "label")
                nx.set_node_attributes(graph, labels_compressed, "compressed_label")

                # Assign the *compressed* labels as the *original*
                # labels of the graph in order to ensure that they
                # are zero-indexed.
                if it == 0:
                    original_labels[idx] = labels_compressed
                else:
                    nx.set_node_attributes(graph, original_labels[idx], 'original_labels')

                weighted_graphs[idx] = self._assign_edge_weights(weighted_graphs[idx])

            graphs = weighted_graphs
            num_labels = 0

            persistence_diagrams = []

            # Calculating label persistence requires us to know the number
            # of distinct labels in the set of graphs as it determines the
            # length of the created feature vector.
            if self.use_label_persistence or self.use_original_features or self.use_cycle_persistence:
                labels = set()

                for graph in graphs:
                    labels.update(nx.get_node_attributes(graph, 'compressed_label').values())
                num_labels = len(labels)

            X = []

            # Fill the feature matrix by calculating persistence-based features for each of the graphs
            for index, graph in enumerate(graphs):
                persistence_diagram, edge_indices_cycles = _persistent_diagrams(graph)
                x = []
                if self.use_infinity_norm:
                    x += [persistence_diagram.infinity_norm(self.p)]
                if self.use_total_persistence:
                    x += [persistence_diagram.total_persistence(self.p)]
                if self.use_original_features:
                    x_original_features = [0] * num_labels
                    for _, _, c in persistence_diagram:
                        label = graph.nodes[c]['compressed_label']
                        x_original_features[label] += 1
                    x += x_original_features
                if self.use_label_persistence:
                    x_label_persistence = [0] * num_labels
                    for ptx, pty, c in persistence_diagram:
                        label = graph.nodes[c]['compressed_label']
                        persistence = abs(ptx - pty) ** self.p
                        x_label_persistence[label] += persistence
                    persistence_diagrams.append(persistence_diagram)
                    x += x_label_persistence
                # Cycle persistence: use the edge information, i.e. the
                # classification gained from the persistence diagram
                # calculation above, in order to assign weights.
                if self.use_cycle_persistence:
                    n = len(persistence_diagram)
                    m = graph.number_of_edges()
                    k = persistence_diagram.betti
                    num_cycles = m - n + k

                    assert num_cycles == len(edge_indices_cycles)

                    x_cycle_persistence = np.zeros(num_labels)
                    total_cycle_persistence = 0.0

                    for edge_index in edge_indices_cycles:
                        edge = list(graph.edges.keys())[edge_index]
                        source, target = edge
                        weight = graph.edges[edge]['weight']

                        total_cycle_persistence += weight ** self.p

                        source_label = graph.nodes[source]['compressed_label']
                        target_label = graph.nodes[target]['compressed_label']

                        x_cycle_persistence[source_label] += weight ** self.p
                        x_cycle_persistence[target_label] += weight ** self.p
                    x += x_cycle_persistence.tolist()
                X.append(x)

            X_per_iteration.append(np.array(X))
            PDs[it] = persistence_diagrams

            if it not in num_columns_per_iteration:
                num_columns_per_iteration[it] = X_per_iteration[-1].shape[1]

        logger.info("Number of columns per iteration: %s", str(num_columns_per_iteration))
        return np.concatenate(X_per_iteration, axis=1), num_columns_per_iteration