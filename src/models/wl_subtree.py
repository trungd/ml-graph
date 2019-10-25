from typing import List

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from dlex.utils import logger


def _get_neighbor_labels(X: nx.Graph, sort: bool = True):
    neighbor_labels = [[X.nodes[u]['label'] for u in X.adj[v].keys()] for v in X.nodes]
    if sort:
        for ls in neighbor_labels:
            ls.sort()
    return neighbor_labels


def weisfeiler_lehman_labels(
        graphs: List[nx.Graph],
        num_iterations,
        preprocess_relabel_dict=None):
    # label_dicts = {}
    # relabel_steps = defaultdict(dict)
    results = {it: {} for it in range(num_iterations + 1)}
    last_new_label = -1
    if preprocess_relabel_dict is None:
        preprocess_relabel_dict = {}

    # relabel graphs
    preprocessed_graphs = []
    for i, g in enumerate(graphs):
        x = g.copy()
        labels = list(nx.get_node_attributes(x, "label").values())

        new_labels = []
        for label in labels:
            if label in preprocess_relabel_dict.keys():
                new_labels.append(preprocess_relabel_dict[label])
            else:
                last_new_label += 1
                preprocess_relabel_dict[label] = last_new_label
                new_labels.append(preprocess_relabel_dict[label])
        for node, label in zip(x.nodes, new_labels):
            x.nodes[node]['label'] = label
        results[0][i] = (labels, new_labels)
        preprocessed_graphs.append(x)
    graphs = preprocessed_graphs

    for it in range(1, num_iterations + 1):
        last_new_label = -1
        label_dict = {}
        for i, g in enumerate(graphs):
            # Get labels of current iteration
            current_labels = nx.get_node_attributes(g, "label").values()

            # Get for each vertex the labels of its neighbors
            neighbor_labels = _get_neighbor_labels(g, sort=True)

            # Prepend the vertex label to the list of labels of its neighbors
            merged_labels = [[b] + a for a, b in zip(neighbor_labels, current_labels)]

            # Generate a label dictionary based on the merged labels
            for merged_label in merged_labels:
                dict_key = '-'.join(map(str, merged_label))
                if dict_key not in label_dict.keys():
                    last_new_label += 1
                    label_dict[dict_key] = last_new_label

            # Relabel the graph
            new_labels = [label_dict['-'.join(map(str, merged))] for merged in merged_labels]
            # relabel_steps[i][it] = {
            #    idx: {old_label: new_labels[idx]} for idx, old_label in enumerate(current_labels)}
            for node, label in zip(g, new_labels):
                g.nodes[node]['label'] = label

            results[it][i] = (merged_labels, new_labels)
        # label_dicts[it] = copy.deepcopy(label_dict)
    return results


class WLSubtree(BaseEstimator, TransformerMixin):
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations

    def fit_transform(self, graphs: List[nx.Graph], y=None, **fit_params):
        label_dicts = weisfeiler_lehman_labels(graphs, self.num_iterations)
        X_per_iteration = []
        num_columns_per_iteration = {}

        for it in sorted(label_dicts.keys()):
            wl_graphs = [graph.copy() for graph in graphs]

            for graph_index in sorted(label_dicts[it].keys()):
                labels_raw, labels_compressed = label_dicts[it][graph_index]

                # Assign the compressed label (an integer) to the
                # 'subtree graph' in order to generate features.
                for node, label in zip(wl_graphs[graph_index].nodes, labels_compressed):
                    wl_graphs[graph_index].nodes[node]['label'] = label

            # Calculates the feature vectors of a sequence of graphs.
            # The `label` attribute is used to calculate features.
            labels = set()
            for graph in wl_graphs:
                labels.update(nx.get_node_attributes(graph, 'label').values())
            num_labels = len(labels)
            # Ensures that the labels form a contiguous sequence of indices so that they can be easily mapped.
            assert min(labels) == 0 and max(labels) == num_labels - 1
            # Increases readability and follows the 'persistent' feature
            # generation method.
            X = np.zeros((len(wl_graphs), num_labels))
            for i, graph in enumerate(graphs):
                x = np.zeros(num_labels)
                for node in graph.nodes:
                    x[graph.nodes[node]['label']] += 1
                X[i, :] = x
            X_per_iteration.append(X)

            if it not in num_columns_per_iteration:
                num_columns_per_iteration[it] = X_per_iteration[-1].shape[1]

        logger.info(str(num_columns_per_iteration))
        return np.concatenate(X_per_iteration, axis=1)