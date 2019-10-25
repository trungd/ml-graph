from typing import List

import numpy as np
import networkx as nx
import copy


def weisfeiler_lehman_kernel(
        graph_list: List[nx.Graph],
        h=1,  # number of iterations
        node_label=True) -> np.ndarray:
    """
    node_label:
        Whether to use original node labels. True for using node labels
        saved in the attribute 'label'. False for using the node
        degree of each node as node attribute.
    """
    num_graphs = len(graph_list)
    adj_lists = []
    n_nodes = 0
    num_nodes_max = 0

    # Compute adjacency lists and n_nodes, the total number of
    # nodes in the dataset.
    for i in range(num_graphs):
        adj_lists.append(graph_list[i].adjacency())
        n_nodes = n_nodes + graph_list[i].number_of_nodes()

        # Computing the maximum number of nodes in the graphs. It
        # will be used in the computation of vectorial
        # representation.
        num_nodes_max = max(num_nodes_max, graph_list[i].number_of_nodes())

    phi = np.zeros((num_nodes_max, num_graphs), dtype=np.uint64)

    # INITIALIZATION: initialize the nodes labels for each graph
    # with their labels or with degrees (for unlabeled graphs)

    labels = []
    label_lookup = {}
    label_counter = 0

    # label_lookup is an associative array, which will contain the
    # mapping from multiset labels (strings) to short labels
    # (integers)

    if node_label is True:
        for i in range(num_graphs):
            l_aux = nx.get_node_attributes(graph_list[i], 'label').values()
            labels.append(np.zeros(len(l_aux), dtype=np.int32))

            for j in range(len(l_aux)):
                if not (l_aux[j] in label_lookup):
                    label_lookup[l_aux[j]] = label_counter
                    labels[-1][j] = label_counter
                    label_counter += 1
                else:
                    labels[-1][j] = label_lookup[l_aux[j]]
                # labels are associated to a natural number
                # starting with 0.
                phi[labels[i][j], i] += 1
    else:
        for i in range(num_graphs):
            d = graph_list[i].degree()
            labels.append(np.array([d[k] for k in range(len(d))]))
            for j in range(len(labels[i])):
                phi[labels[-1][j], i] += 1

    # Simplified vectorial representation of graphs (just taking
    # the vectors before the kernel iterations), i.e., it is just
    # the original nodes degree.
    # vectors = np.copy(phi.transpose())

    k = np.dot(phi.transpose(), phi)

    new_labels = copy.deepcopy(labels)

    for it in range(h):
        # create an empty lookup table
        label_lookup = {}
        label_counter = 0

        phi = np.zeros((n_nodes, n), dtype=np.uint64)
        for i in range(n):
            for v, adjs in enumerate(adj_lists[i]):
                # form a multiset label of the node v of the i'th graph
                # and convert it to a string

                long_label = str(np.concatenate([
                    np.array([labels[i][v]]),
                    np.sort(labels[i][list(adjs[1].keys())])
                ]))
                # if the multiset label has not yet occurred, add it to the
                # lookup table and assign a number to it
                if long_label not in label_lookup:
                    label_lookup[long_label] = label_counter
                    new_labels[i][v] = label_counter
                    label_counter += 1
                else:
                    new_labels[i][v] = label_lookup[long_label]
            # fill the column for i'th graph in phi
            aux = np.bincount(new_labels[i])
            phi[new_labels[i], i] = np.add(phi[new_labels[i], i], aux[new_labels[i]])

        k += np.dot(phi.transpose(), phi)
        labels = copy.deepcopy(new_labels)

    # Compute the normalized version of the kernel
    k_norm = np.zeros(k.shape)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            k_norm[i, j] = k[i, j] / np.sqrt(k[i, i] * k[j, j])

    return k_norm


def weisfeiler_lehman_distance(
        g1: nx.Graph,
        g2: nx.Graph,
        h=1,
        node_label=True):
    gl = [g1, g2]
    return weisfeiler_lehman_kernel(gl, h, node_label)[0, 1]
