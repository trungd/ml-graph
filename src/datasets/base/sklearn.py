import json
import os
import random
import re
from typing import List

import networkx as nx
import numpy as np
from comptopo import PersistenceDiagrams
from comptopo.filtrations import edge_weight_persistence_diagrams, vertex_weight_persistence_diagrams, \
    extended_vertex_weight_persistence_diagrams
from comptopo.vectors import persistence_image, persistence_landscape
from dlex.datasets.sklearn import SklearnDataset
from dlex.utils import logger
from tqdm import tqdm

from ...utils.graph_signatures import assign_vertex_weight, assign_edge_weight


def save_persistence_diagrams(file_name: str, pds: List[PersistenceDiagrams]):
    """
    Save list of persistence diagrams in JSON format
    :param file_name:
    :param pds:
    """
    with open(file_name, "w") as f:
        s = json.dumps([pd.to_dict() for pd in pds], indent=2)
        s = re.sub(r'\s*\[\s*([\-a-zA-Z\.\d]+),\s*([\-a-zA-Z\.\d]+)\s*\](,?)\s*', r'[\1, \2]\3', s)
        s = s.replace("],[", "], [")
        f.write(s)
        logger.debug("Persistence diagrams saved to %s" % file_name)


class SingleGraphDataset(SklearnDataset):
    def __init__(self, builder):
        super().__init__(builder)

    def get_networkx_graph(self) -> nx.Graph:
        raise NotImplementedError

    @property
    def persistence_diagrams_tag(self):
        configs = self.params.dataset.graph_features.persistence_diagram
        return "fil_%s_sig_%s_diagrams" % (
            configs.filtration,
            configs.signature
        )
    
    def _get_persistence_diagrams(self) -> List[PersistenceDiagrams]:
        configs = self.params.dataset.graph_features.persistence_diagram

        file_name = os.path.join(self.builder.get_processed_data_dir(), "%s.json" % self.persistence_diagrams_tag)

        dgms = None
        if configs.reuse and os.path.exists(file_name):
            try:
                with open(file_name, "r") as f:
                    dgms = json.load(f)
                    dgms = [PersistenceDiagrams.from_dict(d) for d in dgms]
                    logger.debug("Persistence diagrams loaded from %s" % file_name)
            except ValueError:
                dgms = None
                
        if not dgms:
            graph = self.get_networkx_graph()
            if configs.filtration == "vertex_weight":
                from comptopo.filtrations import vertex_weight_persistence_diagrams
                nodes = list(graph.nodes)
                dgms = []

                maximum_hops = 5
                if configs.signature == "distance":
                    lengths = dict(nx.all_pairs_dijkstra_path_length(graph))
                    hops = dict(nx.all_pairs_shortest_path_length(graph, cutoff=maximum_hops))

                for src_node in tqdm(nodes, desc="Extracting PDs"):
                    if configs.signature == "distance":
                        weights = {}
                        for node in graph:
                            weights[node] = float('inf')
                            if node in lengths[src_node] and node in hops[src_node]:
                                if hops[src_node][node] <= maximum_hops:
                                    weights[node] = - lengths[src_node][node]
                        assign_vertex_weight(graph, weights=weights)
                        dgms.append(vertex_weight_persistence_diagrams(graph, tool='gudhi', use_clique=False))

            save_persistence_diagrams(file_name, dgms)

        for d in dgms:
            d.normalize()
            d.threshold()

        return dgms

    def extract_features(self):
        params = self.params
        graph_kernel = params.dataset.graph_kernel
        graph_vector = params.dataset.graph_vector

        assert not (graph_vector and graph_kernel), "Only vector or kernel"
        dgms = self._get_persistence_diagrams()
        self.init_dataset(dgms, self.y)

    @property
    def num_classes(self):
        raise NotImplementedError


class MultiGraphsDataset(SklearnDataset):
    graph_features = {}

    def __init__(self, builder):
        super().__init__(builder)
        self._input_size = None

    @property
    def feature_name(self):
        configs = self.params.dataset.graph_features.persistence_diagram
        return "fil_%s_sig_%s_diagrams" % (
            configs.filtration,
            configs.signature
        )

    def _get_persistence_diagrams(self) -> List[PersistenceDiagrams]:
        configs = self.params.dataset.graph_features.persistence_diagram

        file_name = os.path.join(self.builder.get_processed_data_dir(), "%s.json" % self.feature_name)

        dgms = None
        if configs.reuse and os.path.exists(file_name):
            try:
                with open(file_name, "r") as f:
                    dgms = json.load(f)
                    dgms = [PersistenceDiagrams.from_dict(d) for d in dgms]
                    logger.debug("Persistence diagrams loaded from %s" % file_name)
            except ValueError:
                dgms = None

        if not dgms:
            graphs = self.get_networkx_graphs()
            dgms = []
            logger.debug("Filtration: %s", configs.filtration)
            if configs.filtration == "vertex_weight":
                for graph in tqdm(graphs, desc="Extracting PDs"):
                    assign_vertex_weight(graph, configs.signature)
                    dgms.append(vertex_weight_persistence_diagrams(graph, tool='gudhi', use_clique=True))
            elif configs.filtration == "extended_vertex_weight":
                for graph in tqdm(graphs, desc="Extracting PDs"):
                    assign_vertex_weight(graph, configs.signature)
                    dgms.append(extended_vertex_weight_persistence_diagrams(graph))
            elif configs.filtration == "edge_weight":
                for graph in tqdm(graphs, desc="Extracting PDs"):
                    assign_edge_weight(graph, configs.signature)
                    dgms.append(edge_weight_persistence_diagrams(graph, tool='gudhi'))

            save_persistence_diagrams(file_name, dgms)

        for d in dgms:
            d.normalize()
            d.threshold()

        # for key in ['h0', 'h1', 'h0_non_essential', 'h0_essential', 'h1_essential']:
        #     logger.debug("Sample of persistence diagrams (%s): %s", key, str(dgms[0][key]))

        return dgms

    def extract_features(self):
        """
        Extract graph features
            - If there is one feature, its values are stored in X_train and X_test.
            - If there are more than one features, its values are stored in self.graph_features.
                X_train and X_test are the indices of graph in each set
        """
        self.init_dataset(list(range(len(self.y))), self.y)
        G_train = [self.G[i] for i in self.X_train]
        G_test = [self.G[i] for i in self.X_test]

        for feat_name in self.configs.graph_features:
            feat = self.configs.graph_features[feat_name]
            if feat.type == "shortest_path":
                from grakel import GraphKernel
                gk = GraphKernel(kernel=dict(name="shortest_path"), normalize=True)
                X = np.zeros([len(self.G), len(G_train)])
                X[self.X_train] = gk.fit_transform(G_train)
                X[self.X_test] = gk.transform(G_test)
            elif feat.type == "graphlet_sampling":
                from grakel import GraphKernel
                gk = GraphKernel(kernel=[
                    dict(name="graphlet_sampling", sampling=dict(n_samples=500))], normalize=True)
                X = np.zeros([len(self.G), len(G_train)])
                X[self.X_train] = gk.fit_transform(G_train)
                X[self.X_test] = gk.transform(G_test)
            elif feat.type == "wl_subtree":
                from grakel import GraphKernel
                gk = GraphKernel(kernel=[
                    dict(name="weisfeiler_lehman", n_iter=5),
                    dict(name="subtree_wl")
                ], normalize=True)
                X = np.zeros([len(self.G), len(G_train)])
                X[self.X_train] = gk.fit_transform(G_train)
                X[self.X_test] = gk.transform(G_test)
            elif feat.type == "persistence_diagram":
                X = self._get_persistence_diagrams()
                self._input_size = 2
            elif feat.type == "wl-subtree":
                #graphs = self.get_networkx_graphs()
                #from ...models.wl_subtree import WLSubtree
                #feature_extractor = WLSubtree(num_iterations=params.dataset.h)
                #X = feature_extractor.fit_transform(graphs)

                graphs = self.get_networkx_graphs()
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=False,
                    use_cycle_persistence=False,
                    num_iterations=feat.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif feat.type == "persistent_wl":
                graphs = self.get_networkx_graphs()
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=False,
                    num_iterations=feat.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif feat.type == "persistent_wlc":
                graphs = self.get_networkx_graphs()
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=True,
                    num_iterations=feat.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif feat.type == "persistence_image":
                dgms = self._get_persistence_diagrams()
                X = persistence_image([
                    d[0].not_essential_points + d[0].essential_points for d in dgms])
                self._input_size = 400
            elif feat.type == "persistence_landscape":
                dgms = self._get_persistence_diagrams()
                X = persistence_landscape([
                    d[0].not_essential_points + d[0].essential_points for d in dgms])
                self._input_size = 500
            else:
                raise ValueError("Feature type is not valid: %s" % feat.type)

            self.graph_features[feat_name] = X

        if len(self.graph_features) == 1:
            # if there is only one feature, its values go into X. If not, X is a set of indices
            key = list(self.graph_features.keys())[0]
            self.X = self.graph_features[key]
            self.X_train = [self.X[i] for i in self.X_train]
            self.X_test = [self.X[i] for i in self.X_test]

    @property
    def input_size(self):
        return self._input_size

    def get_networkx_graphs(self):
        raise NotImplementedError
