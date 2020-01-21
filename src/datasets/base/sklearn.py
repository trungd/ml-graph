import json
import os
import pickle
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
from dlex.utils import logger, get_file_size
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
        self.init_dataset(dgms, self.labels)

    @property
    def num_classes(self):
        raise NotImplementedError


class MultiGraphsDataset(SklearnDataset):
    graph_features = {}

    def __init__(self, builder):
        super().__init__(builder)
        self._input_size = None
        self._num_node_labels = None
        self._persistence_diagrams = None

    @property
    def feature_name(self):
        configs = self.params.dataset.graph_features.persistence_diagram
        return "fil_%s_sig_%s_diagrams" % (
            configs.filtration,
            configs.signature
        )

    def _get_vertex_weights(self, signature):
        graphs = self.networkx_graphs
        # configs = self.params.dataset.graph_features.persistence_diagram
        filepath = os.path.join(self.builder.get_processed_data_dir(), "vertex_weight_%s.pkl" % self.feature_name)
        if self.configs.reuse and os.path.exists(filepath):
            with open(filepath, "rb") as f:
                weights = pickle.load(f)
                logger.info("Vertex weights loaded from %s" % filepath)
        else:
            for graph in tqdm(graphs, desc="Extracting PDs"):
                assign_vertex_weight(graph, signature, num_labels=self.num_node_labels)
            weights = [nx.get_node_attributes(graph, 'weight') for graph in graphs]
            with open(filepath, "wb") as f:
                pickle.dump(weights, f)
                logger.info("Vertex weights saved to %s" % filepath)
        return weights

    def _get_persistence_diagrams(self) -> List[PersistenceDiagrams]:
        if self._persistence_diagrams:
            return self._persistence_diagrams

        configs = self.params.dataset.graph_features.persistence_diagram
        pkl_filepath = os.path.join(self.builder.get_processed_data_dir(), "%s.pkl" % self.feature_name)
        filepath = os.path.join(self.builder.get_processed_data_dir(), "%s.json" % self.feature_name)

        dgms = None
        if self.configs.reuse and os.path.exists(pkl_filepath):
            logger.info("Loading persistence diagrams from %s (%.2f MB)...", pkl_filepath, get_file_size(pkl_filepath))
            with open(pkl_filepath, "rb") as f:
                dgms = pickle.load(f)
        elif self.configs.reuse and os.path.exists(filepath):
            logger.info("Loading persistence diagrams from %s (%.2f MB)...", filepath, get_file_size(filepath))
            try:
                with open(filepath, "r") as f:
                    dgms = json.load(f)
                    dgms = [PersistenceDiagrams.from_dict(d) for d in dgms]
                    with open(pkl_filepath, "wb") as pkl_f:
                        logger.info("Saving to %s..." % pkl_filepath)
                        pickle.dump(dgms, pkl_f)
            except ValueError:
                dgms = None

        if not dgms:
            graphs = self.networkx_graphs
            dgms = []
            logger.info("Filtration: %s", configs.filtration)
            weights = self._get_vertex_weights(configs.signature)
            for i in range(len(graphs)):
                assign_vertex_weight(graphs[i], weights=weights[i])
            if configs.filtration == "vertex_weight":
                for graph in tqdm(graphs, desc="Extracting PDs", leave=True):
                    dgms.append(vertex_weight_persistence_diagrams(graph, tool='gudhi', use_clique=True))
            elif configs.filtration == "extended_vertex_weight":
                for graph in tqdm(graphs, desc="Extracting PDs", leave=True):
                    dgms.append(extended_vertex_weight_persistence_diagrams(graph))
            elif configs.filtration == "edge_weight":
                for graph in tqdm(graphs, desc="Extracting PDs", leave=True):
                    dgms.append(edge_weight_persistence_diagrams(graph, tool='gudhi'))

            save_persistence_diagrams(filepath, dgms)
            with open(pkl_filepath, "wb") as f:
                pickle.dump(dgms, f)

        logger.info("Persistence diagrams loaded.")
        logger.info("Average size: %.2f", np.mean([np.mean([len(d) for d in ds.diagrams]) for ds in dgms]))
        logger.info("Max size: %d", np.max([np.max([len(d) for d in ds.diagrams]) for ds in dgms]))

        for d in dgms:
            d.normalize()
            d.threshold()

        # for key in ['h0', 'h1', 'h0_non_essential', 'h0_essential', 'h1_essential']:
        #     logger.debug("Sample of persistence diagrams (%s): %s", key, str(dgms[0][key]))

        self._persistence_diagrams = dgms
        return dgms

    def extract_features(self):
        """
        Extract graph features
            - If there is one feature, its values are stored in X_train and X_test.
            - If there are more than one features, its values are stored in self.graph_features.
                X_train and X_test are the indices of graph in each set
        """
        logger.info("Extracting graph features...")
        # split train and test set
        self.init_dataset(list(range(len(self.labels))), self.labels)

        for feat_name in self.configs.graph_features:
            feat = self.configs.graph_features[feat_name]
            if feat.type == "shortest_path":
                from grakel import GraphKernel
                G_train = [self.graphs[i] for i in self.X_train]
                G_test = [self.graphs[i] for i in self.X_test]
                gk = GraphKernel(kernel=dict(name="shortest_path"), normalize=True)
                X = np.zeros([len(self.graphs), len(G_train)])
                X[self.X_train] = gk.fit_transform(G_train)
                X[self.X_test] = gk.transform(G_test)
            elif feat.type == "graphlet_sampling":
                from grakel import GraphKernel
                G_train = [self.graphs[i] for i in self.X_train]
                G_test = [self.graphs[i] for i in self.X_test]
                gk = GraphKernel(kernel=[
                    dict(name="graphlet_sampling", sampling=dict(n_samples=500))], normalize=True)
                X = np.zeros([len(self.graphs), len(G_train)])
                X[self.X_train] = gk.fit_transform(G_train)
                X[self.X_test] = gk.transform(G_test)
            elif feat.type == "wl_subtree":
                from grakel import GraphKernel
                G_train = [self.graphs[i] for i in self.X_train]
                G_test = [self.graphs[i] for i in self.X_test]
                gk = GraphKernel(kernel=[
                    dict(name="weisfeiler_lehman", n_iter=5),
                    dict(name="subtree_wl")
                ], normalize=True)
                X = np.zeros([len(self.graphs), len(G_train)])
                X[self.X_train] = gk.fit_transform(G_train)
                X[self.X_test] = gk.transform(G_test)
            elif feat.type == "persistence_diagram":
                X = self._get_persistence_diagrams()
                self._input_size = 2
            elif feat.type == "wl-subtree":
                #graphs = self.networkx_graphs
                #from ...models.wl_subtree import WLSubtree
                #feature_extractor = WLSubtree(num_iterations=params.dataset.h)
                #X = feature_extractor.fit_transform(graphs)

                graphs = self.networkx_graphs
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=False,
                    use_cycle_persistence=False,
                    num_iterations=feat.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif feat.type == "persistent_wl":
                graphs = self.networkx_graphs
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=False,
                    num_iterations=feat.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif feat.type == "persistent_wlc":
                graphs = self.networkx_graphs
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
        logger.info("Extracting features done.")

    @property
    def input_size(self):
        return self._input_size

    @property
    def networkx_graphs(self):
        raise NotImplementedError
