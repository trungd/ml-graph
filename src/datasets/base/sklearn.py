import os
import pickle
from typing import List

import numpy as np
from dlex.datasets.sklearn import SklearnDataset
from dlex.utils import logger
from tqdm import tqdm
from comptopo import PersistenceDiagrams
from comptopo.vectors import persistence_image, persistence_landscape

from ...utils.graph_signatures import assign_vertex_weight


class SingleGraphDataset(SklearnDataset):
    def __init__(self, builder):
        super().__init__(builder)

    def get_networkx_graph(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()


class MultiGraphsDataset(SklearnDataset):
    def __init__(self, builder):
        super().__init__(builder)
        self._input_size = None

    @property
    def feature_name(self):
        return "fil_%s_sig_%s_diagrams" % (
            self.params.dataset.graph_filtration,
            self.params.dataset.graph_signature
        )

    def _get_persistent_diagrams(self) -> List[List[PersistenceDiagrams]]:
        graph_filtration = self.params.dataset.graph_filtration
        graph_signature = self.params.dataset.graph_signature

        file_name = os.path.join(self.builder.get_processed_data_dir(), "%s.pkl" % self.feature_name)
        load_diagrams = True
        if load_diagrams and os.path.exists(file_name):
            with open(file_name, "rb") as f:
                dgms = pickle.load(f)
            dgms = [PersistenceDiagrams.from_list(d) for d in dgms]
            logger.info("Features loaded from %s" % file_name)
        else:
            graphs = self.get_networkx_graphs()

            if graph_signature == 'vertex_label':
                for graph in graphs:
                    assign_vertex_weight(graph, 'vertex_label')
            elif graph_signature == 'vertex_degree':
                for graph in graphs:
                    assign_vertex_weight(graph, 'degree')
            elif graph_signature == 'hks':
                for graph in graphs:
                    assign_vertex_weight(graph, 'hks', t=0.1)
            elif graph_signature == 'random':
                for graph in graphs:
                    assign_vertex_weight(graph, 'random')

            dgms = []
            if graph_filtration == "vertex_weight":
                from comptopo.filtrations.vertex_weight import vertex_weight_persistence_diagrams
                for graph in tqdm(graphs, desc="Extracting PDs"):
                    dgms.append(vertex_weight_persistence_diagrams(graph, tool='gudhi'))
            elif graph_filtration == "extended_vertex_weight":
                from comptopo.filtrations.vertex_weight_extended import extended_vertex_weight_persistence_diagrams
                for graph in tqdm(graphs, desc="Extracting PDs"):
                    dgms.append(extended_vertex_weight_persistence_diagrams(graph))

            with open(file_name, "wb") as f:
                pickle.dump([d.to_list() for d in dgms], f)

        for graph_dgm in dgms:
            for d in graph_dgm:
                d.normalize()
                d.threshold()

        return dgms

    def extract_features(self):
        params = self.params
        graph_kernel = params.dataset.graph_kernel
        graph_vector = params.dataset.graph_vector
        graph_filtration = params.dataset.graph_filtration
        
        assert not (graph_vector and graph_kernel), "Only vector or kernel"
        if graph_kernel:
            if graph_kernel == "shortest_path":
                from grakel import GraphKernel
                gk = GraphKernel(kernel=dict(name="shortest_path"), normalize=True)
                self.init_dataset(self.G, self.y)
                self.X_train = gk.fit_transform(self.X_train)
                self.X_test = gk.transform(self.X_test)
            elif graph_kernel == "graphlet_sampling":
                from grakel import GraphKernel
                gk = GraphKernel(kernel=[
                    dict(name="graphlet_sampling", sampling=dict(n_samples=500))], normalize=True)
                self.init_dataset(self.G, self.y)
                self.X_train = gk.fit_transform(self.X_train)
                self.X_test = gk.transform(self.X_test)
            elif graph_kernel == "wl-subtree":
                from grakel import GraphKernel
                gk = GraphKernel(kernel=[
                    dict(name="weisfeiler_lehman", niter=5),
                    dict(name="subtree_wl")
                ], normalize=True)
                self.init_dataset(self.G, self.y)
                self.X_train = gk.fit_transform(self.X_train)
                self.X_test = gk.transform(self.X_test)
            elif graph_kernel == "bottleneck_distance":
                dgms = self._get_persistent_diagrams()
                self.init_dataset(dgms, self.y)
            else:
                raise Exception("Graph kernel is not valid: %s" % graph_kernel)
        elif graph_vector:
            if graph_vector == "wl-subtree":
                #graphs = self.get_networkx_graphs()
                #from ...models.wl_subtree import WLSubtree
                #feature_extractor = WLSubtree(num_iterations=params.dataset.h)
                #X = feature_extractor.fit_transform(graphs)

                graphs = self.get_networkx_graphs()
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=False,
                    use_cycle_persistence=False,
                    num_iterations=params.dataset.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif graph_vector == "persistent-wl":
                graphs = self.get_networkx_graphs()
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=False,
                    num_iterations=params.dataset.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif graph_vector == "persistent-wlc":
                graphs = self.get_networkx_graphs()
                from ...models.persistent_wl_subtree import PersistentWLSubtree
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=True,
                    num_iterations=params.dataset.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif graph_vector == "persistence_image":
                dgms = self._get_persistent_diagrams(graph_filtration)
                X = persistence_image([
                    d[0].not_essential_points + d[0].essential_points for d in dgms])
                self._input_size = 400
            elif graph_vector == "persistence_landscape":
                dgms = self._get_persistent_diagrams(graph_filtration)
                X = persistence_landscape([
                    d[0].not_essential_points + d[0].essential_points for d in dgms])
                self._input_size = 500
            else:
                raise Exception("Graph vector is not valid: %s" % graph_vector)
            self.init_dataset(X, self.y)
        else:
            self.init_dataset(self.get_networkx_graphs(), self.y)

    @property
    def input_size(self):
        return self._input_size

    def get_networkx_graphs(self):
        raise NotImplementedError
