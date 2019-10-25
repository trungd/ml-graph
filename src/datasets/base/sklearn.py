import os
import pickle

from dlex.datasets.sklearn import SklearnDataset
from tqdm import tqdm


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
                if graph_filtration == "vertex_degree":
                    file_name = os.path.join(self.builder.get_processed_data_dir(), "vertex_degree_diagrams.pkl")
                    if os.path.exists(file_name):
                        with open(file_name, "rb") as f:
                            dgms = pickle.load(f)
                    else:
                        from ...models.vertex_weight_persistent_feature import vertex_degree_persistent_diagrams
                        graphs = self.get_networkx_graphs()
                        dgms = []
                        for graph in tqdm(graphs, desc="Extracting PDs"):
                            dgms.append(vertex_degree_persistent_diagrams(graph))
                        with open(file_name, "wb") as f:
                            pickle.dump(dgms, f)
                self.init_dataset(dgms, self.y)
            else:
                raise Exception("Graph kernel is not valid: %s" % graph_kernel)
        elif graph_vector:
            if graph_vector == "wl-subtree":
                graphs = self.get_networkx_graphs()
                from ...models.wl_subtree import WLSubtree
                feature_extractor = WLSubtree(num_iterations=params.dataset.h)
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
            else:
                raise Exception("Graph vector is not valid: %s" % graph_vector)
            self.init_dataset(X, self.y)
        else:
            self.init_dataset(self.get_networkx_graphs(), self.y)

    def get_networkx_graphs(self):
        raise NotImplementedError()
