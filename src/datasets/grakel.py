import os

from grakel import datasets
from grakel import GraphKernel
import numpy as np

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.sklearn import SklearnDataset
from dlex.utils import logger
from ..models.persistent_wl_subtree import PersistentWLSubtree
from ..models.wl_subtree import WLSubtree
from ..utils.utils import get_networkx_graph

# from ..utils.kernels.persistent_weisfeiler_lehman import weisfeiler_lehman_subtree_features


class Grakel(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)

    def get_sklearn_wrapper(self, mode: str):
        return GrakelDataset(self)


class GrakelDataset(SklearnDataset):
    def __init__(self, builder):
        super().__init__(builder)

        dataset = datasets.fetch_dataset(builder.params.dataset.dataset_name, verbose=False)
        G, y = dataset.data, dataset.target
        logger.info("Dataset size: %d", len(y))
        logger.info("Average node count: %.2f", sum([len(g[1]) for g in G]) / len(G))
        logger.info("Average edge count: %.2f", sum([len(g[2]) for g in G]) / len(G))

        graph_kernel = builder.params.dataset.graph_kernel
        graph_vector = builder.params.dataset.graph_vector
        assert graph_vector or graph_kernel, "Either vector or kernel must be specified"
        assert not (graph_vector and graph_kernel), "Only vector or kernel"

        if graph_kernel:
            if graph_kernel == "shortest_path":
                gk = GraphKernel(kernel=dict(name="shortest_path"), normalize=True)
            elif graph_kernel == "graphlet_sampling":
                gk = GraphKernel(kernel=[
                    dict(name="graphlet_sampling", sampling=dict(n_samples=150))], normalize=True)
            elif graph_kernel == "wl-subtree":
                gk = GraphKernel(kernel=[
                    dict(name="weisfeiler_lehman", niter=5),
                    dict(name="subtree_wl")
                ], normalize=True)
            else:
                raise Exception("Graph kernel is not valid: %s" % graph_kernel)

            self.init_dataset(G, y)
            self.X_train = gk.fit_transform(self.X_train)
            self.X_test = gk.transform(self.X_test)
        elif graph_vector:
            if graph_vector == "wl-subtree":
                graphs = [get_networkx_graph(g[1], g[0], g[2]) for g in G]
                feature_extractor = WLSubtree(num_iterations=builder.params.dataset.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif graph_vector == "persistent-wl":
                graphs = [get_networkx_graph(g[1], g[0], g[2]) for g in G]
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=False,
                    num_iterations=builder.params.dataset.h)
                X, _ = feature_extractor.fit_transform(graphs)
            elif graph_vector == "persistent-wlc":
                graphs = [get_networkx_graph(g[1], g[0], g[2]) for g in G]
                feature_extractor = PersistentWLSubtree(
                    use_label_persistence=True,
                    use_cycle_persistence=True,
                    num_iterations=builder.params.dataset.h)
                X, _ = feature_extractor.fit_transform(graphs)
            else:
                raise Exception("Graph vector is not valid: %s" % graph_vector)

            self.init_dataset(X, y)