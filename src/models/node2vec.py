import os

import numpy as np

from dlex.configs import AttrDict
from dlex.sklearn.models.word2vec import Word2Vec
from dlex.utils import logger

from ..datasets.base.sklearn import SingleGraphDataset
from ..utils.node2vec import Graph


class Node2Vec(Word2Vec):
    def __init__(self, params: AttrDict, dataset: SingleGraphDataset):
        super().__init__(params)
        self.params = params
        self.dataset = dataset
        self.embeddings = None

    @property
    def output_path(self):
        cfg = self.params.model
        return os.path.join(
            self.params.output_dir,
            "embeddings_d_%d_r_%d_l_%d_k_%d_p_%.2f_q_%.2f.out" % (
                cfg.dimensions,
                cfg.num_walks,
                cfg.walk_length,
                cfg.window_size,
                cfg.p,
                cfg.q
            ))

    def fit(self, X, y=None):
        cfg = self.params.model
        if not os.path.exists(self.output_path):
            logger.info("Calculate node embedding")
            G = self.dataset.get_networkx_graph()
            node2vec_graph = Graph(G, cfg.directed, cfg.p, cfg.q)
            logger.info("Preprocessing transition probabilities...")
            node2vec_graph.preprocess_transition_probs()
            logger.info("Simulating walks...")
            walks = node2vec_graph.simulate_walks(cfg.num_walks, cfg.walk_length)
            walks = [list(map(str, walk)) for walk in walks]
            super().fit(walks)

            # save
            logger.info("Output to %s" % self.output_path)
            embeddings = self.transform(list(G.nodes))
            self.embeddings = {node: emb for node, emb in zip(list(G.nodes), embeddings)}
            with open(self.output_path, "w") as f:
                for node, emb in self.embeddings.items():
                    f.write("%s %s\n" % (node, " ".join([str(k) for k in emb])))
        else:
            logger.info("Load embedding from %s" % self.output_path)
            self.embeddings = {}
            with open(self.output_path) as f:
                for line in f.read().split("\n"):
                    if line.strip() == "":
                        continue
                    line = line.split(' ')
                    self.embeddings[line[0]] = [float(f) for f in line[1:]]
        return self

    def partial_fit(self, X):
        return super().partial_fit(X)

    def transform(self, ls):
        if self.embeddings:
            return np.array([self.embeddings[node] for node in ls])
        else:
            ls = list(map(str, ls))
            return super().transform(ls)