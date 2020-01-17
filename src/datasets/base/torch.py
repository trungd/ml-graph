import json
import os
import pickle
import re
from typing import List

import numpy as np
import networkx as nx
from dlex.datasets.torch import Dataset
from dlex.utils import logger
from tqdm import tqdm
from comptopo import PersistenceDiagrams
from comptopo.vectors import persistence_image, persistence_landscape

from ...utils.graph_signatures import assign_vertex_weight, assign_edge_weight


class SingleGraphDataset(Dataset):
    node_embeddings = None
    adj_matrix = None

    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
    
    def get_networkx_graph(self) -> nx.Graph:
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError


class MultiGraphsDataset(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

    def get_networkx_graphs(self):
        raise NotImplementedError
