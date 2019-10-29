from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.ops_utils import maybe_cuda
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def safe_tensor_size(tensor, dim):
    try:
        return tensor.size(dim)
    except Exception:
        return 0


class SLayer(Module):
    """
    Implementation of the in
    {
      Hofer17c,
      author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
      title     = {Deep Learning with Topological Signatures},
      booktitle = {NIPS},
      year      = 2017,
      note      = {accepted}
    }
    proposed input layer for multisets.
    """
    def __init__(
            self,
            n_elements: int,
            point_dimension: int = 2,
            centers_init: Tensor = None,
            sharpness_init: Tensor = None):
        """
        :param n_elements: number of structure elements used
        :param point_dimension: dimensionality of the points of which the input multi set consists of
        :param centers_init: the initialization for the centers of the structure elements
        :param sharpness_init: initialization for the sharpness of the structure elements
        """
        super(SLayer, self).__init__()

        self.n_elements = n_elements
        self.point_dimension = point_dimension

        if centers_init is None:
            centers_init = torch.rand(self.n_elements, self.point_dimension)

        if sharpness_init is None:
            sharpness_init = torch.ones(self.n_elements, self.point_dimension)*3

        self.centers = maybe_cuda(Parameter(centers_init))
        self.sharpness = maybe_cuda(Parameter(sharpness_init))

    def forward(self, diagrams, masks) -> Variable:
        batch_size, max_points = diagrams.shape[0], diagrams.shape[1]
        diagrams = torch.cat([diagrams] * self.n_elements, 1)
        masks = torch.cat([masks] * self.n_elements, 1)

        centers = torch.cat([self.centers] * max_points, 1)
        centers = centers.view(-1, self.point_dimension)
        centers = torch.stack([centers] * batch_size, 0)

        sharpness = torch.cat([self.sharpness] * max_points, 1)
        sharpness = sharpness.view(-1, self.point_dimension)
        sharpness = torch.stack([sharpness] * batch_size, 0)

        x = centers - diagrams
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.sum(x, 2)
        x = torch.exp(-x)
        x = torch.mul(x, masks.float())
        x = x.view(batch_size, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x


class Toplex:
    def __init__(self, toplices: [tuple], filtration_values: [], deessentialize=False):
        self.simplices = [tuple(t) for t in toplices]
        self.deessentialize = deessentialize
        self.filtration = filtration_values

        self._check_state()

    def _check_state(self):
        if len(self.simplices) != len(self.filtration):
            raise ToplexException("Simplices and filtration are not consistent: Assumed to have same length.")

    @property
    def filtration(self):
        return [self._internal_filt_to_filt[v] for v in self._internal_filt]

    @filtration.setter
    def filtration(self, filt):
        self._filt_to_internal_filt = {}
        self._internal_filt_to_filt = {}
        for i, v in enumerate(sorted(list(set(filt)))):
            self._filt_to_internal_filt[v] = i + 1
            self._internal_filt_to_filt[i + 1] = v

        self._internal_filt = [self._filt_to_internal_filt[v] for v in filt]
        self._internal_filt_to_filt[-1] = max(filt) if self.deessentialize else float('inf')

    def _simplex_to_string_iter(self):
        def num_iter(simplex, filtration_value):
            yield str(len(simplex) - 1)

            for n in simplex:
                yield str(n)

            yield str(filtration_value)

        for s, f in zip(self.simplices, self._internal_filt):
            yield ' '.join(num_iter(s, f))

    def _convert_dgm_from_internal_filt_to_filt(self, dgm):
        # points = [p for p in dgm if p[1] != -1]
        # essential_points = [p for p in dgm if p[1] == -1]

        points = [[self._internal_filt_to_filt[p[0]], self._internal_filt_to_filt[p[1]]] for p in dgm]
        # essential_points = [[self._internal_filt_to_filt[p[0]], float('inf')] for p in essential_points]

        # return points + essential_points
        return points

    def calculate_persistence_diagrams(self):
        # complex = [for self.simplices]
        dgms = _call_perseus('nmfsimtop', complex_string)

        return_value = []

        homology_dimension_upper_bound = max([len(s) for s in self.simplices])
        for dim in range(homology_dimension_upper_bound):
            if dim in dgms:
                return_value.append(self._convert_dgm_from_internal_filt_to_filt(dgms[dim]))
            else:
                return_value.append([])

        return return_value


def toplex_persistence_diagrams(toplices: [tuple], filtration_values: [], deessentialize=False):
    """
    Calculates the persistence diagrams for the given toplex using the given
    filtration. A toplex is a notion of a simplicial complex where just the
    highest dimensional simplices are noted, i.e. toplex
    {[1,2]} stands for the simplicial complex {[1], [2], [1,2]}
    :param toplices: List of toplices given as numeric tuples.
    The numeric value of each dimension of a toplix tuple stands
    for a vertex, e.g. [1, 2, 3] is the 2 simplex built from the vertices 1, 2, 3.
    :param filtration_values: List which gives the filtration value of each toplix
    enlisted in toplices.
    :param deessentialize: If True the death-time of essential classes is mapped to max(filtration_values).
    If False the death time is mapped to float('inf').
    :return: [[[]]
    """
    toplex = Toplex(toplices, filtration_values, deessentialize=deessentialize)
    return toplex.calculate_persistence_diagrams()


class ToplexException(Exception):
    pass


class PersistentDiagrams:
    def __init__(self, values: list = None):
        if values is None:
            self.pd = []
        else:
            self.pd = values

        self.essential_points = [p for p in self.pd if p[1] == float('inf')]
        self.not_essential_points = [p for p in self.pd if p[1] != float('inf')]

    def __getitem__(self, item):
        return self.pd[item]

    def __len__(self):
        return len(self.pd)

    def normalize(self):
        if len(self.pd) == 0:
            return np.array([]), np.array([])

        min_birth = min([p[0] for p in self.pd])
        max_death = max([
            p[1] for p in self.not_essential_points]) if len(self.not_essential_points) != 0 \
            else max([p[0] for p in self.pd])

        norm_fact = max_death - min_birth or 1

        self.not_essential_points = [[
            (p[0] - min_birth) / norm_fact,
            (p[1] - min_birth) / norm_fact
        ] for p in self.not_essential_points]

        self.essential_points = [[(p[0] - min_birth) / norm_fact, 1] for p in self.essential_points]

    def threshold(self, t: float = 0.01):
        self.essential_points = list(p for p in self.essential_points if p[1] - p[0] > t)
        self.not_essential_points = list(p for p in self.not_essential_points if p[1] - p[0] > t)


def vertex_weight_persistent_diagrams(
        graph: nx.Graph) -> List[List[int]]:
    from dionysus import Simplex, Filtration, homology_persistence, init_diagrams

    simplices = [(v,) for v in graph.nodes] + list(graph.edges)

    f_vertices = [graph.nodes[v]['weight'] for v in graph.nodes]
    f_edges = [max(graph.nodes[v]['weight'] for v in e) for e in graph.edges]

    f_values = f_vertices + f_edges

    # dgm_0, dgm_1 = toplex_persistence_diagrams(simplices, f_values)
    f = Filtration([Simplex(s, f) for s, f in zip(simplices, f_values)])
    m = homology_persistence(f)
    dgms = init_diagrams(m, f)
    dgms = [[[p.birth, p.death] for p in d] for d in dgms]
    return dgms


def vertex_degree_persistent_diagrams(graph: nx.Graph):
    for n in graph.nodes:
        graph.nodes[n]['weight'] = graph.degree[n]
    return vertex_weight_persistent_diagrams(graph)


class Model(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.b_1 = maybe_cuda(torch.Tensor([1, 1]) / np.sqrt(2))
        self.b_2 = maybe_cuda(torch.Tensor([-1, 1]) / np.sqrt(2))

        def pers_dgm_center_init(n_elements):
            centers = []
            while len(centers) < n_elements:
                x = np.random.rand(2)
                if x[1] > x[0]:
                    centers.append(x.tolist())
            return torch.FloatTensor(centers)

        centers_init = self._upper_diagonal_transform(pers_dgm_center_init(150))
        self.slayers = [
            SLayer(150, 2, centers_init, torch.ones(150, 2) * 3),  # dim 0
            SLayer(50, 1),  # dim 0 essential
            SLayer(50, 1)  # dim 1 essential
        ]

        self.stage_1 = nn.ModuleList()
        stage_1_ins = [150, 50, 50]
        # stage_1_ins = [150]
        stage_1_outs = [75, 25, 25]
        # stage_1_outs = [75]

        for i, (n_in, n_out) in enumerate(zip(stage_1_ins, stage_1_outs)):
            self.stage_1.append(nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.Dropout(0.1),
                nn.Linear(n_out, n_out),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))

        self.linear = nn.Sequential(
            nn.Linear(sum(stage_1_outs), 200),
            torch.nn.BatchNorm1d(200), nn.ReLU(),
            nn.Linear(200, 100),
            torch.nn.BatchNorm1d(100), torch.nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50), nn.ReLU(),
            nn.Linear(50, dataset.num_classes),
            nn.BatchNorm1d(dataset.num_classes)
        )

    def _upper_diagonal_transform(self, dgm: torch.FloatTensor, nu=0.1):
        if dgm.dim != 2:
            return dgm

        x = torch.mul(dgm, self.b_1.repeat(dgm.size(0), 1))
        x = torch.sum(x, 1)
        y = torch.mul(dgm, self.b_2.repeat(dgm.size(0), 1))
        y = torch.sum(y, 1)
        i = (y <= nu)
        y[i] = torch.log(y[i] / nu) + nu
        ret = torch.stack([x, y], 1)
        return ret

    @staticmethod
    def _reduce_essential_dgm(dgm):
        return dgm if dgm.dim != 2 else dgm[:, 0]

    def forward(self, batch):
        x = [
            self._upper_diagonal_transform(batch.X[0].data),
            self._reduce_essential_dgm(batch.X[1].data),
            self._reduce_essential_dgm(batch.X[2].data)
        ]

        x_sl = []
        for i in range(len(x)):
            x_sl.append(self.slayers[i](x[i], batch.X[i].get_mask()))

        x = [l(xx) for l, xx in zip(self.stage_1, x_sl)]
        x = torch.cat(x, 1)
        x = self.linear(x)
        return x
