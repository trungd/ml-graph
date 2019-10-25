import functools

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.torch import Batch
from dlex.torch.models import BaseModel, ClassificationModel
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

        self.centers = Parameter(centers_init)
        self.sharpness = Parameter(sharpness_init)

    @staticmethod
    def is_prepared_batch(input):
        if not (isinstance(input, tuple) and len(input) == 4):
            return False
        else:
            batch, not_dummy_points, max_points, batch_size = input
            return isinstance(batch, torch.FloatTensor) and \
                   isinstance(not_dummy_points, torch.FloatTensor) and \
                   max_points > 0 and batch_size > 0

    @staticmethod
    def is_list_of_tensors(input):
        try:
            return all([isinstance(x, torch.FloatTensor) for x in input])
        except TypeError:
            return False

    @property
    def is_gpu(self):
        return self.centers.is_cuda

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


def norm_dgm(dgm):
    if len(dgm) == 0:
        return np.array([]), np.array([])

    not_essential_points = [p for p in dgm if p[1] != float('inf')]
    essential_points = [p for p in dgm if p[1] == float('inf')]

    mi = min([p[0] for p in dgm])

    ma = None
    if len(not_essential_points) != 0:
        ma = max([p[1] for p in not_essential_points])
    else:
        ma = max([p[0] for p in dgm])

    norm_fact = 1
    if ma != mi:
        norm_fact = ma - mi

    not_essential_points = [[(p[0] - mi) / norm_fact, (p[1] - mi) / norm_fact] for p in not_essential_points]
    essential_points = [[(p[0] - mi) / norm_fact, 1] for p in essential_points]
    return not_essential_points, essential_points


def threshold_dgm(dgm, t):
    return list(p for p in dgm if p[1]-p[0] > t)


class VertexFiltrationBase:
    def __init__(self, reddit_graph):
        self._graph = reddit_graph

    def __call__(self, simplex):
        if type(simplex) == int:
            return self._filtration(simplex)
        else:
            return max([self._filtration(v) for v in simplex])

    @functools.lru_cache(maxsize=None)
    def _filtration(self, vertex):
        return self._filtration_implementation(vertex)

    def _filtration_implementation(self, vertex):
        pass


class DegreeVertexFiltration(VertexFiltrationBase):
    def _filtration_implementation(self, vertex):
        return self._graph.degree[vertex]
    
    
def vertex_degree_persistent_diagrams(graph: nx.Graph):
    for n in graph.nodes:
        graph.nodes[n]['weight'] = graph.degree[n]
    return vertex_weight_persistent_diagrams(graph)


def vertex_weight_persistent_diagrams(graph: nx.Graph):
    from dionysus import Simplex, Filtration, homology_persistence, init_diagrams
    views = {}

    simplices = [(v,) for v in graph.nodes] + list(graph.edges)

    f_vertices = [graph.nodes[v]['weight'] for v in graph.nodes]
    f_edges = [max(graph.nodes[v]['weight'] for v in e) for e in graph.edges]

    f_values = f_vertices + f_edges

    # dgm_0, dgm_1 = toplex_persistence_diagrams(simplices, f_values)
    f = Filtration([Simplex(s, f) for s, f in zip(simplices, f_values)])
    m = homology_persistence(f)
    dgms = init_diagrams(m, f)

    dgm_0, dgm_0_essential = norm_dgm([(p.birth, p.death) for p in dgms[0]])
    dgm_1, dgm_1_essential = norm_dgm([(p.birth, p.death) for p in dgms[1]])

    dgm_0, dgm_0_essential = threshold_dgm(dgm_0, 0.01), threshold_dgm(dgm_0_essential, 0.01)
    dgm_1, dgm_1_essential = threshold_dgm(dgm_1, 0.01), threshold_dgm(dgm_1_essential, 0.01)

    dgm_0, dgm_0_essential = np.array(dgm_0), np.array(dgm_0_essential)
    dgm_1, dgm_1_essential = np.array(dgm_1), np.array(dgm_1_essential)

    views['dim_0'] = dgm_0
    views['dim_0_essential'] = dgm_0_essential
    views['dim_1'] = dgm_1
    views['dim_1_essential'] = dgm_1_essential

    return views


class UpperDiagonalThresholdedLogTransform:
    def __init__(self, nu):
        self.b_1 = (torch.Tensor([1, 1]) / np.sqrt(2))
        self.b_2 = (torch.Tensor([-1, 1]) / np.sqrt(2))
        self.nu = nu

    def __call__(self, dgm: torch.FloatTensor):
        if dgm.dim != 2:
            return dgm

        if dgm.is_cuda:
            self.b_1 = self.b_1.cuda()
            self.b_2 = self.b_2.cuda()

        x = torch.mul(dgm, self.b_1.repeat(dgm.size(0), 1))
        x = torch.sum(x, 1)
        y = torch.mul(dgm, self.b_2.repeat(dgm.size(0), 1))
        y = torch.sum(y, 1)
        i = (y <= self.nu)
        y[i] = torch.log(y[i] / self.nu) + self.nu
        ret = torch.stack([x, y], 1)
        return ret


def pers_dgm_center_init(n_elements):
    centers = []
    while len(centers) < n_elements:
        x = np.random.rand(2)
        if x[1] > x[0]:
            centers.append(x.tolist())
    return torch.Tensor(centers)


def reduce_essential_dgm(dgm):
    if dgm.dim != 2:
        return dgm
    else:
        return dgm[:, 0]


class Model(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        def get_init(n_elements):
            transform = UpperDiagonalThresholdedLogTransform(0.1)
            return transform(pers_dgm_center_init(n_elements))

        self.dim_0 = SLayer(150, 2, get_init(150), torch.ones(150, 2) * 3)
        self.dim_0_ess = SLayer(50, 1)
        self.dim_1_ess = SLayer(50, 1)
        self.slayers = [self.dim_0, self.dim_0_ess, self.dim_1_ess]

        self.stage_1 = []
        stage_1_outs = [75, 25, 25]

        for i, (n_in, n_out) in enumerate(zip([150, 50, 50], stage_1_outs)):
            seq = nn.Sequential()
            seq.add_module('linear_1', nn.Linear(n_in, n_out))
            seq.add_module('batch_norm', nn.BatchNorm1d(n_out))
            seq.add_module('drop_out_1', nn.Dropout(0.1))
            seq.add_module('linear_2', nn.Linear(n_out, n_out))
            seq.add_module('relu', nn.ReLU())
            seq.add_module('drop_out_2', nn.Dropout(0.1))

            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        linear_1 = nn.Sequential()
        linear_1.add_module('linear_1', nn.Linear(sum(stage_1_outs), 200))
        linear_1.add_module('batchnorm_1', torch.nn.BatchNorm1d(200))
        linear_1.add_module('relu_1', nn.ReLU())
        linear_1.add_module('linear_2', nn.Linear(200, 100))
        linear_1.add_module('batchnorm_2', torch.nn.BatchNorm1d(100))
        linear_1.add_module('drop_out_2', torch.nn.Dropout(0.1))
        linear_1.add_module('relu_2', nn.ReLU())
        linear_1.add_module('linear_3', nn.Linear(100, 50))
        linear_1.add_module('batchnorm_3', nn.BatchNorm1d(50))
        linear_1.add_module('relu_3', nn.ReLU())
        linear_1.add_module('linear_4', nn.Linear(50, dataset.num_classes))
        linear_1.add_module('batchnorm_4', nn.BatchNorm1d(dataset.num_classes))
        self.linear_1 = linear_1

    def forward(self, batch):
        x = [
            self.transform(batch.X[0].data),
            reduce_essential_dgm(batch.X[1].data), 
            reduce_essential_dgm(batch.X[2].data)
        ]

        x_sl = []
        for i in range(len(x)):
            x_sl.append(self.slayers[i](x[i], batch.X[i].get_mask()))

        x = [l(xx) for l, xx in zip(self.stage_1, x_sl)]
        x = torch.cat(x, 1)
        x = self.linear_1(x)
        return x