import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.ops_utils import maybe_cuda
from ..utils.graph_signatures import heat_kernel_signature
from ..utils.persistent_diagram import PersistenceDiagrams
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class SLayer(Module):
    """
    Implementation of the deep learning layer in
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


class VertexBasedPersistenceDiagramsCalculator:
    def __init__(self, signature_fn=None):
        self.signature_fn = signature_fn

    def __call__(self, graph: nx.Graph) -> PersistenceDiagrams:
        if self.signature_fn:
            weights = self.signature_fn(graph)
            for n in graph.nodes:
                graph.nodes[n]['weight'] = weights[n]

        from dionysus import Simplex, Filtration, homology_persistence, init_diagrams

        cliques = set()
        for v1, v2 in graph.edges:
            for v in graph.nodes:
                if v != v1 and v != v2 and graph.has_edge(v, v1) and graph.has_edge(v, v2):
                    cliques.add(tuple(sorted([v, v1, v2])))
        cliques = set()
        simplices = [[v] for v in graph.nodes] + [[u, v] for u, v in graph.edges] + list(cliques)

        max_weight = max([graph.nodes[v]['weight'] for v in graph.nodes])
        f_vertices = [max_weight - graph.nodes[v]['weight'] for v in graph.nodes]
        f_edges = [max_weight - min(graph.nodes[v]['weight'] for v in e) for e in graph.edges]
        f_cliques = [min(graph.nodes[v]['weight'] for v in c) for c in cliques]

        f_values = f_vertices + f_edges + f_cliques

        f = Filtration([Simplex(s, f) for s, f in zip(simplices, f_values)])
        f.sort()
        m = homology_persistence(f)
        dgms = init_diagrams(m, f)
        dgms = [[[p.birth, p.death] for p in d] for d in dgms]
        return PersistenceDiagrams.from_list(dgms)


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

        stage_1_ins = [150, 50, 50]
        # stage_1_ins = [150, 50]
        self.slayers = [
            SLayer(stage_1_ins[0], 2, centers_init, torch.ones(stage_1_ins[0], 2) * 3),  # dim 0
            SLayer(stage_1_ins[1], 1),  # dim 0 essential
            SLayer(stage_1_ins[2], 1)  # dim 1 essential
        ]

        self.stage_1 = nn.ModuleList()

        stage_1_outs = [75, 25, 25]
        # stage_1_outs = [75, 25]

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


import gudhi as gd


class ExtendedVertexBasedPersistenceDiagramsCalculator(VertexBasedPersistenceDiagramsCalculator):
    """
    Implementation of the extended persistence diagrams in the paper
        PersLay: A Neural Network Layer of Persistence Diagrams and New Graph Topological Signatures
    """
    def __init__(self, signature_fn=None):
        super().__init__(signature_fn)

    @staticmethod
    def _get_base_simplex(graph: nx.Graph):
        st = gd.SimplexTree()
        for node in graph.nodes:
            st.insert([node], filtration=-1e10)
        
        for u, v in graph.edges:
            st.insert([u, v], filtration=-1e10)
        return st.get_filtration()

    def __call__(self, graph: nx.Graph) -> PersistenceDiagrams:
        if self.signature_fn:
            weights = self.signature_fn(graph)
            for n in graph.nodes:
                graph.nodes[n]['weight'] = weights[n]
        basesimplex = self._get_base_simplex(graph)
        filtration_val = heat_kernel_signature(graph, 0.1)
        idx2node = {i: val for i, val in zip(range(len(graph)), filtration_val.keys())}
        filtration_val = np.array(list(filtration_val.values()))

        min_val, max_val = filtration_val.min(), filtration_val.max()
        # edge based:
        # else:
        #     min_val = min([filtration_val[xs[i], ys[i]] for i in range(num_edges)])
        #     max_val = max([filtration_val[xs[i], ys[i]] for i in range(num_edges)])
    
        st = gd.SimplexTree()
        st.set_dimension(2)
    
        for simplex, filt in basesimplex:
            st.insert(simplex=simplex + [-2], filtration=-3)
    
        if True:  # vertex-based
            if max_val == min_val:
                fa = -.5 * np.ones(filtration_val.shape)
                fd = .5 * np.ones(filtration_val.shape)
            else:
                fa = -2 + (filtration_val - min_val) / (max_val - min_val)
                fd = 2 - (filtration_val - min_val) / (max_val - min_val)

            for vid in range(len(graph.nodes)):
                st.assign_filtration(simplex=[idx2node[vid]], filtration=fa[vid])
                st.assign_filtration(simplex=[idx2node[vid], -2], filtration=fd[vid])
        #else:
        #    A = nx.adjacency_matrix(graph).todense()
        #    (xs, ys) = np.where(np.triu(A))
        #    num_edges = len(graph.edges)
        #    if max_val == min_val:
        #        fa = -.5 * np.ones(filtration_val.shape)
        #        fd = .5 * np.ones(filtration_val.shape)
        #    else:
        #        fa = -2 + (filtration_val - min_val) / (max_val - min_val)
        #        fd = 2 - (filtration_val - min_val) / (max_val - min_val)
        #    for eid in range(num_edges):
        #        vidx, vidy = xs[eid], ys[eid]
        #        st.assign_filtration(simplex=[vidx, vidy], filtration=fa[vidx, vidy])
        #        st.assign_filtration(simplex=[vidx, vidy, -2], filtration=fd[vidx, vidy])
        #    for vid in range(num_vertices):
        #        if len(np.where(A[vid, :] > 0)[0]) > 0:
        #            st.assign_filtration(simplex=[vid], filtration=min(fa[vid, np.where(A[vid, :] > 0)[0]]))
        #            st.assign_filtration(simplex=[vid, -2], filtration=min(fd[vid, np.where(A[vid, :] > 0)[0]]))
    
        st.make_filtration_non_decreasing()
        distorted_dgm = st.persistence()
        normal_dgm = dict()
        normal_dgm["Ord0"], normal_dgm["Rel1"], normal_dgm["Ext0"], normal_dgm["Ext1"] = [], [], [], []
        for point in range(len(distorted_dgm)):
            dim, b, d = distorted_dgm[point][0], distorted_dgm[point][1][0], distorted_dgm[point][1][1]
            pt_type = "unknown"
            if (-2 <= b <= -1 and -2 <= d <= -1) or (b == -.5 and d == -.5):
                pt_type = "Ord" + str(dim)
            if (1 <= b <= 2 and 1 <= d <= 2) or (b == .5 and d == .5):
                pt_type = "Rel" + str(dim)
            if (-2 <= b <= -1 and 1 <= d <= 2) or (b == -.5 and d == .5):
                pt_type = "Ext" + str(dim)
            if np.isinf(d):
                continue
            else:
                b, d = min_val + (2 - abs(b)) * (max_val - min_val), min_val + (2 - abs(d)) * (max_val - min_val)
                if b <= d:
                    normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([b, d])]))
                else:
                    normal_dgm[pt_type].append(tuple([distorted_dgm[point][0], tuple([d, b])]))
    
        dgmOrd0 = np.array([normal_dgm["Ord0"][point][1] for point in range(len(normal_dgm["Ord0"]))])
        dgmExt0 = np.array([normal_dgm["Ext0"][point][1] for point in range(len(normal_dgm["Ext0"]))])
        dgmRel1 = np.array([normal_dgm["Rel1"][point][1] for point in range(len(normal_dgm["Rel1"]))])
        dgmExt1 = np.array([normal_dgm["Ext1"][point][1] for point in range(len(normal_dgm["Ext1"]))])
        if dgmOrd0.shape[0] == 0:
            dgmOrd0 = np.zeros([0, 2])
        if dgmExt1.shape[0] == 0:
            dgmExt1 = np.zeros([0, 2])
        if dgmExt0.shape[0] == 0:
            dgmExt0 = np.zeros([0, 2])
        if dgmRel1.shape[0] == 0:
            dgmRel1 = np.zeros([0, 2])
        return PersistenceDiagrams.from_list([dgmOrd0, dgmExt0, dgmRel1, dgmExt1])