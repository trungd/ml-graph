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
