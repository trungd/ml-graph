import numpy as np
import torch
import torch.nn as nn
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.torch.utils.variable_length_tensor import get_mask
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
            sharpness_init: Tensor = None,
            perm_op: str = "sum"):
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

        self.perm_op = dict(
            sum=torch.sum,
            max=torch.max,
            min=torch.min
        )[perm_op]

    def forward(self, diagrams, masks) -> Variable:
        batch_size, max_points = diagrams.shape[0], diagrams.shape[1]
        if max_points == 0:
            # logger.warn("Length 0 batch.")
            return maybe_cuda(torch.zeros(batch_size, self.n_elements))
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

        x = self.perm_op(x, 2)
        x = x.squeeze()

        return x


class DeepSets(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        def pers_dgm_center_init(n_elements):
            centers = []
            while len(centers) < n_elements:
                x = np.random.rand(2)
                if x[1] > x[0]:
                    centers.append(x.tolist())
            return torch.FloatTensor(centers)

        self.dgm_keys = list(params.model.deep_set.dim.keys())

        stage_1_ins = [params.model.deep_set.dim[key] for key in self.dgm_keys]
        stage_1_outs = [d // 2 for d in stage_1_ins]

        self.slayers = nn.ModuleList()
        for key in self.dgm_keys:
            dim = 1 if key in ["h0_essential", "h1_essential"] else 2
            self.slayers.append(SLayer(
                params.model.deep_set.dim[key],
                dim,
                pers_dgm_center_init(params.model.deep_set.dim[key]) if dim == 2 else None,
                torch.ones(stage_1_ins[0], 2) * 3 if dim == 2 else None)
            )

        self.stage_1 = nn.ModuleList()
        for i, (n_in, n_out) in enumerate(zip(stage_1_ins, stage_1_outs)):
            self.stage_1.append(nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out),
                nn.Dropout(params.model.dropout),
                nn.Linear(n_out, n_out),
                nn.ReLU(),
                nn.Dropout(params.model.dropout)
            ))

        linear_layers = []
        for in_dim, out_dim in zip(
                [sum(stage_1_outs)] + params.model.dense_dim,
                params.model.dense_dim + [dataset.num_classes]):
            linear_layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
        self.pd_linear = nn.Sequential(*linear_layers)

        if False:
            linear_layers = []
            for in_dim, out_dim in zip(
                    [dataset.embedding_dim] + params.model.dense_dim,
                    params.model.dense_dim + [dataset.num_classes]):
                linear_layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
            self.emb_linear = nn.Sequential(*linear_layers)

    def forward(self, batch):
        x_pd = []
        for key in self.params.model.deep_set.dim:
            pt_dim = 1 if key in ["h0_essential", "h1_essential"] else 2
            x_pd.append((batch.X[key][0].data[:, :, :pt_dim], get_mask(batch.X[key][1])))

        x_sl = []
        for i in range(len(x_pd)):
            x_sl.append(self.slayers[i](x_pd[i][0], x_pd[i][1]))

        x_pd = [l(xx) for l, xx in zip(self.stage_1, x_sl)]
        x_pd = torch.cat(x_pd, 1)
        x_pd = self.pd_linear(x_pd)

        #x_emb = batch.X['embedding']
        #x_emb = self.emb_linear(x_emb)
        return x_pd