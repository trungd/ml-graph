import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex import Params
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.model_utils import linear_layers, MultiLinear
from dlex.torch.utils.variable_length_tensor import get_mask
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.activation import MultiheadAttention


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, layer_norm=False):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(query_dim, value_dim)
        self.fc_k = nn.Linear(key_dim, value_dim)
        self.fc_v = nn.Linear(key_dim, value_dim)

        self.layer_norm_0 = nn.LayerNorm(value_dim) if layer_norm else None
        self.layer_norm_1 = nn.LayerNorm(value_dim) if layer_norm else None

        self.fc_o = nn.Linear(value_dim, value_dim)

    def forward(self, Q, K, mask):
        batch_size = Q.size(0)
        Q = self.fc_q(Q)  # shape: [batch_size, max_len, query_dim]
        K, V = self.fc_k(K), self.fc_v(K)

        split_dim = self.value_dim // self.num_heads
        Q_ = torch.cat(Q.split(split_dim, 2), 0)  # shape: [batch_size * num_heads, max_len, split_dim]
        K_ = torch.cat(K.split(split_dim, 2), 0)
        V_ = torch.cat(V.split(split_dim, 2), 0)

        A = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.key_dim)
        A = torch.softmax(A, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(batch_size, 0), 2)
        O = self.layer_norm_0(O) if self.layer_norm_0 else O
        O = O + F.relu(self.fc_o(O))
        O = self.layer_norm_1(O) if self.layer_norm_1 else O
        return O


class SetAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, layer_norm=False, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.attn = MultiheadAttention(
            input_dim, num_heads, kdim=input_dim, vdim=output_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask):
        X = X.transpose(0, 1)
        X = X.repeat(1, 1, self.num_heads)
        X2, _ = self.attn(X, X, X, key_padding_mask=mask)
        X = X + self.dropout1(X2)
        X = self.norm1(X)
        if hasattr(self, "activation"):
            X2 = self.linear2(self.dropout(self.activation(self.linear1(X))))
        else:  # for backward compatibility
            X2 = self.linear2(self.dropout(F.relu(self.linear1(X))))
        X = X + self.dropout2(X2)
        X = self.norm2(X)
        return X.transpose(0, 1)


class InducedSetAttentionBlock(nn.Module):
    """
    Implementation for Induced Set Attention Block
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, num_inds: int, layer_norm=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, output_dim))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiheadAttention(
            embed_dim=output_dim, kdim=input_dim, vdim=output_dim,
            num_heads=num_heads, layer_norm=layer_norm)
        self.mab1 = MultiheadAttention(
            embed_dim=input_dim, kdim=output_dim, vdim=output_dim,
            num_heads=num_heads, layer_norm=layer_norm)

    def forward(self, X, mask):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
        return self.mab1(X, H, mask)


class PoolingMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, layer_norm=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.attn = MultiheadAttention(dim, num_heads)

    def forward(self, X, mask):
        batch_size = X.size(0)
        X = self.S.repeat(batch_size, 1, 1)
        return self.attn(X, X, mask)


class DeepSets(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        cfg = self.configs
        self.feature_name = params.dataset.graph_features.persistence_diagram.key
        dim_input = len(self.feature_name.split('.')) * 3 + 2

        self.enc = linear_layers([dim_input] + [cfg.hidden_dim] * params.model.num_layers, batch_norm=False)
        self.emb_enc = nn.Linear(dim_input - 2, dim_input - 2)
        self.dec = linear_layers([cfg.hidden_dim + dim_input - 2] + [cfg.dense_dim] * (cfg.num_layers - 1) + [dataset.num_classes])
        self.dropout = nn.Dropout(params.model.dropout)

    def forward(self, batch):
        X, X_len = batch.X
        X_enc = self.dropout(self.enc(X))
        X_emb = X[:, :, :-2]
        X = torch.cat([X_emb, X_enc], -1)
        X = X * get_mask(X_len).unsqueeze(-1).float()
        X = X.sum(-2)
        X = self.dropout(self.dec(X))
        return X


class MultiDeepSets(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        cfg = self.configs
        self.feature_name = params.dataset.graph_features.persistence_diagram.key
        self.has_freq = self.feature_name[:5] == "freq_"
        self.has_pd_embedding = "_non_embed" not in self.feature_name
        dim_input = 3 + 2

        self.enc = linear_layers(
            [dim_input] + [cfg.hidden_dim] * params.model.num_encoder_layers,
            dropout=cfg.dropout,
            norm=nn.BatchNorm1d if cfg.batch_norm else nn.LayerNorm)
        self.emb_enc = nn.Linear(dim_input - 2, dim_input - 2)
        self.dec = linear_layers(
            [cfg.hidden_dim + dim_input - 2] + [cfg.dense_dim] * (cfg.num_decoder_layers - 1) + [dataset.num_classes],
            norm=nn.BatchNorm1d if cfg.batch_norm else nn.LayerNorm,
            dropout=cfg.dropout)
        self.dropout = nn.Dropout(params.model.dropout)

    def forward(self, batch):
        X, X_len = self.to_cuda_tensors(batch.X)
        if self.has_freq:
            X, freq = X[:, :, :, :-1], X[:, :, :, -1]

        X_enc = self.dropout(self.enc(X.reshape([-1, X.shape[3]])))
        X_enc = X_enc.reshape(list(X.shape[:-1]) + [-1])

        X_emb = X[:, :, :, :-2]
        X = torch.cat([X_emb, X_enc], -1)

        for i in range(len(X)):
            X[i] = X[i] * get_mask(X_len[i], max_len=X[i].shape[-2]).unsqueeze(-1).float()

        if self.has_freq:
            X = X * freq.unsqueeze(-1)
        X = X.sum(-2)  # pooling points in pds
        X = X.sum(-2)  # pooling pds

        X = self.dropout(self.dec(X))
        return X


class MultiDeepSetsTransformer(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        cfg = self.configs
        self.feature_name = params.dataset.graph_features.persistence_diagram.key
        self.has_freq = self.feature_name[:5] == "freq_"
        dim_input = 3 + 2

        self.enc = linear_layers(
            [dim_input] + [cfg.hidden_dim] * (params.model.num_encoder_layers - 1) + [cfg.hidden_dim - 3],
            dropout=cfg.dropout,
            norm=nn.LayerNorm)
        self.emb_enc = nn.Linear(dim_input - 2, dim_input - 2)
        self.transformer_enc = Encoder(
            hidden_dim=cfg.encoder.dim_model,
            num_heads=cfg.encoder.num_heads,
            num_layers=cfg.encoder.num_layers,
            dropout=cfg.dropout,
            batch_norm=cfg.batch_norm)
        self.dec = linear_layers(
            [cfg.hidden_dim] + [cfg.dense_dim] * (cfg.num_decoder_layers - 1) + [dataset.num_classes],
            dropout=cfg.dropout,
            norm=nn.BatchNorm1d if cfg.batch_norm else nn.LayerNorm)
        self.dropout = nn.Dropout(params.model.dropout)

    def forward(self, batch):
        X, X_len = batch.X
        if self.has_freq:
            X, freq = X[:, :, :, :-1], X[:, :, :, -1]

        X_enc = self.dropout(self.enc(X.reshape([-1, X.shape[3]])))
        X_enc = X_enc.reshape(list(X.shape[:-1]) + [-1])

        X_emb = X[:, :, :, :-2]
        X = torch.cat([X_emb, X_enc], -1)

        if self.has_freq:
            X = X * freq.unsqueeze(-1)
        mask = torch.stack([get_mask(X_len[i], max_len=X.shape[-2]).float() for i in range(len(X))]).unsqueeze(-1)
        X = X * mask
        X = X.sum(-2)
        X = self.transformer_enc(X)
        X = X.sum(-2)

        X = self.dropout(self.dec(X))
        return X


class MultiWeightedDeepSets(ClassificationModel):
    """
    Config values:
        - multi_encoder: use a separate encoder for each persistence diagram
        - relu_before_pooling: apply relu before pooling to enforce positive element
    """
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        cfg = self.configs
        self.feature_name = params.dataset.graph_features.persistence_diagram.key
        self.has_freq = self.feature_name[:5] == "freq_"
        self.has_pd_embedding = "_non_embed" not in self.feature_name

        self.num_pds = len(self.feature_name.split('.'))
        self.pd_dim = 1 if self.has_pd_embedding else 3
        dim_input = 2
        dim_embed = 3 if self.has_pd_embedding else 0

        encoder_dims = [int(n.strip()) for n in str(cfg.encoder_dim).split(',')]
        if cfg.shared_encoder_dim:
            shared_encoder_dims = [int(n.strip()) for n in str(cfg.shared_encoder_dim).split(',')]
        else:
            shared_encoder_dims = []
        decoder_dims = [int(n.strip()) for n in str(cfg.decoder_dim).split(',')]

        if self.configs.append_embedding_to_encoder_output:
            decoder_dims[0] += dim_embed

        if self.configs.multi_encoder:
            self.enc = nn.ModuleList([MultiLinear(
                [dim_input] + encoder_dims,
                embed_dim=dim_embed,
                norm_layer=None,
                last_layer_activation_fn='relu',
                dropout=cfg.dropout) for _ in range(self.num_pds * self.pd_dim if self.configs.multi_encoder else 1)])
            encoder_output_dim = encoder_dims[-1] + (dim_embed if self.configs.append_embedding_to_encoder_output else 0)
        else:
            encoder_output_dim = dim_input

        if shared_encoder_dims:
            self.shared_enc = MultiLinear(
                [encoder_output_dim] + shared_encoder_dims,
                embed_dim=dim_embed,
                norm_layer=None,
                dropout=cfg.dropout)
        else:
            self.shared_enc = nn.Sequential()

        if self.configs.combine_encoder == "concat":
            encoder_output_dim = self.num_pds * encoder_output_dim
        elif self.configs.combine_encoder == "weight_pooling_2d":
            self.W = torch.nn.Parameter(torch.ones(self.num_pds, encoder_output_dim))
            self.register_parameter(name='weight', param=self.W)
        elif self.configs.combine_encoder == "weight_pooling":
            self.W = torch.nn.Parameter(torch.ones(self.num_pds))
            self.register_parameter(name='weight', param=self.W)
        elif self.configs.combine_encoder == "attention":
            self.attn = MultiheadAttention(
                embed_dim=encoder_output_dim,
                num_heads=cfg.attn_num_heads or 4,
                dropout=cfg.dropout)

        if not self.has_pd_embedding:
            self.W2 = torch.nn.Parameter(torch.rand(3))
            self.register_parameter(name='weight2', param=self.W2)

        self.dec = MultiLinear(
            decoder_dims,
            norm_layer=nn.BatchNorm1d,
            dropout=cfg.dropout)

    def forward(self, batch):
        X, X_len = self.to_cuda_tensors(batch.X)
        batch_size = len(X)
        if self.has_freq:
            X, freq = X[:, :, :, :-1], X[:, :, :, -1]
        if self.configs.append_embedding_to_encoder_output:
            X_emb = X[:, :, :, :-2]
            X = X[:, :, :, -2:]
        else:
            X_emb = None

        if self.configs.multi_encoder:
            X = torch.stack([self.enc[i](X[:, i, :, :], X_emb[:, i, :, :]) for i in range(self.num_pds * self.pd_dim)], dim=-3)
            # for i, module in enumerate(self.enc):
            #     logger.debug("%d: %s", i, str(module[0].weight.cpu().detach().numpy().tolist()))
            if self.configs.append_embedding_to_encoder_output:
                X = torch.cat([X, X_emb], -1)
            # else:
            #     X = X_enc

        # X = self.shared_enc(X, X_emb)

        # pooling points in pd
        mask = torch.stack([get_mask(X_len[i], max_len=X.shape[-2]).float() for i in range(len(X))]).unsqueeze(-1)
        if self.has_freq:
            X = X * freq.unsqueeze(-1)
        X = X * mask
        if self.configs.relu_before_pooling:
            X = X.relu()
        X = X.sum(-2)

        # combine encoder outputs
        if self.configs.combine_encoder == "concat":
            X = X.reshape([X.shape[0], -1])
        elif self.configs.combine_encoder == "pooling":
            X = X.sum(-2)
        elif self.configs.combine_encoder == "weight_pooling":
            if self.has_pd_embedding:
                weights = self.W

                if self.configs.weight_activation == "softmax":
                    weights = torch.softmax(self.W, -1)
                elif self.configs.weight_activation == "relu":
                    weights = torch.relu(self.W)

                X = X * weights.unsqueeze(0).repeat([batch_size, 1]).unsqueeze(-1)

                if self.configs.relu_before_pooling:
                    X = X.relu()
                # logger.debug(", ".join(["%.3f" % i for i in weights.tolist()]))
                X = X.sum(-2)
            else:
                X = X.reshape(X.shape[0], 3, -1, X.shape[-1])
                X = X * torch.softmax(self.W, -1).unsqueeze(0).unsqueeze(0).repeat([batch_size, 3, 1]).unsqueeze(-1)
                X = X.sum(-2)
                X = X * torch.softmax(self.W2, -1).unsqueeze(0).repeat([len(X), 1]).unsqueeze(-1)
                X = X.sum(-2)
        elif self.configs.combine_encoder == "weight_pooling_2d":
            weights = self.W

            if self.configs.weight_activation == "softmax":
                weights = torch.softmax(self.W, -1)
            elif self.configs.weight_activation == "relu":
                weights = torch.relu(self.W)
            X = X * weights.unsqueeze(0).repeat([len(X), 1, 1])
            if self.configs.relu_before_pooling:
                X = X.relu()
            X = X.sum(-2)
        elif self.configs.combine_encoder == "attention":
            X = X.transpose(0, 1)
            X, attn_weights = self.attn(X, X, X)
            X = X.transpose(0, 1)
            X = X.sum(-2)

        X = self.dec(X)
        return X


class WeightedDeepSets(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        cfg = self.configs
        dim_input = 2
        dim_hidden = params.model.hidden_dim
        dim_output = dataset.num_classes
        num_outputs = 1

        self.num_outputs = num_outputs
        self.dim_output = dim_output

        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden))
        self.weight = nn.Linear(2, 1)
        self.dec = linear_layers([dim_hidden + 2] * 4 + [num_outputs * dim_output])

    def forward(self, batch):
        X, X_len = batch.X['h0_h1_with_embeddings']
        X_mask = get_mask(X_len)
        w = self.weight(X[:, :, 2:]).squeeze(-1) * X_mask.float()

        # X = self.enc(X)
        X = torch.cat([X[:, :, :2], self.enc(X[:, :, 2:])], -1)
        X = (X * w.unsqueeze(-1)).sum(-2)
        X = self.dec(X).reshape(-1, self.dim_output)
        return X


class Encoder(nn.Module):
    def __init__(
            self, hidden_dim, num_heads, num_layers, dim_feedforward: int = 256,
             batch_norm: bool = True, dropout=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.enc = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout),
            num_layers,
            norm=nn.LayerNorm(hidden_dim) if batch_norm else None)

    def forward(self, X, mask=None):
        X = X.transpose(0, 1)
        X = self.enc(X, src_key_padding_mask=mask)
        return X.transpose(0, 1)


class SetTransformer(ClassificationModel):
    def __init__(self, params: Params, dataset):
        super(SetTransformer, self).__init__(params, dataset)
        cfg = self.configs

        self.emb = nn.Linear(2, cfg.encoder.dim_model - 2)
        self.enc = Encoder(
            hidden_dim=cfg.encoder.dim_model,
            num_heads=cfg.encoder.num_heads,
            num_layers=cfg.encoder.num_layers,
            dim_feedforward=cfg.encoder.dim_model * cfg.encoder.num_heads,
            dropout=cfg.dropout)

        self.linear_enc = linear_layers([cfg.encoder.dim_model] + params.model.dense_dim)
        self.linear_emb = linear_layers([cfg.encoder.dim_model - 2] + params.model.dense_dim)
        self.linear = nn.Linear([2 * cfg.dense_dim] + [cfg.dense_dim] * (cfg.num_classes - 1), dataset.num_classes)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch):
        X, X_len = batch.X['h0_h1_with_embeddings']
        X, X_topo_emb = X[:, :, 2:], X[:, :, :2]

        mask_h0 = get_mask(X_len)
        # X_h0 = torch.exp(-X_h0)
        X_emb = self.emb(X)
        X_emb = self.dropout(X_emb)
        X_enc = torch.cat([X_emb, X_topo_emb], -1)

        X_enc = self.enc(X_enc, ~mask_h0)
        X_enc = X_enc * mask_h0.float().unsqueeze(2)

        # X = torch.cat([X, X2], -1)
        # X = X2
        # X = self.dec(X, mask)
        # X_h0 = X_h0.mean(1)
        X_enc = X_enc.sum(1)
        X_emb = X_emb.sum(1)

        # X = F.softmax(self.linear(torch.cat([X_h0, X_h1], -1)), -1)
        X_enc = self.dropout(self.linear_enc(X_enc))
        X_emb = self.dropout(self.linear_emb(X_emb))
        X = F.softmax(self.linear(torch.cat([X_enc, X_emb], -1)), -1)
        return X
