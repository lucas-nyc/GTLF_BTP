import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


def _coerce_hidden_dim(value, default=32):
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                item_i = int(item)
            except Exception:
                continue
            if item_i > 0:
                return item_i
        return int(default)
    try:
        value_i = int(value)
    except Exception:
        return int(default)
    return int(value_i) if value_i > 0 else int(default)


class _AugmentedNodeInputMixin:
    def _target_idx(self, n_nodes):
        if int(n_nodes) <= 0:
            return 0
        return max(0, min(int(getattr(self, "target_node_idx", 39)), int(n_nodes) - 1))

    def _extract_graph_args(self, data_or_x, edge_index=None, batch=None):
        data_obj = None
        if edge_index is None and (
            isinstance(data_or_x, PyGData) or
            (hasattr(data_or_x, "x") and hasattr(data_or_x, "edge_index"))
        ):
            data_obj = data_or_x
            x = data_obj.x
            edge_index = data_obj.edge_index
            if batch is None and hasattr(data_obj, "batch"):
                batch = getattr(data_obj, "batch")
        else:
            x = data_or_x
        if x is None or edge_index is None:
            raise ValueError("forward requires x and edge_index.")
        return x, edge_index, batch, data_obj

    def _coord_tensor(self, data_obj, n_nodes, device, dtype):
        if data_obj is not None:
            for name in ("coord", "coord_norm", "pos", "coord_px", "coord_pixels", "xy"):
                if hasattr(data_obj, name):
                    coord = getattr(data_obj, name)
                    if coord is None:
                        continue
                    coord_t = torch.as_tensor(coord, dtype=dtype, device=device)
                    if coord_t.dim() == 2 and coord_t.shape[0] == int(n_nodes):
                        if coord_t.shape[1] >= 2:
                            return coord_t[:, :2]
                        if coord_t.shape[1] == 1:
                            return torch.cat([coord_t, torch.zeros_like(coord_t)], dim=1)
        return torch.zeros((int(n_nodes), 2), dtype=dtype, device=device)

    def _observed_tensor(self, data_obj, n_nodes, device, dtype):
        observed = getattr(data_obj, "observed", None) if data_obj is not None else None
        if observed is None:
            obs = torch.ones((int(n_nodes), 1), dtype=dtype, device=device)
            ti = self._target_idx(n_nodes)
            if 0 <= ti < int(n_nodes):
                obs[ti, 0] = 0.0
            return obs
        obs_t = torch.as_tensor(observed, dtype=dtype, device=device).view(-1, 1)
        if obs_t.shape[0] != int(n_nodes):
            obs = torch.ones((int(n_nodes), 1), dtype=dtype, device=device)
            ti = self._target_idx(n_nodes)
            if 0 <= ti < int(n_nodes):
                obs[ti, 0] = 0.0
            return obs
        return obs_t

    def _augment_node_input(self, x, data_obj):
        coord = self._coord_tensor(data_obj, n_nodes=x.shape[0], device=x.device, dtype=x.dtype)
        observed = self._observed_tensor(data_obj, n_nodes=x.shape[0], device=x.device, dtype=x.dtype)
        return torch.cat([x, coord, observed], dim=1)


class _ResidualNodeRegressor(_AugmentedNodeInputMixin, nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=3, dropout=0.3, target_node_idx=39):
        super().__init__()
        self.in_channels = int(max(1, in_channels))
        self.hidden_channels = int(max(1, hidden_channels))
        self.out_channels = int(max(1, out_channels))
        self.num_layers = int(max(1, num_layers))
        self.target_node_idx = int(max(0, target_node_idx))
        self.input_dim = int(self.in_channels + 3)
        self.dropout_p = float(dropout)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_channels)
        self.input_bn = nn.BatchNorm1d(self.hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.output_head = nn.Linear(self.hidden_channels, self.out_channels)
        self.dropout = nn.Dropout(self.dropout_p)

    def _activation(self, h):
        return F.relu(h)

    def _apply_conv(self, conv, h, edge_index):
        return conv(h, edge_index)

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        self.input_bn.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for bn in self.norms:
            bn.reset_parameters()
        self.output_head.reset_parameters()

    def _encode(self, data_or_x, edge_index=None, batch=None):
        x, edge_index, batch, data_obj = self._extract_graph_args(
            data_or_x, edge_index=edge_index, batch=batch
        )
        x = x.to(dtype=torch.float32)
        edge_index = edge_index.to(device=x.device)

        node_input = self._augment_node_input(x, data_obj)
        h = self.input_proj(node_input)
        h = self.input_bn(h)
        h = self._activation(h)
        h = self.dropout(h)
        for conv, bn in zip(self.convs, self.norms):
            h_new = self._apply_conv(conv, h, edge_index)
            h_new = bn(h_new)
            h_new = self._activation(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new
        return h

    def forward(self, data_or_x, edge_index=None, batch=None):
        h = self._encode(data_or_x, edge_index=edge_index, batch=batch)
        out = self.output_head(h)
        if out.dim() == 1:
            out = out.view(-1, 1)
        return out


class GraphSAGE(_ResidualNodeRegressor):
    def __init__(
        self,
        in_channels,
        hid_channels=64,
        hidden_channels=None,
        out_channels=1,
        num_layers=3,
        dropout=0.3,
        aggr="mean",
        target_node_idx=39,
    ):
        hidden = _coerce_hidden_dim(
            hidden_channels if hidden_channels is not None else hid_channels,
            default=64,
        )
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            target_node_idx=target_node_idx,
        )
        self.aggr = aggr
        self.convs = nn.ModuleList([
            SAGEConv(self.hidden_channels, self.hidden_channels, aggr=self.aggr)
            for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
        self.reset_parameters()


class _ScalarPredictionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x):
        return self.net(x)


class TorchLinear(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3, pred_hidden=64):
        super().__init__()
        hidden = int(max(1, hidden_dim))
        self.encoder = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(),
        )
        self.pred_head = _ScalarPredictionHead(hidden, hidden_dim=int(max(1, pred_hidden)), dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.pred_head(z).view(-1, 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        return self


class TorchMLP(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, dropout=0.3, pred_hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden1)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.ReLU(),
        )
        self.pred_head = _ScalarPredictionHead(int(hidden2), hidden_dim=int(max(1, pred_hidden)), dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.pred_head(z).view(-1, 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        return self


class TorchCNN1D(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, dropout=0.3, kernel_size=3, pred_hidden=64):
        super().__init__()
        ks = kernel_size
        pad = ks // 2
        c1 = int(max(8, int(hidden_dim) // 4))
        c2 = int(max(16, int(hidden_dim) // 2))
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=ks, padding=pad),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=ks, padding=pad),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c2, int(latent_dim)),
            nn.ReLU(),
        )
        self.pred_head = _ScalarPredictionHead(int(latent_dim), hidden_dim=int(max(1, pred_hidden)), dropout=dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        z = self.encoder_head(x)
        return self.pred_head(z).view(-1, 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        return self


class _VectorResNetBlock(nn.Module):
    def __init__(self, hidden_dim, expansion=2, dropout=0.3):
        super().__init__()
        inner_dim = int(max(8, int(hidden_dim) * int(max(1, expansion))))
        self.norm = nn.BatchNorm1d(int(hidden_dim))
        self.fc1 = nn.Linear(int(hidden_dim), inner_dim)
        self.fc2 = nn.Linear(inner_dim, int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class TorchResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_blocks=3, expansion=2, dropout=0.3, latent_dim=64, pred_hidden=64):
        super().__init__()
        self.input_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.blocks = nn.ModuleList([
            _VectorResNetBlock(
                hidden_dim=int(hidden_dim),
                expansion=int(max(1, expansion)),
                dropout=float(dropout),
            )
            for _ in range(int(max(1, num_blocks)))
        ])
        self.out_norm = nn.BatchNorm1d(int(hidden_dim))
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.ReLU(),
        )
        self.pred_head = _ScalarPredictionHead(int(latent_dim), hidden_dim=int(max(1, pred_hidden)), dropout=dropout)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        z = self.output_proj(h)
        return self.pred_head(z).view(-1, 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        return self


class TorchTabNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_steps=4, gamma=1.3, dropout=0.3, latent_dim=64, pred_hidden=64):
        super().__init__()
        self.num_steps = int(max(1, num_steps))
        self.gamma = float(max(1.0, gamma))
        self.input_norm = nn.BatchNorm1d(int(in_dim))
        self.initial_attn = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.maskers = nn.ModuleList([
            nn.Linear(int(hidden_dim), int(in_dim))
            for _ in range(self.num_steps)
        ])
        self.transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(in_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            for _ in range(self.num_steps)
        ])
        self.step_norms = nn.ModuleList([
            nn.LayerNorm(int(hidden_dim))
            for _ in range(self.num_steps)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.ReLU(),
        )
        self.pred_head = _ScalarPredictionHead(int(latent_dim), hidden_dim=int(max(1, pred_hidden)), dropout=dropout)

    def forward(self, x):
        x = self.input_norm(x)
        prior = torch.ones_like(x)
        attn_state = self.initial_attn(x)
        decision = None
        for masker, transformer, norm in zip(self.maskers, self.transformers, self.step_norms):
            mask_logits = masker(attn_state)
            mask = F.softmax(mask_logits + torch.log(prior.clamp_min(1e-6)), dim=1)
            step_hidden = transformer(x * mask)
            step_hidden = norm(step_hidden)
            decision = step_hidden if decision is None else (decision + step_hidden)
            attn_state = step_hidden
            prior = (prior * (self.gamma - mask)).clamp_min(1e-6)
        z = self.output_proj(decision / float(self.num_steps))
        return self.pred_head(z).view(-1, 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        return self


def _choose_vector_transformer_heads(hidden_dim):
    h = int(max(1, hidden_dim))
    for cand in (8, 4, 2):
        if h >= cand and (h % cand) == 0:
            return cand
    return 1


class TorchFTTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.3, latent_dim=64, pred_hidden=64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_weight = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        self.feature_bias = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.column_embedding = nn.Parameter(torch.empty(1, self.in_dim + 1, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=_choose_vector_transformer_heads(self.hidden_dim),
            dim_feedforward=int(max(self.hidden_dim * 2, self.hidden_dim)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_kwargs = {"num_layers": int(max(1, num_layers))}
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                enable_nested_tensor=False,
                **encoder_kwargs,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, **encoder_kwargs)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, int(latent_dim)),
            nn.ReLU(),
        )
        self.pred_head = _ScalarPredictionHead(int(latent_dim), hidden_dim=int(max(1, pred_hidden)), dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.feature_weight)
        nn.init.zeros_(self.feature_bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.column_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.column_embedding[:, :tokens.shape[1], :]
        encoded = self.encoder(tokens)
        cls_repr = self.output_norm(encoded[:, 0, :])
        z = self.output_proj(cls_repr)
        return self.pred_head(z).view(-1, 1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        return self


class GNN(_ResidualNodeRegressor):
    def __init__(
        self,
        in_channels,
        hid_channels=64,
        hidden_channels=None,
        out_channels=1,
        num_layers=3,
        dropout=0.3,
        target_node_idx=39,
    ):
        hidden = _coerce_hidden_dim(
            hidden_channels if hidden_channels is not None else hid_channels,
            default=64,
        )
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            target_node_idx=target_node_idx,
        )
        self.convs = nn.ModuleList([
            GCNConv(self.hidden_channels, self.hidden_channels)
            for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
        self.reset_parameters()


class TorchGAT(_ResidualNodeRegressor):
    def __init__(
        self,
        in_channels,
        hid_channels=64,
        hidden_channels=None,
        out_channels=1,
        aggr="mean",
        heads=4,
        dropout=0.3,
        num_layers=3,
        target_node_idx=39,
    ):
        requested_hidden = _coerce_hidden_dim(
            hidden_channels if hidden_channels is not None else hid_channels,
            default=64,
        )
        self.heads = int(max(1, heads))
        self.per_head_hidden = int(max(1, (requested_hidden + self.heads - 1) // self.heads))
        out_hidden = int(self.per_head_hidden * self.heads)
        super().__init__(
            in_channels=in_channels,
            hidden_channels=out_hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            target_node_idx=target_node_idx,
        )
        self.aggr = aggr
        self.convs = nn.ModuleList([
            GATConv(
                self.hidden_channels,
                self.per_head_hidden,
                heads=self.heads,
                concat=True,
                dropout=self.dropout_p,
            )
            for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
        self.reset_parameters()

    def _activation(self, h):
        return F.elu(h)


class TorchGIN(_ResidualNodeRegressor):
    def __init__(
        self,
        in_channels,
        hid_channels=64,
        hidden_channels=None,
        out_channels=1,
        aggr="mean",
        dropout=0.3,
        num_layers=3,
        target_node_idx=39,
    ):
        hidden = _coerce_hidden_dim(
            hidden_channels if hidden_channels is not None else hid_channels,
            default=64,
        )
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            target_node_idx=target_node_idx,
        )
        self.aggr = aggr
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.ReLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(self.hidden_channels))
        self.reset_parameters()
