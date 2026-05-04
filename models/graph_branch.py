import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from models.tabular_branch import build_tabular_encoder, canonical_tabular_backbone


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.out_dim = int(hidden_dim)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.input_bn = nn.BatchNorm1d(int(hidden_dim))
        self.convs = nn.ModuleList([
            SAGEConv(int(hidden_dim), int(hidden_dim), aggr="mean")
            for _ in range(int(max(1, num_layers)))
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(int(hidden_dim))
            for _ in range(int(max(1, num_layers)))
        ])
        self.dropout_p = float(dropout)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        for conv, bn in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h = h + h_new
        return h


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.out_dim = int(hidden_dim)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.input_bn = nn.BatchNorm1d(int(hidden_dim))
        self.convs = nn.ModuleList([
            GCNConv(int(hidden_dim), int(hidden_dim))
            for _ in range(int(max(1, num_layers)))
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(int(hidden_dim))
            for _ in range(int(max(1, num_layers)))
        ])
        self.dropout_p = float(dropout)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        for conv, bn in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h = h + h_new
        return h


class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3, heads=4):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(max(1, heads))
        self.per_head_hidden = int(max(1, (self.hidden_dim + self.heads - 1) // self.heads))
        self.out_hidden = int(self.per_head_hidden * self.heads)
        self.out_dim = int(self.out_hidden)
        self.input_proj = nn.Linear(int(input_dim), int(self.out_hidden))
        self.input_bn = nn.BatchNorm1d(int(self.out_hidden))
        self.convs = nn.ModuleList([
            GATConv(
                int(self.out_hidden),
                int(self.per_head_hidden),
                heads=self.heads,
                concat=True,
                dropout=float(dropout),
            )
            for _ in range(int(max(1, num_layers)))
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(int(self.out_hidden))
            for _ in range(int(max(1, num_layers)))
        ])
        self.dropout_p = float(dropout)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        for conv, bn in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h = h + h_new
        return h


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.out_dim = int(hidden_dim)
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.input_bn = nn.BatchNorm1d(int(hidden_dim))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(max(1, num_layers))):
            mlp = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(int(hidden_dim)))
        self.dropout_p = float(dropout)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)
        for conv, bn in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_p, training=self.training)
            h = h + h_new
        return h


def build_graph_encoder(backbone, input_dim, hidden_dim, num_layers, dropout):
    key = str(backbone).strip().lower()
    if key == "gnn":
        return GCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if key == "gat":
        return GATEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, heads=4)
    if key == "gin":
        return GINEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    return GraphSAGEEncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)


class MultiBackboneLateFusionRegressor(nn.Module):
    def __init__(
        self,
        in_channels,
        num_nodes,
        target_node_idx=39,
        graph_backbone="graphsage",
        graph_hidden=64,
        graph_layers=3,
        tabular_backbone="mlp",
        tabular_hidden=128,
        fusion_hidden=64,
        dropout=0.3,
    ):
        super().__init__()
        self.in_channels = int(max(1, in_channels))
        self.num_nodes = int(max(1, num_nodes))
        self.target_node_idx = int(max(0, target_node_idx))
        self.graph_backbone = str(graph_backbone).strip().lower()
        self.graph_hidden = int(max(8, graph_hidden))
        self.graph_layers = int(max(1, graph_layers))
        self.tabular_backbone = canonical_tabular_backbone(tabular_backbone)
        self.tabular_hidden = int(max(8, tabular_hidden))
        self.fusion_hidden = int(max(8, fusion_hidden))
        self.dropout_p = float(dropout)

        self.node_input_dim = self.in_channels + 2 + 1
        self.tabular_input_dim = int(max(1, self.num_nodes - 1))

        self.graph_encoder = None
        self.graph_proj = None
        self.graph_decoder = None
        self.tabular_encoder = None
        self.tabular_decoder = None
        self.head = None
        self.graph_encoder_out_dim = int(self.graph_hidden)

        self.graph_encoder = build_graph_encoder(
            backbone=self.graph_backbone,
            input_dim=self.node_input_dim,
            hidden_dim=self.graph_hidden,
            num_layers=self.graph_layers,
            dropout=self.dropout_p,
        )
        self.graph_encoder_out_dim = int(getattr(self.graph_encoder, "out_dim", self.graph_hidden))
        self.graph_proj = nn.Sequential(
            nn.Linear(2 * self.graph_encoder_out_dim, self.graph_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.graph_hidden, self.graph_hidden),
            nn.ReLU(),
        )
        self.graph_decoder = nn.Sequential(
            nn.Linear(self.graph_hidden, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, self.num_nodes),
        )
        self.tabular_encoder = build_tabular_encoder(
            backbone=self.tabular_backbone,
            input_dim=self.tabular_input_dim,
            hidden_dim=self.tabular_hidden,
            out_dim=self.graph_hidden,
            dropout=self.dropout_p,
        )
        self.tabular_decoder = nn.Sequential(
            nn.Linear(self.graph_hidden, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, self.num_nodes),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * self.graph_hidden, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for mod in (
            self.graph_encoder,
            self.graph_proj,
            self.graph_decoder,
            self.tabular_encoder,
            self.tabular_decoder,
            self.head,
        ):
            if mod is None:
                continue
            for layer in mod.modules():
                if hasattr(layer, "reset_parameters"):
                    try:
                        layer.reset_parameters()
                    except Exception:
                        pass

    def _target_idx(self, n_nodes):
        if n_nodes <= 0:
            return 0
        return max(0, min(int(self.target_node_idx), int(n_nodes) - 1))

    def _coord_tensor(self, data, n_nodes, device, dtype):
        coord = getattr(data, "coord", None)
        if coord is None:
            return torch.zeros((n_nodes, 2), dtype=dtype, device=device)
        coord_t = torch.as_tensor(coord, dtype=dtype, device=device)
        if coord_t.dim() != 2:
            coord_t = coord_t.view(n_nodes, -1)
        if coord_t.shape[1] >= 2:
            return coord_t[:, :2]
        if coord_t.shape[1] == 1:
            return torch.cat([coord_t, torch.zeros_like(coord_t)], dim=1)
        return torch.zeros((n_nodes, 2), dtype=dtype, device=device)

    def _observed_tensor(self, data, n_nodes, device, dtype):
        observed = getattr(data, "observed", None)
        if observed is None:
            obs = torch.ones((n_nodes, 1), dtype=dtype, device=device)
            ti = self._target_idx(n_nodes)
            obs[ti, 0] = 0.0
            return obs
        obs_t = torch.as_tensor(observed, dtype=dtype, device=device).view(-1, 1)
        if obs_t.shape[0] != n_nodes:
            obs = torch.ones((n_nodes, 1), dtype=dtype, device=device)
            ti = self._target_idx(n_nodes)
            obs[ti, 0] = 0.0
            return obs
        return obs_t

    def _tabular_tensor(self, data, batch_size, device, dtype):
        tab = getattr(data, "tabular", None)
        if tab is None:
            return torch.full((batch_size, self.num_nodes), -1.0, dtype=dtype, device=device)
        tab_t = torch.as_tensor(tab, dtype=dtype, device=device)
        if tab_t.dim() == 1:
            tab_t = tab_t.view(1, -1)
        return tab_t

    def _split_tabular_target(self, tabular):
        ti = self._target_idx(tabular.shape[1])
        left = tabular[:, :ti]
        right = tabular[:, ti + 1:]
        if left.numel() == 0:
            return right
        if right.numel() == 0:
            return left
        return torch.cat([left, right], dim=1)

    def forward(self, data, return_components=False):
        x = torch.as_tensor(data.x, dtype=torch.float32, device=data.x.device)
        edge_index = data.edge_index.to(device=x.device)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        num_nodes = int(self.num_nodes)
        ti = self._target_idx(num_nodes)

        coord = self._coord_tensor(data, n_nodes=x.shape[0], device=x.device, dtype=x.dtype)
        observed = self._observed_tensor(data, n_nodes=x.shape[0], device=x.device, dtype=x.dtype)
        node_input = torch.cat([x, coord, observed], dim=1)
        node_hidden = self.graph_encoder(node_input, edge_index)
        node_hidden_dense = node_hidden.view(batch_size, num_nodes, -1)
        target_repr = node_hidden_dense[:, ti, :]
        observed_mask = observed.view(batch_size, num_nodes, 1)
        observed_no_target = observed_mask.clone()
        observed_no_target[:, ti, :] = 0.0
        denom = observed_no_target.sum(dim=1).clamp_min(1.0)
        graph_mean = (node_hidden_dense * observed_no_target).sum(dim=1) / denom
        graph_repr = self.graph_proj(torch.cat([target_repr, graph_mean], dim=1))
        graph_recon = self.graph_decoder(graph_repr)

        tabular = self._tabular_tensor(data, batch_size=batch_size, device=x.device, dtype=x.dtype)
        tabular_wo_target = self._split_tabular_target(tabular)
        tabular_repr = self.tabular_encoder(tabular_wo_target)
        tabular_recon = self.tabular_decoder(tabular_repr)

        fused = torch.cat([graph_repr, tabular_repr], dim=1)
        pred = self.head(fused)

        if not return_components:
            return pred.view(-1)
        return {
            "fused": pred.view(-1),
            "graph_global": graph_recon,
            "tabular_global": tabular_recon,
            "graph_repr": graph_repr,
            "tabular_repr": tabular_repr,
        }
