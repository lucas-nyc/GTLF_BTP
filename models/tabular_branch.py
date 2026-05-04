import torch
import torch.nn as nn
import torch.nn.functional as F


def canonical_tabular_backbone(name):
    key = str(name).strip().lower()
    aliases = {
        "ft-transformer": "ft_transformer",
        "fttransformer": "ft_transformer",
    }
    return aliases.get(key, key)


class TabularMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TabularCNN1DEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        c1 = int(max(8, hidden_dim // 4))
        c2 = int(max(16, hidden_dim // 2))
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c2, int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.head(x)


class TabularLinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _TabularResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2, expansion=2):
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


class TabularResNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2, num_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.blocks = nn.ModuleList([
            _TabularResNetBlock(hidden_dim=int(hidden_dim), dropout=dropout, expansion=2)
            for _ in range(int(max(1, num_blocks)))
        ])
        self.out_norm = nn.BatchNorm1d(int(hidden_dim))
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        return self.out(h)


class TabularTabNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2, num_steps=4, gamma=1.3):
        super().__init__()
        self.gamma = float(max(1.0, gamma))
        self.num_steps = int(max(1, num_steps))
        self.input_bn = nn.BatchNorm1d(int(input_dim))
        self.initial_attn = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.maskers = nn.ModuleList([
            nn.Linear(int(hidden_dim), int(input_dim))
            for _ in range(self.num_steps)
        ])
        self.transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
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
        self.out = nn.Sequential(
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.input_bn(x)
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
        return self.out(decision / float(self.num_steps))


def _choose_transformer_heads(hidden_dim):
    h = int(max(1, hidden_dim))
    for cand in (8, 4, 2):
        if h >= cand and (h % cand) == 0:
            return cand
    return 1


class TabularFTTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.2, num_layers=2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_weight = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.feature_bias = nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.column_embedding = nn.Parameter(torch.empty(1, self.input_dim + 1, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=_choose_transformer_heads(self.hidden_dim),
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
            nn.Linear(self.hidden_dim, int(out_dim)),
            nn.ReLU(),
        )

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
        return self.output_proj(cls_repr)


def build_tabular_encoder(backbone, input_dim, hidden_dim, out_dim, dropout):
    key = canonical_tabular_backbone(backbone)
    if key not in ("mlp", "cnn1d", "linear", "resnet", "tabnet", "ft_transformer"):
        raise ValueError(
            f"Unsupported tabular backbone '{backbone}'. "
            "Supported tabular backbones are: mlp, cnn1d, linear, resnet, tabnet, ft_transformer."
        )
    if key == "resnet":
        return TabularResNetEncoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if key == "tabnet":
        return TabularTabNetEncoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if key == "ft_transformer":
        return TabularFTTransformerEncoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if key == "cnn1d":
        return TabularCNN1DEncoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    if key == "linear":
        return TabularLinearEncoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    return TabularMLPEncoder(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
