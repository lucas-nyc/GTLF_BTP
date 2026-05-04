import hashlib
import os
import random
import re

import numpy as np
import pandas as pd
import torch


DEFAULT_BACKBONE_KEYS = [
    "linear", "mlp", "cnn1d", "resnet_mlp", "tabnet", "ft_transformer",
    "gnn", "graphsage", "gat", "gin",
]
DEFAULT_BACKBONE_DISPLAY = {
    "linear": "LINEAR",
    "mlp": "MLP",
    "cnn1d": "CNN",
    "resnet_mlp": "RESNET_MLP",
    "tabnet": "TABNET",
    "ft_transformer": "FT_TRANSFORMER",
    "gnn": "GNN",
    "graphsage": "GRAPHSAGE",
    "gat": "GAT",
    "gin": "GIN",
}
GRAPH_DISPLAY = {
    "graphsage": "GRAPHSAGE",
    "gnn": "GNN_HYBRID",
    "gin": "GIN_HYBRID",
    "gat": "GAT_HYBRID",
}
TABULAR_SUFFIX = {
    "mlp": "MLP",
    "cnn1d": "CNN1D",
    "linear": "LINEAR",
    "resnet": "RESNET",
    "tabnet": "TABNET",
    "ft_transformer": "FT_TRANSFORMER",
}
_BACKBONE_KEY_ALIASES = {"sklearn_lr": "linear"}

GRAPH_BASELINE_CANDIDATES = {
    "GRAPHSAGE": ["GRAPHSAGE"],
    "GNN_HYBRID": ["GNN"],
    "GAT_HYBRID": ["GAT"],
    "GIN_HYBRID": ["GIN"],
}
TABULAR_BASELINE_CANDIDATES = {
    "MLP": ["MLP"],
    "CNN1D": ["CNN1D", "CNN"],
    "LINEAR": ["LINEAR"],
    "RESNET": ["RESNET_MLP", "RESNET"],
    "TABNET": ["TABNET"],
    "FT_TRANSFORMER": ["FT_TRANSFORMER"],
}
_RAW_MISSING_LABELS = {
    "MISSING",
    "FACE_MASK",
    "GLASSES",
    "NON_OCCLUDED",
    "10_PERCENT",
    "20_PERCENT",
    "30_PERCENT",
    "40_PERCENT",
    "50_PERCENT",
    "60_PERCENT",
    "70_PERCENT",
    "80_PERCENT",
    "90_PERCENT",
}


def canonical_backbone_key(key):
    s = str(key).strip().lower()
    return _BACKBONE_KEY_ALIASES.get(s, s)


def canonical_tabular_backbone(name):
    key = str(name).strip().lower()
    aliases = {
        "ft-transformer": "ft_transformer",
        "fttransformer": "ft_transformer",
    }
    return aliases.get(key, key)


def display_name(key, backbone_display=None):
    ckey = canonical_backbone_key(key)
    display_map = dict(DEFAULT_BACKBONE_DISPLAY if backbone_display is None else backbone_display)
    return display_map.get(ckey, str(ckey).upper())


def fusion_display_name(graph_backbone, tabular_backbone):
    g = GRAPH_DISPLAY[str(graph_backbone).strip().lower()]
    t = TABULAR_SUFFIX[canonical_tabular_backbone(tabular_backbone)]
    return f"{g}_{t}"


def canonical_imputation_name(name, canon_map=None):
    if name is None:
        return ""
    s = str(name).strip()
    if s == "":
        return ""
    su = s.upper().replace(" ", "_").replace("-", "_")
    canon_map = {k.upper(): v for k, v in (canon_map or {}).items()}
    if su in canon_map:
        return canon_map[su]
    su_short = su.replace("_IMPUTATION", "")
    if su_short in canon_map:
        return canon_map[su_short]
    for k, v in canon_map.items():
        if k in su or su in k:
            return v
    return su


def normalize_eval_rows(rows, base_method_name, category, canon_map, imputation_method=None, training_time=np.nan):
    for row in rows:
        row["Method"] = base_method_name
        row["Category"] = category
        if imputation_method is not None:
            row["Imputation_Method"] = canonical_imputation_name(imputation_method, canon_map)
        else:
            im = row.get("Imputation_Method", "")
            row["Imputation_Method"] = canonical_imputation_name(im, canon_map) if im else ""
        row.pop("Graph", None)
        row.pop("Method_Name", None)
        row["Training_Time"] = float(training_time) if training_time is not None else np.nan
    return rows


def safe_unpack_loader(outs):
    if not isinstance(outs, (list, tuple)):
        raise RuntimeError("dataloader.load_dataset returned unexpected type")
    ln = len(outs)
    if ln == 9:
        return outs
    if ln == 10:
        (
            train_graphs,
            _val_graphs,
            test_graphs,
            scaler,
            edge_index,
            imputed_records,
            imputed_graph_sets,
            missing_records,
            missing_graph_sets,
            metadata,
        ) = outs
        return (
            train_graphs,
            test_graphs,
            scaler,
            edge_index,
            imputed_records,
            imputed_graph_sets,
            missing_records,
            missing_graph_sets,
            metadata,
        )
    raise RuntimeError(f"Unsupported loader return signature length={ln}")


def safe_mean(vals):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan
    return float(np.nanmean(arr))


def rmse_mae(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    err = yp[mask] - yt[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def reduce_preds(stack, reducer):
    arr = np.asarray(stack, dtype=float)
    if arr.ndim != 2:
        return np.asarray([], dtype=float)
    r = str(reducer).strip().lower()
    if r == "median":
        return np.nanmedian(arr, axis=0)
    return np.nanmean(arr, axis=0)


def has_nonempty_sets(list_of_sets):
    if not list_of_sets:
        return False
    for item in list_of_sets:
        if isinstance(item, list) and len(item) > 0:
            return True
    return False


def flatten_named_set_groups(named_sets):
    flat = []
    if not isinstance(named_sets, dict):
        return flat
    for name in sorted(named_sets.keys(), key=lambda x: str(x)):
        sets = named_sets.get(name) or []
        for item in sets:
            if isinstance(item, list) and len(item) > 0:
                flat.append(item)
    return flat


def flush_partial_results(all_results, run_dir):
    if not all_results:
        return
    partial_path = os.path.join(run_dir, "eval_per_set.partial.csv")
    pd.DataFrame(all_results).to_csv(partial_path, index=False)
    print(f"[PARTIAL] Saved {len(all_results)} row(s) to {partial_path}")


def attach_run_log(run_dir, enabled=True):
    if not enabled:
        return
    import sys

    log_path = os.path.join(run_dir, "run_log.txt")
    log_f = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                try:
                    f.write(data)
                except Exception:
                    pass

        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except Exception:
                    pass

    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"[INFO] Run log is being recorded to: {log_path}")


def stable_seed(base_seed, *parts):
    payload = "|".join([str(int(base_seed))] + [str(p) for p in parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int(int.from_bytes(digest, byteorder="little", signed=False) % 2147483647)


def seed_everything(seed, deterministic=True, warn_only=True):
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    return s


def make_torch_generator(seed):
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def tensor_1d_or_default(val, length, dtype, default_value=0.0):
    if val is None:
        return torch.full((int(length),), float(default_value), dtype=dtype)
    try:
        t = torch.as_tensor(val, dtype=dtype).view(-1)
    except Exception:
        return torch.full((int(length),), float(default_value), dtype=dtype)
    if int(t.shape[0]) == int(length):
        return t
    out = torch.full((int(length),), float(default_value), dtype=dtype)
    m = min(int(length), int(t.shape[0]))
    if m > 0:
        out[:m] = t[:m]
    return out


def tensor_2d_or_default(val, rows, cols, dtype, default_value=0.0):
    if val is None:
        return torch.full((int(rows), int(cols)), float(default_value), dtype=dtype)
    try:
        t = torch.as_tensor(val, dtype=dtype)
    except Exception:
        return torch.full((int(rows), int(cols)), float(default_value), dtype=dtype)
    if t.dim() != 2:
        t = t.view(int(rows), -1) if t.numel() > 0 else torch.empty((int(rows), 0), dtype=dtype)
    out = torch.full((int(rows), int(cols)), float(default_value), dtype=dtype)
    r = min(int(rows), int(t.shape[0]))
    c = min(int(cols), int(t.shape[1]) if t.dim() == 2 else 0)
    if r > 0 and c > 0:
        out[:r, :c] = t[:r, :c]
    return out


def resolve_first_available(candidates, available):
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def parse_fuse_method(method_name):
    m = re.match(r"^(GRAPHSAGE|GNN_HYBRID|GAT_HYBRID|GIN_HYBRID)_(MLP|CNN1D|LINEAR|RESNET|TABNET|FT_TRANSFORMER)$", str(method_name).strip().upper())
    if not m:
        return None
    return m.group(1), m.group(2)


def is_missing_imputation_label(label):
    return str(label).strip().upper() in _RAW_MISSING_LABELS


def pretty_graph(method_name):
    mapping = {
        "GNN_HYBRID": "GNN",
        "GAT_HYBRID": "GAT",
        "GIN_HYBRID": "GIN",
        "GRAPHSAGE": "GraphSAGE",
        "GNN": "GNN",
        "GAT": "GAT",
        "GIN": "GIN",
    }
    return mapping.get(str(method_name).strip().upper(), str(method_name))


def pretty_tabular(method_name):
    mapping = {
        "MLP": "MLP",
        "CNN1D": "CNN1D",
        "LINEAR": "Linear",
        "RESNET": "ResNet",
        "RESNET_MLP": "ResNet",
        "TABNET": "TabNet",
        "FT_TRANSFORMER": "FT-Transformer",
    }
    return mapping.get(str(method_name).strip().upper(), str(method_name))


def pretty_fused_label(method_name):
    parsed = parse_fuse_method(method_name)
    if not parsed:
        return str(method_name)
    graph_prefix, tabular_suffix = parsed
    return f"{pretty_graph(graph_prefix)} + {pretty_tabular(tabular_suffix)}"
