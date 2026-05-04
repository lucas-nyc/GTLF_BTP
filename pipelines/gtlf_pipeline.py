import argparse
import copy
import glob
import hashlib
import json
import os
import pprint
import random
import shutil
import time

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

try:
    from torch_geometric.loader import DataLoader as GeoDataLoader
except Exception:
    GeoDataLoader = None

import config.config as cfg
from models.graph_branch import MultiBackboneLateFusionRegressor
from pipelines import baselines_pipeline as base_main
from utils import dataloader
from utils.train_utils import _build_fold_splits, next_run_dir


MODEL_PREFIX = "gtlf_btp"
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


def _canonical_tabular_backbone(name):
    key = str(name).strip().lower()
    aliases = {
        "ft-transformer": "ft_transformer",
        "fttransformer": "ft_transformer",
    }
    return aliases.get(key, key)


def _display_name(graph_backbone, tabular_backbone):
    g = GRAPH_DISPLAY[str(graph_backbone).strip().lower()]
    t = TABULAR_SUFFIX[_canonical_tabular_backbone(tabular_backbone)]
    return f"{g}_{t}"


def _rmse_mae(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    err = yp[mask] - yt[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def _reduce_preds(stack, reducer):
    arr = np.asarray(stack, dtype=float)
    if arr.ndim != 2:
        return np.asarray([], dtype=float)
    r = str(reducer).strip().lower()
    if r == "median":
        return np.nanmedian(arr, axis=0)
    return np.nanmean(arr, axis=0)


def _loss_fn(pred, y, loss_name="mse", beta=0.1):
    lname = str(loss_name).strip().lower()
    if lname in ("mae", "l1"):
        return F.l1_loss(pred, y)
    if lname in ("smooth_l1", "huber"):
        b = float(beta)
        if (not np.isfinite(b)) or b <= 0.0:
            b = 0.1
        return F.smooth_l1_loss(pred, y, beta=b)
    return F.mse_loss(pred, y)


def _has_nonempty_sets(list_of_sets):
    if not list_of_sets:
        return False
    for s in list_of_sets:
        if isinstance(s, list) and len(s) > 0:
            return True
    return False


def _flatten_named_set_groups(named_sets):
    flat = []
    if not isinstance(named_sets, dict):
        return flat
    for name in sorted(named_sets.keys(), key=lambda x: str(x)):
        sets = named_sets.get(name) or []
        for s in sets:
            if isinstance(s, list) and len(s) > 0:
                flat.append(s)
    return flat


def _flush_partial_results(all_results, run_dir):
    if not all_results:
        return
    partial_path = os.path.join(run_dir, "eval_per_set.partial.csv")
    pd.DataFrame(all_results).to_csv(partial_path, index=False)
    print(f"[PARTIAL] Saved {len(all_results)} row(s) to {partial_path}")


def _attach_run_log(run_dir):
    if not getattr(cfg, "SAVE_RUN_LOG", True):
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


def _stable_seed(base_seed, *parts):
    payload = "|".join([str(int(base_seed))] + [str(p) for p in parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int(int.from_bytes(digest, byteorder="little", signed=False) % 2147483647)


def _seed_everything(seed, deterministic=True, warn_only=True):
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


def _make_torch_generator(seed):
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _tensor_1d_or_default(val, length, dtype, default_value=0.0):
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


def _tensor_2d_or_default(val, rows, cols, dtype, default_value=0.0):
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


def _sanitize_graph_for_batching(g, target_idx=39, edge_attr_dim=3):
    if g is None or (not hasattr(g, "x")) or g.x is None:
        return g
    num_nodes = int(getattr(g, "num_nodes", g.x.shape[0]))
    num_edges = int(g.edge_index.shape[1]) if hasattr(g, "edge_index") and g.edge_index is not None and g.edge_index.dim() == 2 else 0

    coord = getattr(g, "coord", None)
    g.coord = _tensor_2d_or_default(coord, rows=num_nodes, cols=2, dtype=torch.float32, default_value=0.0)

    edge_attr = getattr(g, "edge_attr", None)
    g.edge_attr = _tensor_2d_or_default(edge_attr, rows=num_edges, cols=int(max(0, edge_attr_dim)), dtype=torch.float32, default_value=0.0)

    observed = getattr(g, "observed", None)
    if observed is None:
        obs = torch.ones((num_nodes,), dtype=torch.bool)
        ti = max(0, min(int(target_idx), num_nodes - 1))
        if 0 <= ti < num_nodes:
            obs[ti] = False
        g.observed = obs
    else:
        obs = torch.as_tensor(observed, dtype=torch.bool).view(-1)
        if int(obs.shape[0]) != num_nodes:
            out = torch.ones((num_nodes,), dtype=torch.bool)
            m = min(num_nodes, int(obs.shape[0]))
            if m > 0:
                out[:m] = obs[:m]
            g.observed = out
        else:
            g.observed = obs

    tabular = getattr(g, "tabular", None)
    if tabular is not None:
        g.tabular = _tensor_2d_or_default(tabular, rows=1, cols=num_nodes, dtype=torch.float32, default_value=-1.0)

    if hasattr(g, "raw_row") or getattr(g, "raw_row", None) is not None:
        g.raw_row = _tensor_1d_or_default(getattr(g, "raw_row", None), length=num_nodes, dtype=torch.float32, default_value=float("nan"))
    return g


def _sanitize_graph_list(graphs, target_idx=39, edge_attr_dim=3):
    if not isinstance(graphs, list):
        return graphs
    return [
        _sanitize_graph_for_batching(g, target_idx=target_idx, edge_attr_dim=edge_attr_dim)
        for g in graphs
    ]


def _make_loader(graphs, batch_size, shuffle, seed=None):
    graphs = _sanitize_graph_list(
        list(graphs),
        target_idx=int(getattr(cfg, "TARGET", 39)),
        edge_attr_dim=3,
    )
    if GeoDataLoader is None:
        if bool(shuffle):
            if seed is None:
                rng = np.random.RandomState()
            else:
                rng = np.random.RandomState(int(seed))
            order = rng.permutation(len(graphs)).tolist()
            return [graphs[i] for i in order]
        return graphs
    loader_kwargs = {
        "batch_size": int(max(1, batch_size)),
        "shuffle": bool(shuffle),
    }
    gen = _make_torch_generator(seed if bool(shuffle) else None)
    if gen is not None:
        loader_kwargs["generator"] = gen
    return GeoDataLoader(graphs, **loader_kwargs)


def _extract_target_from_batch(data, target_idx, num_nodes):
    y = torch.as_tensor(data.y, dtype=torch.float32, device=data.x.device)
    if y.dim() == 1:
        y = y.view(-1, 1)
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    dense_y, mask = to_dense_batch(y, batch, max_num_nodes=int(num_nodes))
    ti = max(0, min(int(target_idx), int(dense_y.shape[1]) - 1))
    target = dense_y[:, ti, 0]
    valid = mask[:, ti]
    if bool(valid.all().item()):
        return target
    fallback = dense_y[:, :, 0].sum(dim=1) / mask.sum(dim=1).to(dtype=dense_y.dtype).clamp_min(1.0)
    return torch.where(valid, target, fallback)


def _extract_global_targets_from_batch(data, num_nodes):
    y = torch.as_tensor(data.y, dtype=torch.float32, device=data.x.device)
    if y.dim() == 1:
        y = y.view(-1, 1)
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    dense_y, mask = to_dense_batch(y, batch, max_num_nodes=int(num_nodes))
    return dense_y[:, :, 0], mask


def _masked_global_loss(pred_dense, target_dense, mask, loss_name, smooth_l1_beta):
    if pred_dense.dim() != 2:
        pred_dense = pred_dense.view(target_dense.shape[0], -1)
    if target_dense.dim() != 2:
        target_dense = target_dense.view(pred_dense.shape[0], -1)
    valid = mask.bool()
    if pred_dense.shape != target_dense.shape:
        min_nodes = int(min(pred_dense.shape[1], target_dense.shape[1]))
        pred_dense = pred_dense[:, :min_nodes]
        target_dense = target_dense[:, :min_nodes]
        valid = valid[:, :min_nodes]
    if int(valid.sum().item()) <= 0:
        return pred_dense.sum() * 0.0
    return _loss_fn(pred_dense[valid], target_dense[valid], loss_name=loss_name, beta=smooth_l1_beta)


def _fusion_loss_from_outputs(
    outputs,
    target,
    global_target,
    global_mask,
    loss_name,
    smooth_l1_beta,
    fused_weight=1.0,
    graph_weight=1.0,
    tabular_weight=1.0,
):
    fused_loss = _loss_fn(outputs["fused"], target, loss_name=loss_name, beta=smooth_l1_beta)
    graph_loss = _masked_global_loss(
        outputs["graph_global"], global_target, global_mask, loss_name=loss_name, smooth_l1_beta=smooth_l1_beta
    )
    tabular_loss = _masked_global_loss(
        outputs["tabular_global"], global_target, global_mask, loss_name=loss_name, smooth_l1_beta=smooth_l1_beta
    )
    total = (
        (float(fused_weight) * fused_loss) +
        (float(graph_weight) * graph_loss) +
        (float(tabular_weight) * tabular_loss)
    )
    return {
        "total": total,
        "fused": fused_loss,
        "graph": graph_loss,
        "tabular": tabular_loss,
    }


def train_one_epoch(
    model,
    optimizer,
    graphs,
    device,
    target_idx,
    batch_size,
    loss_name,
    smooth_l1_beta,
    fused_loss_weight,
    graph_loss_weight,
    tabular_loss_weight,
    loader_seed=None,
):
    model.train()
    total = 0.0
    total_fused = 0.0
    total_graph = 0.0
    total_tabular = 0.0
    num_batches = 0
    loader = _make_loader(graphs, batch_size=batch_size, shuffle=True, seed=loader_seed)
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch, return_components=True)
        target = _extract_target_from_batch(batch, target_idx=target_idx, num_nodes=model.num_nodes).to(
            device=device, dtype=outputs["fused"].dtype
        )
        global_target, global_mask = _extract_global_targets_from_batch(batch, num_nodes=model.num_nodes)
        global_target = global_target.to(device=device, dtype=outputs["graph_global"].dtype)
        losses = _fusion_loss_from_outputs(
            outputs,
            target,
            global_target,
            global_mask,
            loss_name=loss_name,
            smooth_l1_beta=smooth_l1_beta,
            fused_weight=fused_loss_weight,
            graph_weight=graph_loss_weight,
            tabular_weight=tabular_loss_weight,
        )
        loss = losses["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        total_fused += float(losses["fused"].item())
        total_graph += float(losses["graph"].item())
        total_tabular += float(losses["tabular"].item())
        num_batches += 1
        denom = max(1, num_batches)
    return {
        "total": total / denom,
        "fused": total_fused / denom,
        "graph": total_graph / denom,
        "tabular": total_tabular / denom,
    }


def eval_one_epoch(
    model,
    graphs,
    device,
    target_idx,
    batch_size,
    loss_name,
    smooth_l1_beta,
    fused_loss_weight,
    graph_loss_weight,
    tabular_loss_weight,
):
    model.eval()
    total = 0.0
    total_fused = 0.0
    total_graph = 0.0
    total_tabular = 0.0
    num_batches = 0
    y_true = []
    y_pred = []
    loader = _make_loader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch, return_components=True)
            pred = outputs["fused"].view(-1)
            target = _extract_target_from_batch(batch, target_idx=target_idx, num_nodes=model.num_nodes).to(
                device=device, dtype=pred.dtype
            )
            global_target, global_mask = _extract_global_targets_from_batch(batch, num_nodes=model.num_nodes)
            global_target = global_target.to(device=device, dtype=outputs["graph_global"].dtype)
            losses = _fusion_loss_from_outputs(
                outputs,
                target,
                global_target,
                global_mask,
                loss_name=loss_name,
                smooth_l1_beta=smooth_l1_beta,
                fused_weight=fused_loss_weight,
                graph_weight=graph_loss_weight,
                tabular_weight=tabular_loss_weight,
            )
            loss = losses["total"]
            total += float(loss.item())
            total_fused += float(losses["fused"].item())
            total_graph += float(losses["graph"].item())
            total_tabular += float(losses["tabular"].item())
            num_batches += 1
            y_true.extend(target.detach().cpu().numpy().reshape(-1).tolist())
            y_pred.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
    rmse, mae = _rmse_mae(y_true, y_pred)
    denom = max(1, num_batches)
    return {
        "total": total / denom,
        "fused": total_fused / denom,
        "graph": total_graph / denom,
        "tabular": total_tabular / denom,
        "rmse": rmse,
        "mae": mae,
    }


def _predict_scalar_batch(model, graphs, device, batch_size):
    if not graphs:
        return np.asarray([], dtype=float)
    model.eval()
    preds = []
    loader = _make_loader(graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            preds.append(out.detach().cpu().numpy())
    if not preds:
        return np.asarray([], dtype=float)
    return np.concatenate(preds, axis=0)


def _build_model(model_kwargs):
    return MultiBackboneLateFusionRegressor(**model_kwargs)


def train_cv_fuse_model(
    graphs,
    run_dir,
    model_key,
    model_display,
    model_kwargs,
    n_splits,
    seed,
    torch_epochs,
    device,
    lr,
    weight_decay,
    patience,
    batch_size,
    loss_name,
    smooth_l1_beta,
    fused_loss_weight,
    graph_loss_weight,
    tabular_loss_weight,
    use_cv,
):
    os.makedirs(run_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)

    split_pairs = _build_fold_splits(
        n_samples=len(graphs),
        n_splits=n_splits,
        seed=seed,
        use_cv=bool(use_cv),
        holdout_ratio=float(getattr(cfg, "NO_CV_VAL_RATIO", 0.3)),
    )
    n_folds_used = int(len(split_pairs))
    target_idx = int(model_kwargs["target_node_idx"])
    train_verbose_gnn = bool(getattr(cfg, "TRAIN_VERBOSE_GNN", True))
    train_log_every = max(1, int(getattr(cfg, "TRAIN_LOG_EVERY", 5)))
    deterministic_training = bool(getattr(cfg, "DETERMINISTIC_TRAINING", True))
    deterministic_warn_only = bool(getattr(cfg, "DETERMINISTIC_WARN_ONLY", True))

    fold_states = []
    fold_paths = []
    cv_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(split_pairs, start=1):
        train_graphs = [graphs[i] for i in tr_idx]
        val_graphs = [graphs[i] for i in val_idx]
        fold_seed = _stable_seed(seed, model_key, f"fold{int(fold)}")
        _seed_everything(
            fold_seed,
            deterministic=deterministic_training,
            warn_only=deterministic_warn_only,
        )

        model = _build_model(model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, torch_epochs)) if torch_epochs > 1 else None

        best_val = float("inf")
        best_state = None
        pat = 0

        if train_verbose_gnn:
            print(
                f"[TRAIN][{model_key}] fold={fold}/{n_folds_used} start "
                f"train_graphs={len(train_graphs)} val_graphs={len(val_graphs)} "
                f"loss={loss_name} fused_target_w={float(fused_loss_weight):.3g} "
                f"graph_global_w={float(graph_loss_weight):.3g} tab_global_w={float(tabular_loss_weight):.3g}"
            )

        for ep in range(torch_epochs):
            train_stats = train_one_epoch(
                model,
                optimizer,
                train_graphs,
                device=device,
                target_idx=target_idx,
                batch_size=batch_size,
                loss_name=loss_name,
                smooth_l1_beta=smooth_l1_beta,
                fused_loss_weight=fused_loss_weight,
                graph_loss_weight=graph_loss_weight,
                tabular_loss_weight=tabular_loss_weight,
                loader_seed=_stable_seed(fold_seed, "epoch", int(ep)),
            )
            val_stats = eval_one_epoch(
                model,
                val_graphs,
                device=device,
                target_idx=target_idx,
                batch_size=batch_size,
                loss_name=loss_name,
                smooth_l1_beta=smooth_l1_beta,
                fused_loss_weight=fused_loss_weight,
                graph_loss_weight=graph_loss_weight,
                tabular_loss_weight=tabular_loss_weight,
            )
            val_loss = float(val_stats["total"])

            if scheduler is not None:
                scheduler.step()

            if np.isfinite(val_loss) and (val_loss < best_val):
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1

            if train_verbose_gnn and (((ep + 1) % train_log_every == 0) or ep == 0 or (ep + 1) == torch_epochs):
                print(
                    f"[TRAIN][{model_key}] fold={fold}/{n_folds_used} ep={ep+1}/{torch_epochs} "
                    f"train_total={train_stats['total']:.6g} train_fused={train_stats['fused']:.6g} "
                    f"train_graph={train_stats['graph']:.6g} train_tabular={train_stats['tabular']:.6g} "
                    f"val_total={val_loss:.6g} val_fused={val_stats['fused']:.6g} "
                    f"val_graph={val_stats['graph']:.6g} val_tabular={val_stats['tabular']:.6g} "
                    f"val_rmse={val_stats['rmse']:.6g} val_mae={val_stats['mae']:.6g} "
                    f"best={best_val:.6g} pat={pat}/{patience}"
                )

            if pat >= patience:
                if train_verbose_gnn:
                    print(f"[TRAIN][{model_key}] fold={fold}/{n_folds_used} early_stop ep={ep+1} best={best_val:.6g}")
                break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
            best_val = float("nan")
        else:
            model.load_state_dict(best_state)

        val_stats = eval_one_epoch(
            model,
            val_graphs,
            device=device,
            target_idx=target_idx,
            batch_size=batch_size,
            loss_name=loss_name,
            smooth_l1_beta=smooth_l1_beta,
            fused_loss_weight=fused_loss_weight,
            graph_loss_weight=graph_loss_weight,
            tabular_loss_weight=tabular_loss_weight,
        )
        cv_metrics.append({
            "fold": int(fold),
            "rmse": float(val_stats["rmse"]),
            "mae": float(val_stats["mae"]),
            "best_objective": float(best_val) if np.isfinite(best_val) else float("nan"),
        })

        checkpoint_record = {
            "state_dict": best_state,
            "model_kwargs": dict(model_kwargs),
            "model_name": model_key,
            "display_name": model_display,
            "loss_config": {
                "loss_name": str(loss_name),
                "smooth_l1_beta": float(smooth_l1_beta),
                "fused_loss_weight": float(fused_loss_weight),
                "graph_loss_weight": float(graph_loss_weight),
                "tabular_loss_weight": float(tabular_loss_weight),
            },
        }
        fname = os.path.join(model_dir, f"{model_key}_fold{fold:03d}.pt")
        torch.save(checkpoint_record, fname)
        fold_states.append(checkpoint_record)
        fold_paths.append(fname)

    return {
        "type": "graph_scalar_fusion",
        "models": fold_states,
        "paths": fold_paths,
        "model_name": model_key,
        "display_name": model_display,
        "model_kwargs": dict(model_kwargs),
        "cv_metrics": cv_metrics,
    }


def evaluate_fuse_ensemble(
    sets,
    scaler,
    category,
    ground_truth_graph,
    fold_checkpoints,
    model_kwargs,
    device,
    target_node_idx,
    ensemble_reducer="mean",
    batch_size=32,
):
    results = []
    loaded_checkpoints = []
    for s in fold_checkpoints:
        if isinstance(s, str):
            try:
                loaded = torch.load(s, map_location="cpu")
            except Exception:
                loaded = None
            if isinstance(loaded, dict) and "state_dict" in loaded:
                loaded_checkpoints.append(loaded)
            elif isinstance(loaded, dict):
                loaded_checkpoints.append({"state_dict": loaded, "model_kwargs": dict(model_kwargs)})
        elif isinstance(s, dict):
            if "state_dict" in s and isinstance(s["state_dict"], dict):
                loaded_checkpoints.append(s)
            else:
                loaded_checkpoints.append({"state_dict": s, "model_kwargs": dict(model_kwargs)})

    if not loaded_checkpoints:
        return []

    n_features = getattr(scaler, "n_features_in_", None)
    for set_idx, set_graphs in enumerate(sets, start=1):
        if not set_graphs:
            continue
        num_samples = len(set_graphs)
        preds_by_fold = []
        total_inference_time = 0.0
        valid_folds = 0

        for ckpt in loaded_checkpoints:
            ckpt_kwargs = dict(ckpt.get("model_kwargs") or model_kwargs)
            state_dict = ckpt.get("state_dict")
            if not isinstance(state_dict, dict):
                continue

            model = _build_model(ckpt_kwargs).to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            t0 = time.time()
            fold_preds = _predict_scalar_batch(model, set_graphs, device=device, batch_size=batch_size)
            t1 = time.time()
            if fold_preds.shape[0] != num_samples:
                continue

            preds_by_fold.append(fold_preds)
            total_inference_time += (t1 - t0)
            valid_folds += 1

        if valid_folds == 0:
            continue

        preds_stack = np.vstack(preds_by_fold)
        preds_agg = _reduce_preds(preds_stack, reducer=ensemble_reducer)

        y_true_norm = []
        for si in range(num_samples):
            gt = ground_truth_graph[si]
            y_true_norm.append(float(torch.as_tensor(gt.y).view(-1)[int(target_node_idx)].detach().cpu().item()))
        y_true_norm = np.asarray(y_true_norm, dtype=float)

        if n_features is not None:
            try:
                full_pred = np.zeros((num_samples, int(n_features)), dtype=float)
                full_true = np.zeros((num_samples, int(n_features)), dtype=float)
                full_pred[:, int(target_node_idx)] = preds_agg
                full_true[:, int(target_node_idx)] = y_true_norm
                y_pred_dn = scaler.inverse_transform(full_pred)[:, int(target_node_idx)]
                y_true_dn = scaler.inverse_transform(full_true)[:, int(target_node_idx)]
                rmse, mae = _rmse_mae(y_true_dn, y_pred_dn)
            except Exception:
                rmse, mae = _rmse_mae(y_true_norm, preds_agg)
        else:
            rmse, mae = _rmse_mae(y_true_norm, preds_agg)

        avg_inference_time = float(total_inference_time / valid_folds) if valid_folds > 0 else 0.0
        results.append({
            "Method_Name": model_kwargs.get("tabular_backbone", "fusion"),
            "Category": category,
            "Dataset_Index": int(set_idx),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Inference_Time": float(avg_inference_time),
        })

    return results


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Standalone multi-backbone fusion runner for scalar target prediction."
    )
    parser.add_argument("--no-cv", action="store_true", help="Run single-split training without K-fold CV.")
    parser.add_argument("--with-cv", action="store_true", help="Force K-fold CV even if config disables it.")
    parser.add_argument(
        "--graph-backbones",
        nargs="+",
        default=["graphsage", "gnn", "gin", "gat"],
        help="Graph backbones to use. Supported: graphsage, gnn, gin, gat.",
    )
    parser.add_argument(
        "--tabular-backbones",
        nargs="+",
        default=["mlp", "cnn1d", "linear", "resnet", "tabnet", "ft_transformer"],
        help="Tabular backbones to fuse. Supported: mlp, cnn1d, linear, resnet, tabnet, ft_transformer.",
    )
    parser.add_argument(
        "--imputation-methods",
        nargs="+",
        default=["CMILK"],
        help="Optional subset of imputation methods to load/evaluate.",
    )
    parser.add_argument("--graph-hidden", type=int, default=64, help="Hidden size for the graph branch.")
    parser.add_argument("--graph-layers", type=int, default=3, help="Number of graph encoder layers.")
    parser.add_argument("--tabular-hidden", type=int, default=128, help="Hidden size for the tabular branch.")
    parser.add_argument("--fusion-hidden", type=int, default=64, help="Hidden size for the shallow fusion head.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=int(getattr(cfg, "PATIENCE", 10)), help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for graph training/evaluation.")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae", "smooth_l1"], help="Scalar training loss.")
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1, help="Beta parameter when --loss smooth_l1.")
    parser.add_argument("--ensemble-reducer", type=str, default="mean", choices=["mean", "median"], help="Fold ensembling reducer.")
    parser.add_argument("--fused-loss-weight", type=float, default=1.0, help="Weight for the fused target loss.")
    parser.add_argument("--graph-loss-weight", type=float, default=1.0, help="Weight for the graph-branch global reconstruction loss.")
    parser.add_argument("--tabular-loss-weight", type=float, default=1.0, help="Weight for the tabular-branch global reconstruction loss.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate checkpoints from --pretrained-dir.")
    parser.add_argument(
        "--pretrained-dir",
        default=getattr(cfg, "GTLF_SAVE_DIR", os.path.join("save", "gtlf")),
        help="Directory containing pre-trained GTLF checkpoint subfolders.",
    )
    parser.add_argument(
        "--save-pretrained",
        action="store_true",
        help="After training, copy GTLF checkpoints/config into --pretrained-dir.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.imputation_methods:
        cfg.IMPUTATION_METHODS = list(args.imputation_methods)
        base_main.CANON_IMPUTE_MAP = {m.upper(): m for m in getattr(cfg, "IMPUTATION_METHODS", [])}
        print(f"[INFO] Imputed evaluation restricted to: {cfg.IMPUTATION_METHODS}")

    requested_graph_backbones = []
    for name in args.graph_backbones:
        key = str(name).strip().lower()
        if key not in GRAPH_DISPLAY:
            print(f"[WARN] Unsupported graph backbone '{name}', skipping.")
            continue
        if key not in requested_graph_backbones:
            requested_graph_backbones.append(key)
    if not requested_graph_backbones:
        raise RuntimeError("No supported graph backbones were requested.")

    requested_tabular_backbones = []
    for name in args.tabular_backbones:
        key = _canonical_tabular_backbone(name)
        if key not in TABULAR_SUFFIX:
            print(f"[WARN] Unsupported tabular backbone '{name}', skipping.")
            continue
        if key not in requested_tabular_backbones:
            requested_tabular_backbones.append(key)
    if not requested_tabular_backbones:
        raise RuntimeError("No supported tabular backbones were requested.")

    run_dir = next_run_dir(out_root=os.path.join(getattr(cfg, "OUT_PATH", "out")))
    print("Outputs & models saved to:", run_dir)
    _attach_run_log(run_dir)

    t0_global = time.time()
    run_without_cv = bool(getattr(cfg, "RUN_WITHOUT_CV", False))
    if bool(args.no_cv):
        run_without_cv = True
    if bool(args.with_cv):
        run_without_cv = False
    setattr(cfg, "RUN_WITHOUT_CV", bool(run_without_cv))
    if run_without_cv:
        print("[INFO] Running without K-fold CV (single holdout split).")
    else:
        print("[INFO] Running with K-fold CV.")
    print(
        "[INFO] Using fused target loss plus branch global reconstruction losses: "
        f"loss={str(args.loss)} fused_target_w={float(args.fused_loss_weight):.3g} "
        f"graph_global_w={float(args.graph_loss_weight):.3g} "
        f"tab_global_w={float(args.tabular_loss_weight):.3g}"
    )
    seed = int(getattr(cfg, "RANDOM_SEED", 42))
    deterministic_training = bool(getattr(cfg, "DETERMINISTIC_TRAINING", True))
    deterministic_warn_only = bool(getattr(cfg, "DETERMINISTIC_WARN_ONLY", True))
    _seed_everything(seed, deterministic=deterministic_training, warn_only=deterministic_warn_only)
    print(
        f"[INFO] Reproducibility enabled: seed={seed} "
        f"deterministic={deterministic_training} warn_only={deterministic_warn_only}"
    )

    print("Loading dataset...")
    outs = dataloader.load_dataset(cfg)
    (
        train_graphs,
        test_graphs,
        scaler,
        _edge_index,
        imputed_records,
        imputed_graph_sets_non,
        missing_records,
        missing_graph_sets_non,
        metadata,
    ) = base_main._safe_unpack_loader(outs)

    print("\n[LOADER SUMMARY]")
    for k, recs in (imputed_records or {}).items():
        try:
            print(f"  imputed: {k}: {len(recs)} set(s)")
        except Exception:
            print(f"  imputed: {k}: (count unknown)")
    for topk, scen_map in (missing_records or {}).items():
        print(f"  missing top={topk}: scen_count={len(scen_map.keys())}")
    print("End loader summary\n")

    if not train_graphs:
        raise RuntimeError("No training graphs were produced.")

    target_idx = int(getattr(cfg, "TARGET", 39))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_splits_cfg = int(getattr(cfg, "CV_SPLITS", 5))
    if run_without_cv:
        n_splits = 1
    else:
        n_splits = max(2, min(n_splits_cfg, int(len(train_graphs))))

    g0 = train_graphs[0]
    num_nodes = int(getattr(g0, "num_nodes", g0.x.shape[0]))
    in_channels = int(g0.x.shape[1])

    model_specs = []

    def _append_spec(model_key, model_display, model_kwargs, fused_weight, graph_weight, tabular_weight):
        model_specs.append({
            "key": model_key,
            "display": model_display,
            "kwargs": model_kwargs,
            "fused_loss_weight": float(fused_weight),
            "graph_loss_weight": float(graph_weight),
            "tabular_loss_weight": float(tabular_weight),
        })

    for graph_backbone in requested_graph_backbones:
        for tabular_backbone in requested_tabular_backbones:
            model_key = f"{MODEL_PREFIX}_{graph_backbone}_{tabular_backbone}"
            model_display = _display_name(graph_backbone, tabular_backbone)
            model_kwargs = {
                "in_channels": in_channels,
                "num_nodes": num_nodes,
                "target_node_idx": target_idx,
                "graph_backbone": graph_backbone,
                "graph_hidden": int(args.graph_hidden),
                "graph_layers": int(args.graph_layers),
                "tabular_backbone": tabular_backbone,
                "tabular_hidden": int(args.tabular_hidden),
                "fusion_hidden": int(args.fusion_hidden),
                "dropout": float(args.dropout),
            }
            _append_spec(
                model_key=model_key,
                model_display=model_display,
                model_kwargs=model_kwargs,
                fused_weight=float(args.fused_loss_weight),
                graph_weight=float(args.graph_loss_weight),
                tabular_weight=float(args.tabular_loss_weight),
            )

    config_path = os.path.join(run_dir, f"{MODEL_PREFIX}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_prefix": MODEL_PREFIX,
                "requested_graph_backbones": requested_graph_backbones,
                "requested_tabular_backbones": requested_tabular_backbones,
                "run_without_cv": bool(run_without_cv),
                "n_splits": int(n_splits),
                "seed": int(seed),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "patience": int(args.patience),
                "batch_size": int(args.batch_size),
                "loss": str(args.loss),
                "smooth_l1_beta": float(args.smooth_l1_beta),
                "fused_loss_weight": float(args.fused_loss_weight),
                "graph_loss_weight": float(args.graph_loss_weight),
                "tabular_loss_weight": float(args.tabular_loss_weight),
                "ensemble_reducer": str(args.ensemble_reducer),
                "models": model_specs,
            },
            f,
            indent=2,
        )
    print("Saved run config to:", config_path)

    trained = {}
    training_times = {}
    cv_rows = []

    if bool(args.eval_only):
        pretrained_dir = os.path.abspath(str(args.pretrained_dir))
        print(f"[INFO] Evaluation only: loading GTLF checkpoints from {pretrained_dir}")
        for spec in model_specs:
            model_dir = os.path.join(pretrained_dir, spec["key"])
            paths = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
            if not paths:
                print(f"[WARN] No checkpoints found for {spec['key']} in {model_dir}")
                continue
            trained[spec["key"]] = {
                "type": "graph_scalar_fusion",
                "models": paths,
                "paths": paths,
                "model_name": spec["key"],
                "display_name": spec["display"],
                "model_kwargs": dict(spec["kwargs"]),
                "cv_metrics": [],
            }
            training_times[spec["key"]] = np.nan
            print(f"[INFO] Loaded {len(paths)} checkpoint path(s) for {spec['display']}")
    else:
        for spec in model_specs:
            print(f"Training {spec['display']}...")
            t_train = time.time()
            res = train_cv_fuse_model(
                graphs=train_graphs,
                run_dir=run_dir,
                model_key=spec["key"],
                model_display=spec["display"],
                model_kwargs=spec["kwargs"],
                n_splits=n_splits,
                seed=seed,
                torch_epochs=int(args.epochs),
                device=device,
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                patience=int(args.patience),
                batch_size=int(args.batch_size),
                loss_name=str(args.loss),
                smooth_l1_beta=float(args.smooth_l1_beta),
                fused_loss_weight=float(spec.get("fused_loss_weight", args.fused_loss_weight)),
                graph_loss_weight=float(spec.get("graph_loss_weight", args.graph_loss_weight)),
                tabular_loss_weight=float(spec.get("tabular_loss_weight", args.tabular_loss_weight)),
                use_cv=(not run_without_cv),
            )
            trained[spec["key"]] = res
            training_times[spec["key"]] = time.time() - t_train
            for m in (res.get("cv_metrics") or []):
                cv_rows.append({
                    "Trained_Key": spec["key"],
                    "Method": spec["display"],
                    "Fold": int(m.get("fold")) if m.get("fold") is not None else np.nan,
                    "RMSE": float(m.get("rmse", np.nan)),
                    "MAE": float(m.get("mae", np.nan)),
                    "Best_Objective": float(m.get("best_objective", np.nan)) if m.get("best_objective") is not None else np.nan,
                    "Training_Time": float(training_times.get(spec["key"], np.nan)),
                })

        if bool(args.save_pretrained):
            pretrained_dir = os.path.abspath(str(args.pretrained_dir))
            os.makedirs(pretrained_dir, exist_ok=True)
            shutil.copy2(config_path, os.path.join(pretrained_dir, os.path.basename(config_path)))
            for spec in model_specs:
                src_dir = os.path.join(run_dir, spec["key"])
                dst_dir = os.path.join(pretrained_dir, spec["key"])
                if os.path.isdir(src_dir):
                    if os.path.isdir(dst_dir):
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
            print(f"[INFO] Saved GTLF pre-trained checkpoints to: {pretrained_dir}")

    all_results = []
    processed_impute_counts = defaultdict(int)
    processed_missing_counts = defaultdict(int)

    def _eval_and_append(spec, sets, ground_truth_graph, category, imputation_method=None):
        fold_states = trained[spec["key"]].get("models", [])
        fold_states = [fs for fs in (fold_states or []) if fs is not None]
        if len(fold_states) == 0:
            print(f"[WARN] no valid folds for {spec['key']}; skipping")
            return

        raw_rows = evaluate_fuse_ensemble(
            sets=sets,
            scaler=scaler,
            category=category,
            ground_truth_graph=ground_truth_graph,
            fold_checkpoints=fold_states,
            model_kwargs=trained[spec["key"]]["model_kwargs"],
            device=device,
            target_node_idx=target_idx,
            ensemble_reducer=str(args.ensemble_reducer),
            batch_size=int(args.batch_size),
        )
        if raw_rows:
            all_results.extend(
                base_main._normalize_eval_rows(
                    raw_rows,
                    base_method_name=spec["display"],
                    category=category,
                    imputation_method=imputation_method,
                    training_time=training_times.get(spec["key"], np.nan),
                )
            )
        else:
            print(f"[INFO] evaluate returned 0 rows for {spec['key']} / {category}")

    print("Evaluating original test graphs...")
    for spec in model_specs:
        _eval_and_append(spec, [test_graphs], test_graphs, "Original")
    _flush_partial_results(all_results, run_dir)

    print("Evaluating graph-based imputed graph-sets (PCC)...")
    filtered_imputed_non = {k: v for k, v in (imputed_graph_sets_non or {}).items() if _has_nonempty_sets(v)}
    if not filtered_imputed_non:
        print("[INFO] No imputed graph sets found after filtering.")
    for key, sets in filtered_imputed_non.items():
        parts = key.split("_", 1)
        category = parts[0] if len(parts) > 0 else ""
        impute_method = parts[1] if len(parts) > 1 else key
        print(f"[GRAPH][IMPUTED] {key}: evaluating {len(sets)} set list(s)")
        for spec in model_specs:
            _eval_and_append(spec, sets, test_graphs, category, imputation_method=impute_method)
        processed_impute_counts[base_main.canonical_imputation_name(impute_method)] += int(len(sets))
        _flush_partial_results(all_results, run_dir)

    print("Evaluating graph-based missing datasets...")
    filtered_missing_non = {"MNAR": {}, "MCAR": {}}
    for topk, scen_map in (missing_graph_sets_non or {}).items():
        for scen, sets in (scen_map or {}).items():
            if _has_nonempty_sets(sets):
                filtered_missing_non.setdefault(topk, {})[scen] = sets
    if not any(filtered_missing_non.values()):
        print("[INFO] No missing graph sets found after filtering.")
    for topk, scen_map in (filtered_missing_non or {}).items():
        flat_sets = _flatten_named_set_groups(scen_map)
        if not flat_sets:
            continue
        scen_names = ", ".join(sorted(str(k) for k in scen_map.keys()))
        print(
            f"[GRAPH][MISSING] {topk}: evaluating {len(flat_sets)} set list(s) "
            f"collapsed from scenarios=[{scen_names}]"
        )
        for spec in model_specs:
            _eval_and_append(spec, flat_sets, test_graphs, topk, imputation_method="Missing")
        processed_missing_counts[topk] += int(len(flat_sets))
        _flush_partial_results(all_results, run_dir)

    df_res = pd.DataFrame(all_results)
    per_set_csv = os.path.join(run_dir, "eval_per_set.csv")
    expected_cols = ["Method", "Imputation_Method", "Category", "Dataset_Index", "RMSE", "MAE", "Inference_Time", "Training_Time"]
    for col in expected_cols:
        if col not in df_res.columns:
            df_res[col] = np.nan
    for drop_col in ("Graph", "Method_Name"):
        if drop_col in df_res.columns:
            df_res = df_res.drop(columns=[drop_col])
    df_res.to_csv(per_set_csv, index=False)
    print("Saved per-set results to", per_set_csv)

    cv_cols = ["Trained_Key", "Method", "Fold", "RMSE", "MAE", "Best_Objective", "Training_Time"]
    cv_df = pd.DataFrame(cv_rows)
    for c in cv_cols:
        if c not in cv_df.columns:
            cv_df[c] = np.nan
    cv_path = os.path.join(run_dir, "cv_metrics.csv")
    cv_df.to_csv(cv_path, index=False)
    print("Saved CV metrics to", cv_path)

    print("\nImputation processing summary:")
    for im in getattr(cfg, "IMPUTATION_METHODS", []):
        cnt = processed_impute_counts.get(im, 0)
        print(f"  {im}: processed_sets={cnt}")

    print("\nMissing datasets processing summary:")
    for cat in ["MNAR", "MCAR"]:
        cnt = processed_missing_counts.get(cat, 0)
        print(f"  {cat}: processed_sets={cnt}")

    if df_res.empty:
        print("No results to summarize.")
    else:
        df_work = df_res.copy()
        if "Dataset_Index" in df_work.columns:
            df_work = df_work[~df_work["Dataset_Index"].isnull()].copy()
            df_work["Dataset_Index"] = df_work["Dataset_Index"].astype(int)
        if "Imputation_Method" not in df_work.columns:
            df_work["Imputation_Method"] = ""

        per_dataset_avg = df_work.groupby(["Method", "Imputation_Method", "Category", "Dataset_Index"], dropna=False).agg(
            RMSE_per_dataset=("RMSE", "mean"),
            MAE_per_dataset=("MAE", "mean"),
            IT_per_dataset=("Inference_Time", "mean"),
            TT_per_dataset=("Training_Time", "mean"),
        ).reset_index()

        def agg_stats(vals):
            if vals is None or len(vals) == 0:
                return (np.nan, np.nan)
            arr = np.asarray(vals, dtype=float)
            meanv = float(np.nanmean(arr))
            sdv = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            return (meanv, sdv)

        method_names = sorted(per_dataset_avg["Method"].dropna().astype(str).unique().tolist())
        out_rows = []
        known_imputes_upper = {m.upper() for m in getattr(cfg, "IMPUTATION_METHODS", [])}
        for method_name in method_names:
            method_mask = per_dataset_avg["Method"].str.upper() == method_name.upper()
            mask_orig = (per_dataset_avg["Category"].astype(str).str.upper() == "ORIGINAL") & method_mask
            orig_rmse = float(per_dataset_avg.loc[mask_orig, "RMSE_per_dataset"].astype(float).mean()) if mask_orig.any() else np.nan
            orig_mae = float(per_dataset_avg.loc[mask_orig, "MAE_per_dataset"].astype(float).mean()) if mask_orig.any() else np.nan
            orig_it = float(per_dataset_avg.loc[mask_orig, "IT_per_dataset"].astype(float).mean()) if mask_orig.any() else np.nan
            train_time = float(per_dataset_avg.loc[method_mask, "TT_per_dataset"].astype(float).mean()) if method_mask.any() else np.nan

            for impute in getattr(cfg, "IMPUTATION_METHODS", []):
                mask_mnar_missing = (
                    method_mask
                    & (per_dataset_avg["Category"].str.upper() == "MNAR")
                    & (~per_dataset_avg["Imputation_Method"].astype(str).str.upper().isin(known_imputes_upper))
                )
                mnar_miss_vals_rmse = per_dataset_avg.loc[mask_mnar_missing, "RMSE_per_dataset"].astype(float).values if mask_mnar_missing.any() else np.array([], dtype=float)
                mnar_miss_vals_mae = per_dataset_avg.loc[mask_mnar_missing, "MAE_per_dataset"].astype(float).values if mask_mnar_missing.any() else np.array([], dtype=float)
                mnar_miss_vals_it = per_dataset_avg.loc[mask_mnar_missing, "IT_per_dataset"].astype(float).values if mask_mnar_missing.any() else np.array([], dtype=float)
                mnar_miss_rmse_mean, mnar_miss_rmse_sd = agg_stats(mnar_miss_vals_rmse)
                mnar_miss_mae_mean, mnar_miss_mae_sd = agg_stats(mnar_miss_vals_mae)
                mnar_miss_it_mean, _ = agg_stats(mnar_miss_vals_it)

                mask_mnar_imp = (
                    method_mask
                    & (per_dataset_avg["Category"].str.upper() == "MNAR")
                    & (per_dataset_avg["Imputation_Method"].astype(str).str.upper() == impute.upper())
                )
                mnar_imp_vals_rmse = per_dataset_avg.loc[mask_mnar_imp, "RMSE_per_dataset"].astype(float).values if mask_mnar_imp.any() else np.array([], dtype=float)
                mnar_imp_vals_mae = per_dataset_avg.loc[mask_mnar_imp, "MAE_per_dataset"].astype(float).values if mask_mnar_imp.any() else np.array([], dtype=float)
                mnar_imp_vals_it = per_dataset_avg.loc[mask_mnar_imp, "IT_per_dataset"].astype(float).values if mask_mnar_imp.any() else np.array([], dtype=float)
                mnar_imp_rmse_mean, mnar_imp_rmse_sd = agg_stats(mnar_imp_vals_rmse)
                mnar_imp_mae_mean, mnar_imp_mae_sd = agg_stats(mnar_imp_vals_mae)
                mnar_imp_it_mean, _ = agg_stats(mnar_imp_vals_it)

                mask_mcar_missing = (
                    method_mask
                    & (per_dataset_avg["Category"].str.upper() == "MCAR")
                    & (~per_dataset_avg["Imputation_Method"].astype(str).str.upper().isin(known_imputes_upper))
                )
                mcar_miss_vals_rmse = per_dataset_avg.loc[mask_mcar_missing, "RMSE_per_dataset"].astype(float).values if mask_mcar_missing.any() else np.array([], dtype=float)
                mcar_miss_vals_mae = per_dataset_avg.loc[mask_mcar_missing, "MAE_per_dataset"].astype(float).values if mask_mcar_missing.any() else np.array([], dtype=float)
                mcar_miss_vals_it = per_dataset_avg.loc[mask_mcar_missing, "IT_per_dataset"].astype(float).values if mask_mcar_missing.any() else np.array([], dtype=float)
                mcar_miss_rmse_mean, mcar_miss_rmse_sd = agg_stats(mcar_miss_vals_rmse)
                mcar_miss_mae_mean, mcar_miss_mae_sd = agg_stats(mcar_miss_vals_mae)
                mcar_miss_it_mean, _ = agg_stats(mcar_miss_vals_it)

                mask_mcar_imp = (
                    method_mask
                    & (per_dataset_avg["Category"].str.upper() == "MCAR")
                    & (per_dataset_avg["Imputation_Method"].astype(str).str.upper() == impute.upper())
                )
                mcar_imp_vals_rmse = per_dataset_avg.loc[mask_mcar_imp, "RMSE_per_dataset"].astype(float).values if mask_mcar_imp.any() else np.array([], dtype=float)
                mcar_imp_vals_mae = per_dataset_avg.loc[mask_mcar_imp, "MAE_per_dataset"].astype(float).values if mask_mcar_imp.any() else np.array([], dtype=float)
                mcar_imp_vals_it = per_dataset_avg.loc[mask_mcar_imp, "IT_per_dataset"].astype(float).values if mask_mcar_imp.any() else np.array([], dtype=float)
                mcar_imp_rmse_mean, mcar_imp_rmse_sd = agg_stats(mcar_imp_vals_rmse)
                mcar_imp_mae_mean, mcar_imp_mae_sd = agg_stats(mcar_imp_vals_mae)
                mcar_imp_it_mean, _ = agg_stats(mcar_imp_vals_it)

                out_rows.append({
                    "Method": method_name,
                    "Imputation Method": impute,
                    "Training_Time": train_time,
                    "Original_RMSE": orig_rmse,
                    "Original_MAE": orig_mae,
                    "Original_Inference_Time": orig_it,
                    "MNAR_Missing_RMSE_Mean": mnar_miss_rmse_mean,
                    "MNAR_Missing_RMSE_SD": mnar_miss_rmse_sd,
                    "MNAR_Missing_MAE_Mean": mnar_miss_mae_mean,
                    "MNAR_Missing_MAE_SD": mnar_miss_mae_sd,
                    "MNAR_Missing_Inference_Time_Mean": mnar_miss_it_mean,
                    "MNAR_Imputed_RMSE_Mean": mnar_imp_rmse_mean,
                    "MNAR_Imputed_RMSE_SD": mnar_imp_rmse_sd,
                    "MNAR_Imputed_MAE_Mean": mnar_imp_mae_mean,
                    "MNAR_Imputed_MAE_SD": mnar_imp_mae_sd,
                    "MNAR_Imputed_Inference_Time_Mean": mnar_imp_it_mean,
                    "MCAR_Missing_RMSE_Mean": mcar_miss_rmse_mean,
                    "MCAR_Missing_RMSE_SD": mcar_miss_rmse_sd,
                    "MCAR_Missing_MAE_Mean": mcar_miss_mae_mean,
                    "MCAR_Missing_MAE_SD": mcar_miss_mae_sd,
                    "MCAR_Missing_Inference_Time_Mean": mcar_miss_it_mean,
                    "MCAR_Imputed_RMSE_Mean": mcar_imp_rmse_mean,
                    "MCAR_Imputed_RMSE_SD": mcar_imp_rmse_sd,
                    "MCAR_Imputed_MAE_Mean": mcar_imp_mae_mean,
                    "MCAR_Imputed_MAE_SD": mcar_imp_mae_sd,
                    "MCAR_Imputed_Inference_Time_Mean": mcar_imp_it_mean,
                })

        summary_df = pd.DataFrame(out_rows)
        out_summary_path = os.path.join(run_dir, "eval_summary_methods_wide.csv")
        summary_df.to_csv(out_summary_path, index=False)
        print("Saved wide method-level summary to:", out_summary_path)

    print("\n[FINAL DIAGNOSTIC SUMMARY]")
    if df_res.empty:
        print("No evaluation rows were produced (df_res empty).")
    else:
        combos = df_res.groupby(["Method"]).size().reset_index(name="count")
        print("Method counts (sample):")
        print(combos.head(60).to_string(index=False))

    print("Done. run dir:", run_dir)
    pprint.pprint({k: len(v["models"]) for k, v in trained.items() if isinstance(v, dict) and "models" in v})
    print("Total elapsed (s):", time.time() - t0_global)

    return run_dir


if __name__ == "__main__":
    main()
