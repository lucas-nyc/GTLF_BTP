# utils/train_utils.py
import os
import copy
import numpy as np
import time
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import config.config as cfg

# import the active baseline/fusion-support model classes
from models.baseline_models import (
    TorchLinear,
    TorchMLP,
    TorchCNN1D,
    TorchResNetMLP,
    TorchTabNet,
    TorchFTTransformer,
    GNN,
    GraphSAGE,
    TorchGAT,
    TorchGIN
)

# ----------------------
# Small helper dataset for vector models
# ----------------------
class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.reshape(-1, 1).astype('float32')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def _build_fold_splits(n_samples, n_splits=5, seed=42, use_cv=True, holdout_ratio=0.2):
    """
    Build train/val index splits.
    - use_cv=True: KFold splits (>=2 folds).
    - use_cv=False: one shuffled holdout split.
    """
    n_samples = int(max(0, n_samples))
    if n_samples <= 0:
        return []

    if bool(use_cv):
        k = int(max(2, min(int(n_splits), n_samples)))
        kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))
        return list(kf.split(np.arange(n_samples)))

    idx = np.arange(n_samples, dtype=int)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(idx)

    if n_samples == 1:
        return [(idx.copy(), idx.copy())]

    ratio = float(holdout_ratio)
    if not np.isfinite(ratio):
        ratio = 0.2
    ratio = float(min(max(ratio, 0.05), 0.5))
    val_size = int(round(n_samples * ratio))
    val_size = max(1, min(val_size, n_samples - 1))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    if tr_idx.size == 0:
        tr_idx = idx.copy()
    return [(tr_idx, val_idx)]


def _forward_graph_model(model, data, device):
    """
    Robust forward helper for graph models. Returns numpy vector per node.
    """
    model.eval()
    with torch.no_grad():
        try:
            out = model(data.to(device))
        except Exception:
            try:
                out = model(data.x.to(device), data.edge_index.to(device))
            except Exception:
                out = model(data.x.to(device))
        if isinstance(out, torch.Tensor):
            return out.detach().cpu().numpy().reshape(-1)
        try:
            return torch.as_tensor(out[0]).detach().cpu().numpy().reshape(-1)
        except Exception as e:
            raise RuntimeError(f"Unsupported model forward output type: {type(out)} ({e})")

# ----------------------
# Run dir helper
# ----------------------
def next_run_dir(out_root="out"):
    os.makedirs(out_root, exist_ok=True)
    existing = [d for d in os.listdir(out_root) if d.startswith("run")]
    nums = []
    for d in existing:
        try:
            n = int(d.replace("run", ""))
            nums.append(n)
        except Exception:
            pass
    nxt = max(nums) + 1 if nums else 1
    run_dir = os.path.join(out_root, f"run{nxt:03d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# ----------------------
# Torch trainer for vector models (MLP, CNN1D)
# ----------------------
def train_torch_model(model, train_loader, val_loader, device='cpu', epochs=50, lr=1e-3, weight_decay=0.0,
                      verbose=False, log_every=5, model_name="torchmodel", fold=None, n_folds=None):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_losses.append(float(loss.item()))
        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))
        if len(val_losses) == 0:
            continue
        avg_train = float(np.mean(tr_losses)) if len(tr_losses) > 0 else float('nan')
        avg_val = float(np.mean(val_losses))
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (((ep + 1) % max(1, int(log_every)) == 0) or ep == 0 or (ep + 1) == epochs):
            fold_txt = f" fold={fold}/{n_folds}" if (fold is not None and n_folds is not None) else ""
            print(f"[TRAIN][{model_name}]{fold_txt} ep={ep+1}/{epochs} train_mse={avg_train:.6g} val_mse={avg_val:.6g} best_val={best_val:.6g}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def _pointwise_regression_loss_vec(err, loss_name="mse", smooth_l1_beta=0.1):
    lname = str(loss_name).strip().lower()
    if lname in ("smooth_l1", "huber"):
        beta = float(smooth_l1_beta)
        if not np.isfinite(beta) or beta <= 0.0:
            beta = 0.1
        abs_err = torch.abs(err)
        return torch.where(abs_err < beta, 0.5 * (err ** 2) / beta, abs_err - 0.5 * beta)
    if lname in ("mae", "l1"):
        return torch.abs(err)
    return err ** 2


def _nodewise_regression_loss(
    out_flat,
    y_flat,
    target_node_idx=39,
    target_loss_weight=1.0,
    target_only_loss=False,
    loss_name="mse",
    smooth_l1_beta=0.1,
):
    """
    Per-graph node regression loss with optional target-only or target-weighted objective.
    """
    out_flat = out_flat.view(-1)
    y_flat = y_flat.view(-1)
    n = int(y_flat.shape[0])
    try:
        ti = int(target_node_idx)
    except Exception:
        ti = -1

    err = out_flat - y_flat
    per_node_loss = _pointwise_regression_loss_vec(
        err, loss_name=loss_name, smooth_l1_beta=smooth_l1_beta
    )

    if target_only_loss and (0 <= ti < n):
        return torch.mean(per_node_loss[ti:ti+1])

    if (0 <= ti < n) and (float(target_loss_weight) != 1.0):
        w = torch.ones_like(per_node_loss)
        w[ti] = float(target_loss_weight)
        return torch.sum(w * per_node_loss) / torch.sum(w)
    return torch.mean(per_node_loss)


def train_single_gnn_epoch(model, optimizer, train_graphs, device,
                           target_node_idx=39, target_loss_weight=1.0,
                           target_only_loss=False, return_breakdown=False,
                           loss_name="mse", smooth_l1_beta=0.1):
    """
    Robust per-epoch training for graph models.
    Tries calling model(data) first (for GraphSAGE that expects a Data object),
    falls back to model(x, edge_index) if that fails (for GNN that accepts x, edge_index).
    Returns: average training loss over graphs (float)
    """
    model.train()
    total_loss = 0.0
    total_pred = 0.0

    for data in train_graphs:
        data = data.to(device)

        optimizer.zero_grad()

        try:
            out = model(data)                      # preferred for GraphSAGE-style forward(self, data)
        except Exception:
            try:
                out = model(data.x, data.edge_index)  # alternative (x, edge_index)
            except Exception:
                out = model(data.x)

        # ensure tensor shape is compatible
        # out might be (num_nodes, 1) or (num_nodes,) -> flatten to (num_nodes,)
        if isinstance(out, torch.Tensor):
            out_flat = out.view(-1)
        else:
            # If model returned tuple (out, ..) or list-like, grab first element
            try:
                out_flat = torch.as_tensor(out[0]).view(-1).to(device)
            except Exception:
                raise RuntimeError("Model forward returned unsupported type in train_single_gnn_epoch")

        pred_loss = _nodewise_regression_loss(
            out_flat, data.y.view(-1),
            target_node_idx=target_node_idx,
            target_loss_weight=target_loss_weight,
            target_only_loss=target_only_loss,
            loss_name=loss_name,
            smooth_l1_beta=smooth_l1_beta
        )
        loss = pred_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_pred += float(pred_loss.item())

    denom = max(1, len(train_graphs))
    out_stats = {
        "total": total_loss / denom,
        "pred": total_pred / denom,
    }
    if return_breakdown:
        return out_stats
    return out_stats["total"]


def eval_gnn_on_graphs(model, val_graphs, device,
                       target_node_idx=39, target_loss_weight=1.0,
                       target_only_loss=False, return_breakdown=False,
                       loss_name="mse", smooth_l1_beta=0.1):
    """
    Robust evaluation for GNN-like models on a list of Data objects.
    Tries model(data) first then falls back to model(x, edge_index).
    """
    model.eval()
    tot = 0.0
    tot_pred = 0.0

    with torch.no_grad():
        for data in val_graphs:
            data = data.to(device)

            try:
                out = model(data)
            except Exception:
                try:
                    out = model(data.x, data.edge_index)
                except Exception:
                    out = model(data.x)

            if isinstance(out, torch.Tensor):
                out_flat = out.view(-1)
            else:
                try:
                    out_flat = torch.as_tensor(out[0]).view(-1).to(device)
                except Exception:
                    raise RuntimeError("Model forward returned unsupported type in eval_gnn_on_graphs")

            pred_loss = _nodewise_regression_loss(
                out_flat, data.y.view(-1),
                target_node_idx=target_node_idx,
                target_loss_weight=target_loss_weight,
                target_only_loss=target_only_loss,
                loss_name=loss_name,
                smooth_l1_beta=smooth_l1_beta
            )
            loss = pred_loss
            tot += float(loss.item())
            tot_pred += float(pred_loss.item())

    denom = max(1, len(val_graphs))
    out_stats = {
        "total": tot / denom,
        "pred": tot_pred / denom,
    }
    if return_breakdown:
        return out_stats
    return out_stats["total"]


def make_graph_wrapper_class(base_class, model_in, default_hid=32):
    """
    Returns a wrapper class whose __init__ signature accepts (in_channels=..., hid_channels=..., out_channels=1, **kwargs)
    but will instantiate base_class with the fixed model_in as the x feature dim used during training.
    """
    class Wrapper(base_class):  # inherit to preserve method names if possible
        def __init__(self, in_channels=None, hid_channels=default_hid, out_channels=1, *args, **kwargs):
            # ignore incoming in_channels and use the model_in that was used for training
            mi = int(model_in) if model_in is not None else (in_channels or default_hid)
            # try common constructor signatures flexibly
            try:
                super(Wrapper, self).__init__(in_channels=mi, hid_channels=hid_channels, out_channels=out_channels, *args, **kwargs)
            except TypeError:
                try:
                    super(Wrapper, self).__init__(mi, hid_channels)
                except Exception:
                    # last resort: call base_class() without args; hope state_dict keys will match.
                    super(Wrapper, self).__init__()
    Wrapper.__name__ = f"{base_class.__name__}_wrap_in{model_in}"
    return Wrapper


def train_cv(modelspec, X=None, y=None, graphs=None,
             n_splits=5, torch_epochs=50, device=None, run_dir="out/run001",
             seed=42, batch_size=16, lr=1e-3, weight_decay=0.0, patience=10,
             model_kwargs=None, model_name_override=None, graph_objective_overrides=None,
             use_cv=None, fixed_val_X=None, fixed_val_y=None,
             fixed_val_graphs=None, fixed_eval_X=None, fixed_eval_y=None):
    """
    Returns: dict with keys:
       'type': 'vector'|'gnn'|'sklearn'
       'models': list (sklearn wrappers OR torch model instances OR gnn state_dicts)
       'paths': list of saved filepaths
       'model_name': string
       'model_in': (for GNN training) the in_channels used for training (int)
    """
    os.makedirs(run_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if use_cv is None:
        use_cv = not bool(getattr(cfg, "RUN_WITHOUT_CV", False))
    no_cv_holdout_ratio = float(getattr(cfg, "NO_CV_VAL_RATIO", 0.3))
    train_verbose = bool(getattr(cfg, "TRAIN_VERBOSE", True))
    train_verbose_gnn = bool(getattr(cfg, "TRAIN_VERBOSE_GNN", train_verbose))
    train_verbose_vector = bool(getattr(cfg, "TRAIN_VERBOSE_VECTOR", False))
    train_log_every = max(1, int(getattr(cfg, "TRAIN_LOG_EVERY", 5)))

    models_list = []
    saved_paths = []
    model_name = None
    model_in = None
    cv_metrics = []

    # --- vector torch models (mlp / cnn1d / resnet / tabnet / ft-transformer / custom callable) ---
    vector_torch_strings = ("linear", "mlp", "cnn1d", "resnet_mlp", "tabnet", "ft_transformer", "ft-transformer")
    if (isinstance(modelspec, str) and modelspec in vector_torch_strings) or (
        callable(modelspec) and modelspec not in (
            GNN, GraphSAGE, TorchGAT, TorchGIN
        )
    ):
        if X is None or y is None:
            raise ValueError("X and y required for vector torch models")
        split_pairs = _build_fold_splits(
            n_samples=len(X), n_splits=n_splits, seed=seed,
            use_cv=bool(use_cv), holdout_ratio=no_cv_holdout_ratio
        )
        n_folds_used = int(len(split_pairs))
        if isinstance(modelspec, str) and modelspec == "linear":
            model_name = "linear"
        elif isinstance(modelspec, str) and modelspec == "mlp":
            model_name = "mlp"
        elif isinstance(modelspec, str) and modelspec == "cnn1d":
            model_name = "cnn1d"
        elif isinstance(modelspec, str) and modelspec == "resnet_mlp":
            model_name = "resnet_mlp"
        elif isinstance(modelspec, str) and modelspec == "tabnet":
            model_name = "tabnet"
        elif isinstance(modelspec, str) and modelspec in ("ft_transformer", "ft-transformer"):
            model_name = "ft_transformer"
        else:
            model_name = "torchmodel"
        model_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for fold, (tr_idx, val_idx) in enumerate(split_pairs, start=1):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            use_fixed_val = (
                fixed_val_X is not None and fixed_val_y is not None and
                len(fixed_val_X) > 0 and len(fixed_val_y) > 0
            )
            X_val_used = fixed_val_X if use_fixed_val else X_val
            y_val_used = fixed_val_y if use_fixed_val else y_val
            if isinstance(modelspec, str) and modelspec == "linear":
                mdl = TorchLinear(X.shape[1])
                fname = "linear"
            elif isinstance(modelspec, str) and modelspec == "mlp":
                mdl = TorchMLP(X.shape[1])
                fname = "mlp"
            elif isinstance(modelspec, str) and modelspec == "cnn1d":
                mdl = TorchCNN1D(X.shape[1])
                fname = "cnn1d"
            elif isinstance(modelspec, str) and modelspec == "resnet_mlp":
                mdl = TorchResNetMLP(X.shape[1])
                fname = "resnet_mlp"
            elif isinstance(modelspec, str) and modelspec == "tabnet":
                mdl = TorchTabNet(X.shape[1])
                fname = "tabnet"
            elif isinstance(modelspec, str) and modelspec in ("ft_transformer", "ft-transformer"):
                mdl = TorchFTTransformer(X.shape[1])
                fname = "ft_transformer"
            else:
                mdl = modelspec(X.shape[1])
                fname = "torchmodel"
            train_ds = VecDataset(X_tr, y_tr); val_ds = VecDataset(X_val_used, y_val_used)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            mdl = train_torch_model(
                mdl, train_loader, val_loader,
                device=device, epochs=torch_epochs, lr=lr, weight_decay=weight_decay,
                verbose=train_verbose_vector, log_every=train_log_every,
                model_name=model_name, fold=fold, n_folds=n_folds_used
            )
            try:
                mdl.eval()
                with torch.no_grad():
                    x_val_t = torch.tensor(X_val_used.astype('float32')).to(device)
                    y_pred_val = mdl(x_val_t).detach().cpu().numpy().reshape(-1)
                rmse_f, mae_f = _rmse_mae(np.asarray(y_val_used).reshape(-1), y_pred_val)
            except Exception:
                rmse_f, mae_f = float("nan"), float("nan")
            cv_metrics.append({
                "fold": int(fold),
                "rmse": float(rmse_f),
                "mae": float(mae_f)
            })
            path = os.path.join(model_dir, f"{fname}_fold{fold:03d}.pt")
            torch.save(mdl.state_dict(), path)
            models_list.append(mdl)
            saved_paths.append(path)
        return {"type": "vector_torch", "models": models_list, "paths": saved_paths, "model_name": model_name, "cv_metrics": cv_metrics}

    # --- GNN / GraphSAGE / GAT / GIN training path ---
    gnn_strings = ("gnn", "graphsage", "gat", "gin")
    if (isinstance(modelspec, str) and modelspec in gnn_strings) or (
        callable(modelspec) and modelspec in (
            GNN, GraphSAGE, TorchGAT, TorchGIN
        )
    ):
        if graphs is None or len(graphs) == 0:
            raise ValueError("graphs required for GNN training (train_graphs list)")
        split_pairs = _build_fold_splits(
            n_samples=len(graphs), n_splits=n_splits, seed=seed,
            use_cv=bool(use_cv), holdout_ratio=no_cv_holdout_ratio
        )
        n_folds_used = int(len(split_pairs))
        model_kwargs = dict(model_kwargs or {})
        graph_objective_overrides = dict(graph_objective_overrides or {})

        # select class and model_name
        if isinstance(modelspec, str):
            if modelspec == "gnn":
                model_name = "gnn"
                ModelClass = GNN
            elif modelspec == "graphsage":
                model_name = "graphsage"
                ModelClass = GraphSAGE
            elif modelspec == "gat":
                model_name = "gat"
                ModelClass = TorchGAT
            elif modelspec == "gin":
                model_name = "gin"
                ModelClass = TorchGIN
            else:
                ModelClass = GNN
                model_name = "gnn"
        else:
            ModelClass = modelspec
            model_name = getattr(ModelClass, "__name__", "gnnmodel").lower()
        if model_name_override is not None and str(model_name_override).strip():
            model_name = str(model_name_override).strip()

        model_dir = os.path.join(run_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # determine in_channels from first graph (robust to missing .x)
        if not hasattr(graphs[0], "x") or graphs[0].x is None:
            raise RuntimeError(f"graphs[0] has no .x feature matrix")
        in_channels = int(graphs[0].x.shape[1])
        model_in = in_channels  # record for evaluation-time wrapper creation
        target_node_idx = int(getattr(cfg, "TARGET", 39))

        # Graph-objective knobs. The retained GraphSAGE baseline keeps the
        # original target-focused objective, while the other graph backbones
        # default to all-node MSE unless explicitly overridden.
        is_target_focused_graph = bool(
            (isinstance(modelspec, str) and modelspec == "graphsage") or
            (ModelClass in (GraphSAGE,))
        )
        if is_target_focused_graph:
            target_loss_weight = float(getattr(cfg, "GRAPH_TARGET_LOSS_WEIGHT", 1.0))
            target_only_loss = bool(getattr(cfg, "GRAPH_TARGET_ONLY_LOSS", True))
            loss_name = str(getattr(cfg, "GRAPH_LOSS_NAME", "mse")).strip().lower()
            smooth_l1_beta = float(getattr(cfg, "GRAPH_SMOOTH_L1_BETA", 0.1))
        else:
            target_loss_weight = 1.0
            target_only_loss = False
            loss_name = "mse"
            smooth_l1_beta = 0.1

        # Apply explicit objective overrides for any graph model. This keeps the
        # default behavior unchanged while allowing callers to force a shared
        # loss across graph backbones.
        if "target_loss_weight" in graph_objective_overrides:
            target_loss_weight = float(graph_objective_overrides["target_loss_weight"])
        if "target_only_loss" in graph_objective_overrides:
            target_only_loss = bool(graph_objective_overrides["target_only_loss"])
        if "loss_name" in graph_objective_overrides:
            loss_name = str(graph_objective_overrides["loss_name"]).strip().lower()
        if "smooth_l1_beta" in graph_objective_overrides:
            smooth_l1_beta = float(graph_objective_overrides["smooth_l1_beta"])

        fold_state_dicts = []
        fold_paths = []
        for fold, (tr_idx, val_idx) in enumerate(split_pairs, start=1):
            train_graphs = [graphs[i] for i in tr_idx]
            val_graphs = [graphs[i] for i in val_idx]
            use_fixed_val_graphs = bool(fixed_val_graphs is not None and len(fixed_val_graphs) > 0)
            val_graphs_used = list(fixed_val_graphs) if use_fixed_val_graphs else val_graphs
            if train_verbose_gnn:
                print(
                    f"[TRAIN][{model_name}] fold={fold}/{n_folds_used} start "
                    f"train_graphs={len(train_graphs)} val_graphs={len(val_graphs_used)} "
                    f"target_only={target_only_loss} target_w={target_loss_weight} "
                    f"loss={loss_name} beta={smooth_l1_beta:.6g}"
                )

            # instantiate model with flexible signatures
            model = None
            instantiate_errors = []
            if model_kwargs:
                try:
                    kw = dict(model_kwargs)
                    kw.setdefault("in_channels", in_channels)
                    model = ModelClass(**kw)
                except Exception as e:
                    instantiate_errors.append(f"kwargs={kw} err={e}")
            if model is None:
                try:
                    model = ModelClass(in_channels=in_channels, hid_channels=32, out_channels=1)
                except Exception as e:
                    instantiate_errors.append(f"in_channels/hid/out err={e}")
            if model is None:
                try:
                    model = ModelClass(in_channels=in_channels, hidden_channels=32, out_channels=1)
                except Exception as e:
                    instantiate_errors.append(f"in_channels/hidden/out err={e}")
            if model is None:
                try:
                    model = ModelClass(in_channels=in_channels)
                except Exception as e:
                    instantiate_errors.append(f"in_channels kw err={e}")
            if model is None:
                try:
                    model = ModelClass(in_channels)
                except Exception as e:
                    instantiate_errors.append(f"in_channels pos err={e}")
            if model is None:
                try:
                    model = ModelClass()
                except Exception as e:
                    instantiate_errors.append(f"no-arg err={e}")
                    raise RuntimeError(
                        f"Could not instantiate {getattr(ModelClass, '__name__', str(ModelClass))} "
                        f"for model_name={model_name}. Errors: {instantiate_errors}"
                    )

            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, torch_epochs)) if torch_epochs > 1 else None
            best_val = float('inf'); best_state = None; pat = 0

            for ep in range(torch_epochs):
                train_stats = train_single_gnn_epoch(
                    model, optimizer, train_graphs, device,
                    target_node_idx=target_node_idx,
                    target_loss_weight=target_loss_weight,
                    target_only_loss=target_only_loss,
                    return_breakdown=True,
                    loss_name=loss_name,
                    smooth_l1_beta=smooth_l1_beta
                )
                val_stats = None
                try:
                    val_stats = eval_gnn_on_graphs(
                        model, val_graphs_used, device,
                        target_node_idx=target_node_idx,
                        target_loss_weight=target_loss_weight,
                        target_only_loss=target_only_loss,
                        return_breakdown=True,
                        loss_name=loss_name,
                        smooth_l1_beta=smooth_l1_beta
                    )
                except Exception:
                    val_stats = {"total": float('inf'), "pred": float('nan')}

                val_loss = float(val_stats.get("total", float('inf')))

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
                        f"[TRAIN][{model_name}] fold={fold}/{n_folds_used} ep={ep+1}/{torch_epochs} "
                        f"train_total={train_stats.get('total', float('nan')):.6g} "
                        f"train_pred={train_stats.get('pred', float('nan')):.6g} "
                        f"val_total={val_stats.get('total', float('nan')):.6g} "
                        f"val_pred={val_stats.get('pred', float('nan')):.6g} "
                        f"best={best_val:.6g} pat={pat}/{patience}"
                    )

                if pat >= patience:
                    if train_verbose_gnn:
                        print(f"[TRAIN][{model_name}] fold={fold}/{n_folds_used} early_stop ep={ep+1} best={best_val:.6g}")
                    break

            # ensure we always have a best_state (fallback to current model state)
            if best_state is None:
                print(f"[WARN][train_cv] fold {fold} did not find finite val/train comparator; using final model state as fallback.")
                best_state = copy.deepcopy(model.state_dict())
            else:
                model.load_state_dict(best_state)

            # Per-fold target-node validation metrics (normalized scale)
            try:
                y_true_fold = []
                y_pred_fold = []
                for vg in val_graphs_used:
                    pred_nodes = _forward_graph_model(model, vg, device=device)
                    y_pred_fold.append(float(pred_nodes[target_node_idx]))
                    y_true_fold.append(float(vg.y.view(-1)[target_node_idx].detach().cpu().numpy()))
                rmse_f, mae_f = _rmse_mae(y_true_fold, y_pred_fold)
            except Exception:
                rmse_f, mae_f = float("nan"), float("nan")
            cv_metrics.append({
                "fold": int(fold),
                "rmse": float(rmse_f),
                "mae": float(mae_f),
                "best_objective": float(best_val) if np.isfinite(best_val) else float("nan")
            })

            fname = os.path.join(model_dir, f"{model_name}_fold{fold:03d}.pt")
            try:
                torch.save(best_state, fname)
            except Exception:
                pass
            fold_state_dicts.append(best_state)
            fold_paths.append(fname)

        return {"type": "gnn", "models": fold_state_dicts, "paths": fold_paths, "model_name": model_name, "model_in": model_in, "cv_metrics": cv_metrics}

    raise ValueError("Unknown modelspec or unsupported configuration")
