# utils/evaluate.py
import os
import re
import numpy as np
from time import perf_counter
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import DataLoader
import time
import traceback
import config.config as cfg
import inspect

from utils.train_utils import VecDataset

# Evaluate vector ensembles (sklearn wrappers or torch models)
def evaluate_models(model_list, X, y, scaler, target_idx, device=None, batch_size=32):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preds = []
    total_it = 0.0

    if not model_list:
        return float('nan'), float('nan'), float('nan')

    # detect sklearn vs torch (sklearn models have .predict and are not torch.nn.Module)
    is_sklearn = any(hasattr(m, "predict") and not isinstance(m, torch.nn.Module) for m in model_list)

    if is_sklearn:
        for m in model_list:
            t0 = perf_counter()
            p = m.predict(X)
            t1 = perf_counter(); total_it += (t1 - t0)
            preds.append(np.asarray(p).reshape(-1))
        avg_pred = np.mean(np.stack(preds, axis=1), axis=1)
    else:
        loader = DataLoader(VecDataset(X, y), batch_size=batch_size, shuffle=False)
        for m in model_list:
            m.eval()
            m.to(device)
            preds_fold = []
            t0 = perf_counter()
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(device)
                    out = m(xb).cpu().numpy().reshape(-1)
                    preds_fold.append(out)
            t1 = perf_counter(); total_it += (t1 - t0)
            if preds_fold:
                preds.append(np.concatenate(preds_fold, axis=0))
            else:
                preds.append(np.zeros(len(X)))
        avg_pred = np.mean(np.stack(preds, axis=1), axis=1)

    # inverse transform
    try:
        n_features = int(scaler.n_features_in_)
        full_pred = np.zeros((len(avg_pred), n_features))
        full_true = np.zeros_like(full_pred)
        full_pred[:, target_idx] = avg_pred
        full_true[:, target_idx] = y
        y_pred_dn = scaler.inverse_transform(full_pred)[:, target_idx]
        y_true_dn = scaler.inverse_transform(full_true)[:, target_idx]
        rmse = float(mean_squared_error(y_true_dn, y_pred_dn, squared=False))
        mae = float(mean_absolute_error(y_true_dn, y_pred_dn))
    except Exception:
        # fallback: compute metrics in normalized space if scaler not available or fails
        rmse = float(mean_squared_error(y, avg_pred, squared=False))
        mae = float(mean_absolute_error(y, avg_pred))

    return rmse, mae, total_it


def _infer_hid_from_state_dict(state_dict, in_channels_guess=1, default_hid=32):
    """Try to infer hidden channel size used when the checkpoint was created."""
    try:
        cand = []
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                r, c = v.shape
                if r > in_channels_guess:
                    cand.append(int(r))
        if len(cand) > 0:
            return max(cand)
    except Exception:
        pass
    return default_hid


def _infer_sage_hidden_from_state_dict(state_dict, default_hid=64):
    """Infer hidden size from BatchNorm/GraphSAGE parameters when available."""
    try:
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("sage_bn1.weight") and v.ndim == 1 and int(v.shape[0]) > 0:
                return int(v.shape[0])
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("sage1.lin_l.weight") and v.ndim == 2 and int(v.shape[0]) > 0:
                return int(v.shape[0])
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("sage_head.weight") and v.ndim == 2 and int(v.shape[1]) > 0:
                return int(v.shape[1])
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if re.search(r"(?:^|\.)convs\.0\.lin_l\.weight$", str(k)) and v.ndim == 2 and int(v.shape[0]) > 0:
                return int(v.shape[0])
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if re.search(r"(?:^|\.)convs\.0\.lin_r\.weight$", str(k)) and v.ndim == 2 and int(v.shape[0]) > 0:
                return int(v.shape[0])
    except Exception:
        pass
    return int(default_hid)


def _infer_edge_attr_dim_from_state_dict(state_dict, default_dim=3):
    """Infer edge_attr_dim from NN edge MLP first layer input width."""
    try:
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("nn_edge_mlp.0.weight") and v.ndim == 2 and int(v.shape[1]) > 0:
                return int(v.shape[1])
    except Exception:
        pass
    return int(default_dim)


def _infer_hidden_tuple_from_state_dict(state_dict, default_tuple=(64, 32, 16)):
    """Infer GNN hidden tuple (h1,h2,h3) from common conv key shapes."""
    try:
        h1 = h2 = h3 = None
        # Prefer bias vectors (unambiguous output dims)
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("conv1.bias") and v.ndim == 1:
                h1 = int(v.shape[0])
            elif k.endswith("conv2.bias") and v.ndim == 1:
                h2 = int(v.shape[0])
            elif k.endswith("conv3.bias") and v.ndim == 1:
                h3 = int(v.shape[0])
        # Fallback to weight rows if needed
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            if h1 is None and ("conv1" in k and "weight" in k):
                h1 = int(v.shape[0])
            elif h2 is None and ("conv2" in k and "weight" in k):
                h2 = int(v.shape[0])
            elif h3 is None and ("conv3" in k and "weight" in k):
                h3 = int(v.shape[0])
        if all(isinstance(x, int) and x > 0 for x in (h1, h2, h3)):
            return (h1, h2, h3)
    except Exception:
        pass
    return default_tuple


def _infer_residual_num_layers_from_state_dict(state_dict, default_layers=3):
    try:
        max_idx = -1
        for k in state_dict.keys():
            m = re.search(r"(?:^|\.)convs\.(\d+)\.", str(k))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx >= 0:
            return int(max_idx + 1)
    except Exception:
        pass
    return int(default_layers)


def _infer_gat_heads_from_state_dict(state_dict, default_heads=4):
    try:
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("conv1.att_src") and v.ndim >= 3 and int(v.shape[1]) > 0:
                return int(v.shape[1])
            if k.endswith("local_convs.0.att_src") and v.ndim >= 3 and int(v.shape[1]) > 0:
                return int(v.shape[1])
            if re.search(r"(?:^|\.)convs\.0\.att_src$", str(k)) and v.ndim >= 3 and int(v.shape[1]) > 0:
                return int(v.shape[1])
        for k, v in state_dict.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k.endswith("conv1.att_dst") and v.ndim >= 3 and int(v.shape[1]) > 0:
                return int(v.shape[1])
            if k.endswith("local_convs.0.att_dst") and v.ndim >= 3 and int(v.shape[1]) > 0:
                return int(v.shape[1])
            if re.search(r"(?:^|\.)convs\.0\.att_dst$", str(k)) and v.ndim >= 3 and int(v.shape[1]) > 0:
                return int(v.shape[1])
    except Exception:
        pass
    return int(default_heads)


def _safe_model_instantiate_and_load(model_class, state_dict, in_channels, device, default_hid=32):
    """
    Robust instantiation & load:
      - inspect model_class.__init__ signature and try to map commonly-used arg names
      - try keyword args based on detected parameter names
      - try positional args as fallback
      - try no-arg constructor as last resort
      - try strict load then non-strict
    Returns (model_on_device, load_info_dict)
    """
    inferred_hid = _infer_hid_from_state_dict(state_dict, in_channels_guess=in_channels, default_hid=default_hid)
    inferred_hidden_tuple = _infer_hidden_tuple_from_state_dict(state_dict, default_tuple=(64, 32, 16))
    inferred_gat_heads = _infer_gat_heads_from_state_dict(state_dict, default_heads=4)
    tried = []
    model_class_name = getattr(model_class, "__name__", "")
    inferred_edge_attr_dim = _infer_edge_attr_dim_from_state_dict(state_dict, default_dim=3)
    if model_class_name == "GraphSAGE":
        inferred_hid = _infer_sage_hidden_from_state_dict(state_dict, default_hid=max(16, int(default_hid)))

    # 1) inspect __init__ parameters (exclude 'self' / varargs)
    try:
        sig = inspect.signature(model_class.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        param_names = [p.name for p in params]
    except Exception:
        params = []
        param_names = []

    # mapping candidates: map likely parameter names to our values
    name_map = {
        "in_channels": in_channels, "in_ch": in_channels, "in_feats": in_channels,
        "num_node_features": in_channels, "num_features": in_channels, "input_dim": in_channels,
        "node_in": in_channels, "n_in": in_channels,
        # hidden
        "hid_channels": inferred_hid, "hidden": inferred_hid, "hidden_feats": inferred_hid,
        "hid": inferred_hid, "n_hidden": inferred_hid,
        # edge attrs
        "edge_attr_dim": inferred_edge_attr_dim,
        # target
        "target_node_idx": int(getattr(cfg, "TARGET", 39)),
        "target_idx": int(getattr(cfg, "TARGET", 39)),
        # out
        "out_channels": 1, "out_feats": 1, "out_dim": 1, "n_out": 1
    }

    # 2) try to build kwargs for constructor using param_names
    kwargs = {}
    for pname in param_names:
        # exact/special handling first (avoid ambiguous substring matches)
        pl = str(pname).lower()
        if pl == "hidden_channels":
            if model_class_name == "GNN":
                has_residual_stack = any(
                    re.search(r"(?:^|\.)convs\.\d+\.", str(k))
                    for k in state_dict.keys()
                )
                kwargs[pname] = int(inferred_hid) if has_residual_stack else inferred_hidden_tuple
            else:
                kwargs[pname] = inferred_hid
            continue
        # pick best candidate from name_map by substring match
        chosen = None
        for cand_name, val in name_map.items():
            if cand_name == pname or cand_name in pname or pname in cand_name:
                chosen = val
                break
        if chosen is not None:
            kwargs[pname] = chosen

    # Model-specific overrides for safer constructor matching.
    if model_class_name == "GraphSAGE":
        if "hidden_channels" in param_names:
            kwargs["hidden_channels"] = int(inferred_hid)
        if "num_layers" in param_names:
            kwargs["num_layers"] = _infer_residual_num_layers_from_state_dict(state_dict, default_layers=3)
    if model_class_name in ("GNN", "GraphSAGE", "TorchGAT", "TorchGIN"):
        if "hidden_channels" in param_names:
            kwargs["hidden_channels"] = int(inferred_hid)
        if "hid_channels" in param_names:
            kwargs["hid_channels"] = int(inferred_hid)
        if "num_layers" in param_names:
            kwargs["num_layers"] = _infer_residual_num_layers_from_state_dict(state_dict, default_layers=3)
    if model_class_name == "TorchGAT":
        if "heads" in param_names:
            kwargs["heads"] = int(inferred_gat_heads)

    # Try kwargs if any mappings present
    if kwargs:
        try:
            mdl = model_class(**kwargs)
            tried.append(("kw_ok", kwargs))
        except Exception as e:
            tried.append(("kw_fail", kwargs, str(e)))
            mdl = None
    else:
        mdl = None

    # 3) Special constructor paths before generic positional attempt
    if mdl is None and model_class_name == "GNN":
        try:
            mdl = model_class(in_channels=in_channels, hidden_channels=inferred_hidden_tuple)
            tried.append(("gnn_kw_ok", {"in_channels": in_channels, "hidden_channels": inferred_hidden_tuple}))
        except Exception as e:
            tried.append(("gnn_kw_fail", {"in_channels": in_channels, "hidden_channels": inferred_hidden_tuple}, str(e)))
            try:
                mdl = model_class(in_channels, inferred_hidden_tuple)
                tried.append(("gnn_pos_ok", (in_channels, inferred_hidden_tuple)))
            except Exception as e2:
                tried.append(("gnn_pos_fail", (in_channels, inferred_hidden_tuple), str(e2)))
                mdl = None

    # 4) Positional attempt using the param order: fill known slots with values
    if mdl is None:
        pos_args = []
        for pname in param_names:
            if any(k in pname for k in ("in_chan", "in_ch", "in_feats", "num_node", "num_feature", "input")):
                pos_args.append(in_channels)
            elif str(pname).lower() == "hidden_channels" and model_class_name == "GNN":
                pos_args.append(inferred_hidden_tuple)
            elif any(k in pname for k in ("hid", "hidden")):
                pos_args.append(inferred_hid)
            elif any(k in pname for k in ("out", "n_out", "out_feat", "out_dim")):
                pos_args.append(1)
            else:
                # give up on filling unknown required params (leave to later no-arg attempt)
                pos_args.append(None)
        # prune trailing None's (if any)
        pos_args_pruned = [a for a in pos_args if a is not None]
        try:
            mdl = model_class(*pos_args_pruned)
            tried.append(("pos_ok", tuple(pos_args_pruned)))
        except Exception as e:
            tried.append(("pos_fail", tuple(pos_args_pruned), str(e)))
            mdl = None

    # 5) last resort no-arg constructor
    if mdl is None:
        try:
            mdl = model_class()
            tried.append(("noarg_ok", None))
        except Exception as e:
            tried.append(("noarg_fail", str(e)))
            # provide very explicit diagnostic before raising
            sig_text = ""
            try:
                sig_text = str(inspect.signature(model_class.__init__))
            except Exception:
                sig_text = "<could not read signature>"
            raise RuntimeError(f"Could not instantiate model_class; attempts: {tried}. Constructor signature: {sig_text}")

    mdl = mdl.to(device)

    # 5) load state dict - strict True then False
    try:
        mdl.load_state_dict(state_dict)
        return mdl, {"load_strict": True, "missing_keys": [], "unexpected_keys": []}
    except Exception:
        try:
            res = mdl.load_state_dict(state_dict, strict=False)
            missing = getattr(res, "missing_keys", None) or res.get("missing_keys", [])
            unexpected = getattr(res, "unexpected_keys", None) or res.get("unexpected_keys", [])
            return mdl, {"load_strict": False, "missing_keys": missing, "unexpected_keys": unexpected}
        except Exception as e2:
            tb = traceback.format_exc()
            raise RuntimeError(f"Failed to load state_dict into model. Tried: {tried}. Err: {e2}. TB: {tb}")



def _attempt_forward(model, data, device):
    """
    Try several forward signatures to produce node-level predictions.
    Returns numpy array of shape (num_nodes,)
    """
    model.eval()
    with torch.no_grad():
        # try model(data)
        try:
            out = model(data.to(device))
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy().ravel()
        except Exception:
            pass

        # try model(x, edge_index)
        try:
            x = data.x.to(device)
            e = data.edge_index.to(device) if hasattr(data, "edge_index") else None
            if e is not None:
                out = model(x, e)
                if isinstance(out, torch.Tensor):
                    return out.detach().cpu().numpy().ravel()
        except Exception:
            pass

        # try model(x) only
        try:
            x = data.x.to(device)
            out = model(x)
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy().ravel()
        except Exception:
            pass

    raise RuntimeError("Model forward failed for all tried signatures (data, (x,e), x).")


def evaluate_gnn_ensemble(sets, scaler, method_name, category,
                          ground_truth_graph, fold_state_dicts, model_class,
                          device="cpu", target_node_idx=39, model_in=None,
                          ensemble_reducer="mean"):
    """
    Evaluate a GNN ensemble.
    - sets: list of "set" (each set is a list of Data graphs for that imputed/missing dataset).
    - ground_truth_graph: list of Data graphs (test set) used to obtain true y for target_node_idx.
    - fold_state_dicts: list of state_dicts (or filepaths) for trained folds.
    - model_class: class used to instantiate model
    - model_in: optional integer specifying the in_channels that were used during training (preferred)
    Returns list of rows with these keys:
      {"Method_Name":..., "Category":..., "Dataset_Index":..., "RMSE":..., "MAE":..., "Inference_Time":...}
    """
    results = []
    # load states (if file paths provided)
    loaded_states = []
    for s in fold_state_dicts:
        if isinstance(s, str):
            try:
                st = torch.load(s, map_location="cpu")
            except Exception:
                st = None
        else:
            st = s
        if st is None:
            continue
        loaded_states.append(st)

    if len(loaded_states) == 0:
        return []  # nothing to evaluate

    # decide in_channels: prefer explicit model_in, else infer from ground_truth_graph
    in_channels = None
    if model_in is not None:
        try:
            in_channels = int(model_in)
        except Exception:
            in_channels = None

    if in_channels is None:
        try:
            if isinstance(ground_truth_graph, (list, tuple)) and len(ground_truth_graph) > 0:
                g0 = ground_truth_graph[0]
                if hasattr(g0, "x") and g0.x is not None:
                    in_channels = int(g0.x.shape[1])
        except Exception:
            in_channels = None

    if in_channels is None:
        in_channels = 1  # safe fallback

    # normalize states (unwrap wrappers if necessary)
    normalized_states = []
    for st in loaded_states:
        if isinstance(st, dict) and ("model" in st and isinstance(st["model"], dict)):
            normalized_states.append(st["model"])
            continue
        if isinstance(st, dict) and ("state_dict" in st and isinstance(st["state_dict"], dict)):
            normalized_states.append(st["state_dict"])
            continue
        normalized_states.append(st)

    # number of features for inverse transform (if scaler present)
    n_features = getattr(scaler, "n_features_in_", None)

    # iterate each imputed/missing set
    for set_idx, set_graphs in enumerate(sets, start=1):
        num_samples = len(set_graphs)
        if num_samples == 0:
            continue

        preds_by_fold = []
        total_inference_time = 0.0
        valid_folds = 0

        for st_idx, st in enumerate(normalized_states):
            try:
                state_dict = st
                if "state_dict" in st and isinstance(st["state_dict"], dict):
                    state_dict = st["state_dict"]

                # instantiate and load
                try:
                    mdl, load_info = _safe_model_instantiate_and_load(model_class, state_dict, in_channels=in_channels, device=device, default_hid=getattr(cfg, "DEFAULT_GNN_HID", 32))
                except Exception as e:
                    # If instantiate/load failed, warn and skip fold
                    print(f"[evaluate_gnn_ensemble] Warning: could not instantiate/load fold {st_idx}: {e}")
                    continue

                valid_folds += 1
                fold_preds = np.zeros((num_samples,), dtype=float)

                t0 = time.time()
                for si, g in enumerate(set_graphs):
                    try:
                        node_preds = _attempt_forward(mdl, g, device)
                        fold_preds[si] = float(node_preds[int(target_node_idx)])
                    except Exception as e:
                        raise RuntimeError(f"Forward failed for fold {st_idx}, sample {si}: {e}")
                t1 = time.time()

                preds_by_fold.append(fold_preds)
                total_inference_time += (t1 - t0)
            except Exception as e:
                print(f"[evaluate_gnn_ensemble] Warning: failed to use fold {st_idx}: {e}")
                continue

        if valid_folds == 0:
            print(f"[evaluate_gnn_ensemble] no valid folds for set_idx={set_idx}; skipping")
            continue

        # stack & aggregate across folds
        try:
            preds_stack = np.vstack(preds_by_fold)  # shape (valid_folds, num_samples)
            reducer = str(ensemble_reducer).strip().lower()
            if reducer == "median":
                preds_agg = np.nanmedian(preds_stack, axis=0)
            else:
                preds_agg = np.nanmean(preds_stack, axis=0)
        except Exception as e:
            print(f"[evaluate_gnn_ensemble] Error stacking preds for set_idx={set_idx}: {e}")
            continue

        # obtain ground-truth y for each sample (from ground_truth_graph)
        y_true_norm = []
        for si in range(num_samples):
            try:
                gt = ground_truth_graph[si]
                if hasattr(gt, "y"):
                    # gt.y may be tensor
                    yv = gt.y[int(target_node_idx)]
                    if isinstance(yv, torch.Tensor):
                        yv = float(yv.detach().cpu().numpy().ravel()[0])
                    else:
                        yv = float(np.asarray(yv).ravel()[0])
                    y_true_norm.append(yv)
                else:
                    y_true_norm.append(np.nan)
            except Exception:
                y_true_norm.append(np.nan)
        y_true_norm = np.array(y_true_norm, dtype=float)

        # try to invert via scaler if available
        if n_features is not None:
            try:
                full_pred = np.zeros((num_samples, n_features))
                full_true = np.zeros((num_samples, n_features))
                # place normalized predictions & truths at the target index
                full_pred[:, target_node_idx] = preds_agg
                full_true[:, target_node_idx] = y_true_norm
                y_pred_dn = scaler.inverse_transform(full_pred)[:, target_node_idx]
                y_true_dn = scaler.inverse_transform(full_true)[:, target_node_idx]
                # compute metrics on denormalized values
                mask_valid = ~np.isnan(y_true_dn) & ~np.isnan(y_pred_dn)
                if mask_valid.sum() == 0:
                    rmse = float("nan"); mae = float("nan")
                else:
                    rmse = float(np.sqrt(mean_squared_error(y_true_dn[mask_valid], y_pred_dn[mask_valid])))
                    mae = float(mean_absolute_error(y_true_dn[mask_valid], y_pred_dn[mask_valid]))
            except Exception as e:
                # fallback to normalized metrics
                mask_valid = ~np.isnan(y_true_norm) & ~np.isnan(preds_agg)
                if mask_valid.sum() == 0:
                    rmse = float("nan"); mae = float("nan")
                else:
                    rmse = float(np.sqrt(mean_squared_error(y_true_norm[mask_valid], preds_agg[mask_valid])))
                    mae = float(mean_absolute_error(y_true_norm[mask_valid], preds_agg[mask_valid]))
        else:
            # no scaler available -> compute metrics in normalized space
            mask_valid = ~np.isnan(y_true_norm) & ~np.isnan(preds_agg)
            if mask_valid.sum() == 0:
                rmse = float("nan"); mae = float("nan")
            else:
                rmse = float(np.sqrt(mean_squared_error(y_true_norm[mask_valid], preds_agg[mask_valid])))
                mae = float(mean_absolute_error(y_true_norm[mask_valid], preds_agg[mask_valid]))

        avg_inference_time = float(total_inference_time / valid_folds) if valid_folds > 0 else 0.0

        results.append({
            "Method_Name": method_name,
            "Category": category,
            "Dataset_Index": int(set_idx),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Inference_Time": float(avg_inference_time)
        })

    return results
