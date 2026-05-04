import os
import time
import pprint
import argparse
import shutil
import numpy as np
import pandas as pd
import torch

from utils import dataloader
from utils.train_utils import (
    train_cv, next_run_dir
)

from utils.evaluate import evaluate_models, evaluate_gnn_ensemble
import models.baseline_models as models
import config.config as cfg
from collections import defaultdict

_BACKBONE_KEY_ALIASES = {
    "sklearn_lr": "linear",
}


def _canonical_backbone_key(key):
    s = str(key).strip().lower()
    return _BACKBONE_KEY_ALIASES.get(s, s)

_MODEL_KEY_TO_CLASSNAME = {
    "gnn": "GNN",
    "graphsage": "GraphSAGE",
    "gat": "TorchGAT",
    "gin": "TorchGIN"
}

VECTOR_BACKBONES = [_canonical_backbone_key(k) for k in getattr(cfg, "BASELINE_VECTOR_BACKBONES", [])]
GRAPH_BACKBONES = [_canonical_backbone_key(k) for k in getattr(cfg, "BASELINE_GRAPH_BACKBONES", [])]
_SUPPORTED_BACKBONE_KEYS = set(VECTOR_BACKBONES + GRAPH_BACKBONES)
BACKBONE_KEYS = []
for _k in getattr(cfg, "BACKBONE_KEYS", VECTOR_BACKBONES + GRAPH_BACKBONES):
    _ck = _canonical_backbone_key(_k)
    if _ck not in _SUPPORTED_BACKBONE_KEYS or _ck in BACKBONE_KEYS:
        continue
    BACKBONE_KEYS.append(_ck)
BACKBONE_DISPLAY = {}
for _k, _v in dict(getattr(cfg, "BACKBONE_DISPLAY", {})).items():
    _ck = _canonical_backbone_key(_k)
    if _ck in _SUPPORTED_BACKBONE_KEYS:
        BACKBONE_DISPLAY[_ck] = _v

CANON_IMPUTE_MAP = {m.upper(): m for m in getattr(cfg, "IMPUTATION_METHODS", [])}
_GRAPH_OBJECTIVE_OVERRIDES = {
    "target_loss_weight": 1.0,
    "target_only_loss": True,
    "loss_name": "mse",
    "smooth_l1_beta": 0.1,
}


def canonical_imputation_name(name):
    if name is None:
        return ""
    s = str(name).strip()
    if s == "":
        return ""
    su = s.upper().replace(" ", "_").replace("-", "_")
    if su in CANON_IMPUTE_MAP:
        return CANON_IMPUTE_MAP[su]
    su_short = su.replace("_IMPUTATION", "")
    if su_short in CANON_IMPUTE_MAP:
        return CANON_IMPUTE_MAP[su_short]
    for k in CANON_IMPUTE_MAP:
        if k in su or su in k:
            return CANON_IMPUTE_MAP[k]
    return su


def _display_name(key):
    ckey = _canonical_backbone_key(key)
    return BACKBONE_DISPLAY.get(ckey, str(ckey).upper())


def _normalize_eval_rows(rows, base_method_name, category, imputation_method=None, training_time=np.nan):
    """Normalize rows from evaluate_gnn_ensemble for consistent columns."""
    for r in rows:
        r["Method"] = base_method_name
        r["Category"] = category
        if imputation_method is not None:
            r["Imputation_Method"] = canonical_imputation_name(imputation_method)
        else:
            im = r.get("Imputation_Method", "")
            r["Imputation_Method"] = canonical_imputation_name(im) if im else ""
        if "Graph" in r:
            r.pop("Graph", None)
        if "Method_Name" in r:
            r.pop("Method_Name", None)
        r["Training_Time"] = float(training_time) if training_time is not None else np.nan
    return rows


def _safe_unpack_loader(outs):
    """
    Accept various loader return signatures and normalize to:
    (
        train_graphs, test_graphs,
        scaler, edge_index,
        imputed_records, imputed_graph_sets,
        missing_records, missing_graph_sets,
        metadata
    )
    """
    if not isinstance(outs, (list, tuple)):
        raise RuntimeError("dataloader.load_dataset returned unexpected type")
    ln = len(outs)
    if ln == 9:
        return outs
    if ln == 10:
        # Backward compatibility for legacy train/val/test loader returns (no edge graphs).
        (train_graphs, _val_graphs, test_graphs,
         scaler, edge_index,
         imputed_records, imputed_graph_sets,
         missing_records, missing_graph_sets,
         metadata) = outs
        return (
            train_graphs, test_graphs,
            scaler, edge_index,
            imputed_records, imputed_graph_sets,
            missing_records, missing_graph_sets,
            metadata
        )
    raise RuntimeError(f"Unexpected loader return length: {ln}. Expected 9 or 10.")


def _parse_cli_flags(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-cv", action="store_true", help="Run single-split training without K-fold CV.")
    parser.add_argument("--with-cv", action="store_true", help="Force K-fold CV even if config disables it.")
    parser.add_argument("--eval-only", action="store_true", help="Reserved for pre-trained baseline evaluation.")
    parser.add_argument(
        "--pretrained-dir",
        default=getattr(cfg, "BASELINE_SAVE_DIR", os.path.join("save", "baseline")),
        help="Directory containing pre-trained baseline checkpoint subfolders.",
    )
    parser.add_argument(
        "--save-pretrained",
        action="store_true",
        help="After training, copy baseline checkpoints into --pretrained-dir.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


# ------------------------- Main -------------------------
def main(argv=None):
    global CANON_IMPUTE_MAP
    run_dir = next_run_dir(out_root=os.path.join(getattr(cfg, "OUT_PATH", "out")))
    print("Outputs & models saved to:", run_dir)

    if getattr(cfg, "SAVE_RUN_LOG", True):
        import sys
        log_path = os.path.join(run_dir, "run_log.txt")
        log_f = open(log_path, "w", buffering=1)
        class _Tee:
            def __init__(self, *files): self.files = files
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
        
    t0_global = time.time()
    cli_flags = _parse_cli_flags(argv)
    run_without_cv = bool(getattr(cfg, "RUN_WITHOUT_CV", False))
    if bool(getattr(cli_flags, "no_cv", False)):
        run_without_cv = True
    if bool(getattr(cli_flags, "with_cv", False)):
        run_without_cv = False
    setattr(cfg, "RUN_WITHOUT_CV", bool(run_without_cv))
    if run_without_cv:
        print("[INFO] Running without K-fold CV (single holdout split per method).")
    else:
        print("[INFO] Running with K-fold CV.")

    CANON_IMPUTE_MAP = {m.upper(): m for m in getattr(cfg, "IMPUTATION_METHODS", [])}
    print(f"[INFO] Imputed evaluation methods: {cfg.IMPUTATION_METHODS}")

    print("Loading dataset...")
    outs = dataloader.load_dataset(cfg)
    (train_graphs, test_graphs,
     scaler, edge_index,
     imputed_records, imputed_graph_sets_non,
     missing_records, missing_graph_sets_non,
     metadata) = _safe_unpack_loader(outs)

    active_imputation_methods = {
        canonical_imputation_name(method)
        for method in getattr(cfg, "IMPUTATION_METHODS", [])
    }

    def _filter_imputed_payload(payload):
        keep = {}
        for key, value in (payload or {}).items():
            parts = str(key).split("_", 1)
            impute_method = parts[1] if len(parts) > 1 else key
            if canonical_imputation_name(impute_method) in active_imputation_methods:
                keep[key] = value
        return keep

    imputed_records = _filter_imputed_payload(imputed_records)
    imputed_graph_sets_non = _filter_imputed_payload(imputed_graph_sets_non)

    # quick summary of loaded imputed/missing keys
    print("\n[LOADER SUMMARY]")
    for k, recs in (imputed_records or {}).items():
        try:
            print(f"  imputed: {k}: {len(recs)} set(s)")
        except Exception:
            print(f"  imputed: {k}: (count unknown)")
    for topk, scen_map in (missing_records or {}).items():
        print(f"  missing top={topk}: scen_count={len(scen_map.keys())}")
    print("End loader summary\n")

    # raw arrays required for vector models
    raw = metadata.get("raw", {})
    graph_meta = metadata.get("graphs", {})
    train_raw = raw.get("train"); val_raw = raw.get("val"); test_raw = raw.get("test")
    val_graphs = graph_meta.get("val", [])
    if train_raw is None or test_raw is None:
        raise RuntimeError("raw train/test arrays missing in metadata['raw']")
    if val_raw is None:
        val_raw = np.empty((0, train_raw.shape[1]), dtype=float)

    # prepare arrays for vector models
    train_norm = scaler.transform(train_raw).astype(np.float32)
    val_norm = scaler.transform(val_raw).astype(np.float32) if val_raw is not None and len(val_raw) > 0 else np.zeros((0, train_raw.shape[1]), dtype=np.float32)
    test_norm = scaler.transform(test_raw).astype(np.float32)

    target_idx = int(getattr(cfg, "TARGET", 39))
    X_train = np.delete(train_norm, target_idx, axis=1)
    y_train = train_norm[:, target_idx]
    X_val = np.delete(val_norm, target_idx, axis=1) if val_norm.shape[0] > 0 else np.zeros((0, X_train.shape[1]), dtype=np.float32)
    y_val = val_norm[:, target_idx] if val_norm.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    X_test = np.delete(test_norm, target_idx, axis=1)
    y_test = test_norm[:, target_idx]

    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_external_vector_val = bool(X_val.shape[0] > 0 and y_val.shape[0] > 0)
    use_external_graph_val = bool(val_graphs is not None and len(val_graphs) > 0)
    fixed_eval_kwargs = {"fixed_eval_X": X_val, "fixed_eval_y": y_val} if use_external_vector_val else {}
    fixed_val_kwargs = {"fixed_val_X": X_val, "fixed_val_y": y_val} if use_external_vector_val else {}
    fixed_graph_val_kwargs = {"fixed_val_graphs": val_graphs} if use_external_graph_val else {}
    print(
        f"Validation usage: external_vector_val={use_external_vector_val}, "
        f"external_graph_val={use_external_graph_val}"
    )
    if run_without_cv:
        n_splits = 1
    else:
        n_splits = int(max(2, min(int(getattr(cfg, "CV_SPLITS", 5)), int(X_train.shape[0]))))
    torch_epochs = int(getattr(cfg, "TORCH_EPOCHS", 80))
    seed = int(getattr(cfg, "RANDOM_SEED", 42))
    if bool(getattr(cli_flags, "eval_only", False)):
        raise RuntimeError(
            "Baseline --eval-only is not available yet because baseline checkpoints "
            "include mixed vector and graph model formats. Use fusion --eval-only, "
            "or run baseline training with --save-pretrained to stage checkpoints."
        )

    trained = {}
    training_times = {}
    cv_rows = []

    def _graph_train_cv(modelspec, **kwargs):
        merged = dict(kwargs.pop("graph_objective_overrides", None) or {})
        for key, value in _GRAPH_OBJECTIVE_OVERRIDES.items():
            merged.setdefault(key, value)
        return train_cv(modelspec, graph_objective_overrides=merged, **kwargs)

    def _cv_method_key_from_trained_key(trained_key):
        return trained_key

    def _append_cv_metrics(trained_key, res, display_name=None):
        if not isinstance(res, dict):
            return
        method_display = str(display_name).strip() if display_name else _display_name(_cv_method_key_from_trained_key(trained_key))
        for m in (res.get("cv_metrics") or []):
            cv_rows.append({
                "Trained_Key": trained_key,
                "Method": method_display,
                "Fold": int(m.get("fold")) if m.get("fold") is not None else np.nan,
                "RMSE": float(m.get("rmse", np.nan)),
                "MAE": float(m.get("mae", np.nan)),
                "Best_Objective": float(m.get("best_objective", np.nan)) if m.get("best_objective", None) is not None else np.nan,
                "Training_Time": float(training_times.get(trained_key, np.nan))
            })

    processed_impute_counts = defaultdict(int)
    skipped_impute_reasons = defaultdict(list)
    processed_missing_counts = defaultdict(int)
    skipped_missing_reasons = defaultdict(list)

    # ----------------
    # 1) Linear projection regressor
    # ----------------
    print("Training Torch linear ensemble...")
    try:
        t_train = time.time()
        res_linear = train_cv("linear", X=X_train, y=y_train,
                              n_splits=n_splits, torch_epochs=torch_epochs, device=device,
                              run_dir=run_dir, lr=1e-3, batch_size=16, weight_decay=0.0, seed=seed,
                              **fixed_val_kwargs)
        trained["linear"] = res_linear
        training_times["linear"] = time.time() - t_train
        _append_cv_metrics("linear", res_linear)
    except Exception as e:
        print("linear train_cv failed:", e)

    # ----------------
    # 2) MLP
    # ----------------
    print("Training Torch MLP ensemble...")
    try:
        t_train = time.time()
        res_mlp = train_cv("mlp", X=X_train, y=y_train,
                           n_splits=n_splits, torch_epochs=torch_epochs, device=device,
                           run_dir=run_dir, lr=1e-3, batch_size=16, weight_decay=0.0, seed=seed,
                           **fixed_val_kwargs)
        trained["mlp"] = res_mlp
        training_times["mlp"] = time.time() - t_train
        _append_cv_metrics("mlp", res_mlp)
    except Exception as e:
        print("mlp train_cv failed:", e)

    # ----------------
    # 3) CNN1D 
    # ----------------
    print("Training Torch CNN1D ensemble...")
    try:
        t_train = time.time()
        res_cnn = train_cv("cnn1d", X=X_train, y=y_train,
                           n_splits=n_splits, torch_epochs=torch_epochs, device=device,
                           run_dir=run_dir, lr=1e-3, batch_size=16, weight_decay=0.0, seed=seed,
                           **fixed_val_kwargs)
        trained["cnn1d"] = res_cnn
        training_times["cnn1d"] = time.time() - t_train
        _append_cv_metrics("cnn1d", res_cnn)
    except Exception as e:
        print("cnn1d train_cv failed:", e)

    # ----------------
    # 4) ResNet-style MLP
    # ----------------
    print("Training Torch ResNet-MLP ensemble...")
    try:
        t_train = time.time()
        res_resnet_mlp = train_cv("resnet_mlp", X=X_train, y=y_train,
                                  n_splits=n_splits, torch_epochs=torch_epochs, device=device,
                                  run_dir=run_dir, lr=1e-3, batch_size=16, weight_decay=0.0, seed=seed,
                                  **fixed_val_kwargs)
        trained["resnet_mlp"] = res_resnet_mlp
        training_times["resnet_mlp"] = time.time() - t_train
        _append_cv_metrics("resnet_mlp", res_resnet_mlp)
    except Exception as e:
        print("resnet_mlp train_cv failed:", e)

    # ----------------
    # 5) TabNet 
    # ----------------
    print("Training Torch TabNet ensemble...")
    try:
        t_train = time.time()
        res_tabnet = train_cv("tabnet", X=X_train, y=y_train,
                              n_splits=n_splits, torch_epochs=torch_epochs, device=device,
                              run_dir=run_dir, lr=1e-3, batch_size=16, weight_decay=0.0, seed=seed,
                              **fixed_val_kwargs)
        trained["tabnet"] = res_tabnet
        training_times["tabnet"] = time.time() - t_train
        _append_cv_metrics("tabnet", res_tabnet)
    except Exception as e:
        print("tabnet train_cv failed:", e)

    # ----------------
    # 6) FT-Transformer 
    # ----------------
    print("Training Torch FT-Transformer ensemble...")
    try:
        t_train = time.time()
        res_ft_transformer = train_cv("ft_transformer", X=X_train, y=y_train,
                                      n_splits=n_splits, torch_epochs=torch_epochs, device=device,
                                      run_dir=run_dir, lr=1e-3, batch_size=16, weight_decay=0.0, seed=seed,
                                      **fixed_val_kwargs)
        trained["ft_transformer"] = res_ft_transformer
        training_times["ft_transformer"] = time.time() - t_train
        _append_cv_metrics("ft_transformer", res_ft_transformer)
    except Exception as e:
        print("ft_transformer train_cv failed:", e)

    # ----------------
    # 7) GNN
    # ----------------
    print("Training GNN ensemble (PCC graphs)...")
    try:
        t_train = time.time()
        res_gnn = _graph_train_cv("gnn", graphs=train_graphs,
                                  n_splits=n_splits, torch_epochs=int(getattr(cfg, "GNN_EPOCHS", 100)),
                                  device=device, run_dir=run_dir, lr=float(getattr(cfg, "GNN_LR", 5e-3)),
                                  weight_decay=0.0, patience=int(getattr(cfg, "PATIENCE", 10)), seed=seed,
                                  model_kwargs={
                                      "hidden_channels": 64,
                                      "num_layers": 3,
                                      "dropout": 0.3,
                                  },
                                  **fixed_graph_val_kwargs)
        trained["gnn"] = res_gnn
        training_times["gnn"] = time.time() - t_train
        _append_cv_metrics("gnn", res_gnn)
    except Exception as e:
        print("gnn train_cv failed:", e)

    # ----------------
    # 8) GraphSAGE
    # ----------------

    print("Training GraphSAGE (PCC graphs)...")
    try:
        t_train = time.time()
        res_graphsage = _graph_train_cv(
            "graphsage",
            graphs=train_graphs,
            n_splits=n_splits,
            torch_epochs=int(getattr(cfg, "GRAPH_SAGE_EPOCHS", 100)),
            device=device,
            run_dir=run_dir,
            lr=float(getattr(cfg, "GRAPH_SAGE_LR", 5e-3)),
            weight_decay=0.0,
            patience=int(getattr(cfg, "GRAPH_SAGE_PATIENCE", getattr(cfg, "PATIENCE", 10))),
            seed=seed,
            model_kwargs={
                "hidden_channels": 64,
                "num_layers": 3,
                "dropout": 0.3,
            },
            **fixed_graph_val_kwargs
        )
        trained["graphsage"] = res_graphsage
        training_times["graphsage"] = time.time() - t_train
        _append_cv_metrics("graphsage", res_graphsage)
    except Exception as e:
        print("graphsage train_cv failed:", e)

    # ----------------
    # 9) GAT + GIN
    # ----------------
    print("Training GAT (PCC graphs)...")
    try:
        t_train = time.time()
        res_gat = _graph_train_cv("gat", graphs=train_graphs,
                                  n_splits=n_splits, torch_epochs=int(getattr(cfg, "GAT_EPOCHS", 100)),
                                  device=device, run_dir=run_dir, lr=float(getattr(cfg, "GAT_LR", 5e-3)),
                                  weight_decay=0.0, patience=int(getattr(cfg, "PATIENCE", 10)), seed=seed,
                                  model_kwargs={
                                      "hidden_channels": 64,
                                      "num_layers": 3,
                                      "dropout": 0.3,
                                      "heads": 4,
                                  },
                                  **fixed_graph_val_kwargs)
        trained["gat"] = res_gat
        training_times["gat"] = time.time() - t_train
        _append_cv_metrics("gat", res_gat)
    except Exception as e:
        print("gat train_cv failed:", e)

    print("Training GIN (PCC graphs)...")
    try:
        t_train = time.time()
        res_gin = _graph_train_cv("gin", graphs=train_graphs,
                                  n_splits=n_splits, torch_epochs=int(getattr(cfg, "GIN_EPOCHS", 100)),
                                  device=device, run_dir=run_dir, lr=float(getattr(cfg, "GIN_LR", 5e-3)),
                                  weight_decay=0.0, patience=int(getattr(cfg, "PATIENCE", 10)), seed=seed,
                                  model_kwargs={
                                      "hidden_channels": 64,
                                      "num_layers": 3,
                                      "dropout": 0.3,
                                  },
                                  **fixed_graph_val_kwargs)
        trained["gin"] = res_gin
        training_times["gin"] = time.time() - t_train
        _append_cv_metrics("gin", res_gin)
    except Exception as e:
        print("gin train_cv failed:", e)

    if bool(getattr(cli_flags, "save_pretrained", False)):
        pretrained_dir = os.path.abspath(str(getattr(cli_flags, "pretrained_dir", getattr(cfg, "BASELINE_SAVE_DIR", os.path.join("save", "baseline")))))
        os.makedirs(pretrained_dir, exist_ok=True)
        for name, info in trained.items():
            src_paths = [p for p in (info.get("paths") or []) if isinstance(p, str) and os.path.isfile(p)]
            if not src_paths:
                continue
            dst_dir = os.path.join(pretrained_dir, str(name))
            if os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir)
            os.makedirs(dst_dir, exist_ok=True)
            for src in src_paths:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        print(f"[INFO] Saved baseline pre-trained checkpoints to: {pretrained_dir}")

    all_results = []
    for name in VECTOR_BACKBONES:
        if name not in trained:
            continue
        info = trained[name]
        model_objs = info["models"]
        try:
            rmse, mae, it = evaluate_models(model_objs, X_test, y_test, scaler=scaler, target_idx=target_idx, device=device)
            print(f"Eval {name} Original -> RMSE {rmse:.6g}, MAE {mae:.6g}")
        except Exception as e:
            print(f"Warning: evaluate_models failed for {name} original:", e)
            rmse = mae = it = float("nan")
        all_results.append({
            "Method": _display_name(name),
            "Imputation_Method": "",
            "Category": "Original",
            "Dataset_Index": 1,
            "RMSE": float(rmse),
            "MAE": float(mae),
            "Inference_Time": float(it),
            "Training_Time": float(training_times.get(name, np.nan))
        })

    # ------------ Graph-based ORIGINAL evaluation ------------
    def _eval_and_append(trained_key, sets, ground_truth_graph, base_model_key, category, imputation_method=None, display_name=None):
        if trained_key not in trained:
            return
        if not sets:
            return
        try:
            fold_states = trained[trained_key].get("models", [])
            fold_states = [fs for fs in (fold_states or []) if fs is not None]
            if len(fold_states) == 0:
                print(f"[WARN] no valid folds for {trained_key}; skipping")
                return

            model_cls_name = _MODEL_KEY_TO_CLASSNAME.get(trained_key, None)
            if model_cls_name is None:
                model_cls_name = _MODEL_KEY_TO_CLASSNAME.get(base_model_key, None)
            base_model_class = None
            if model_cls_name is not None:
                base_model_class = getattr(models, model_cls_name, None)
            if base_model_class is None:
                base_model_class = models.__dict__.get(str(base_model_key).lower(), None) or getattr(models, str(base_model_key), None)

            if base_model_class is None:
                print(f"[WARN] could not resolve base model class for {trained_key} / {base_model_key}; skipping")
                return

            model_in = trained[trained_key].get("model_in", None)
            model_class = base_model_class

            raw_rows = evaluate_gnn_ensemble(
                sets, scaler=scaler, method_name=f"{trained_key}_{category}",
                category=category, ground_truth_graph=ground_truth_graph,
                fold_state_dicts=fold_states,
                model_class=model_class,
                device=device, target_node_idx=target_idx, model_in=model_in
            )
            if raw_rows:
                all_results.extend(_normalize_eval_rows(
                    raw_rows,
                    base_method_name=str(display_name).strip() if display_name else _display_name(base_model_key),
                    category=category,
                    imputation_method=imputation_method,
                    training_time=training_times.get(trained_key, np.nan)
                ))
            else:
                print(f"[INFO] evaluate_gnn_ensemble returned 0 rows for {trained_key} / {category}")
        except Exception as e:
            print(f"Warning: evaluation for {trained_key} failed:", e)

    graph_eval_specs = [
        {"trained_key": key, "model_key": key, "display_name": _display_name(key)}
        for key in GRAPH_BACKBONES
    ]

    for spec in graph_eval_specs:
        _eval_and_append(
            spec["trained_key"], [test_graphs], test_graphs,
            spec["model_key"], "Original", display_name=spec.get("display_name")
        )

    # -------------------------
    # DF-level imputed datasets: vector backbones
    # -------------------------
    print("Evaluating DF-level imputed datasets (vector backbones)...")
    for backbone in VECTOR_BACKBONES:
        if backbone not in trained:
            continue
        model_objs = trained[backbone]["models"]
        display = _display_name(backbone)
        for key, recs in (imputed_records or {}).items():
            parts = key.split("_", 1)
            category = parts[0] if len(parts) > 0 else ""
            impute_method = parts[1] if len(parts) > 1 else key
            for rec in recs:
                set_idx = rec.get("set_idx", None)
                df = rec.get("df", None)
                if df is None:
                    skipped_impute_reasons[impute_method].append("df_none")
                    continue
                if df.shape[0] > 0 and df.iloc[0].apply(lambda x: isinstance(x, str)).any():
                    df = df.drop(index=0).reset_index(drop=True)
                try:
                    vals = df.values.astype(float)
                except Exception:
                    skipped_impute_reasons[impute_method].append("astype_float_failed")
                    continue
                if vals.shape[1] != scaler.n_features_in_:
                    skipped_impute_reasons[impute_method].append(f"ncols_mismatch:{vals.shape[1]}")
                    continue
                if vals.shape[0] != X_test.shape[0]:
                    try:
                        test_idx = metadata["indices"]["test"]
                        vals = vals[test_idx]
                    except Exception:
                        skipped_impute_reasons[impute_method].append("rowcount_mismatch_and_indexing_failed")
                        continue
                vals_norm = scaler.transform(vals)
                X_imp = np.delete(vals_norm, target_idx, axis=1).astype(np.float32)
                y_imp = vals_norm[:, target_idx].astype(np.float32)
                if np.isnan(X_imp).any() or np.isnan(y_imp).any():
                    rmse_i = mae_i = it_i = float("nan")
                else:
                    try:
                        rmse_i, mae_i, it_i = evaluate_models(model_objs, X_imp, y_imp, scaler=scaler, target_idx=target_idx, device=device)
                    except Exception as e:
                        print(f"[WARN] evaluate_models failed for {display} on imputed {key} set_idx={set_idx}: {e}")
                        rmse_i = mae_i = it_i = float("nan")
                all_results.append({
                    "Method": display,
                    "Imputation_Method": canonical_imputation_name(impute_method),
                    "Category": category,
                    "Dataset_Index": int(set_idx) if set_idx is not None else None,
                    "RMSE": float(rmse_i),
                    "MAE": float(mae_i),
                    "Inference_Time": float(it_i),
                    "Training_Time": float(training_times.get(backbone, np.nan))
                })
                processed_impute_counts[impute_method] += 1

    # -------------------------
    # DF-level missing datasets: vector backbones
    # -------------------------
    print("Evaluating DF-level missing datasets (vector backbones)...")
    for backbone in VECTOR_BACKBONES:
        if backbone not in trained:
            continue
        model_objs = trained[backbone]["models"]
        display = _display_name(backbone)
        for category, scen_map in (missing_records or {}).items():
            flat = []
            for _, scen_payload in scen_map.items():
                if isinstance(scen_payload, tuple) and len(scen_payload) > 0:
                    df_list = scen_payload[0]
                else:
                    df_list = scen_payload
                if isinstance(df_list, list):
                    flat.extend(df_list)
            for idx, df_set in enumerate(flat, start=1):
                if df_set is None:
                    skipped_missing_reasons[category].append("df_none")
                    continue
                df_work = df_set.copy()
                if df_work.shape[0] > 0 and df_work.iloc[0].apply(lambda x: isinstance(x, str)).any():
                    df_work = df_work.drop(index=0).reset_index(drop=True)
                df_work = df_work.fillna(-1.0)
                try:
                    vals = df_work.values.astype(float)
                except Exception:
                    try:
                        df_num = df_work.apply(pd.to_numeric, errors="coerce")
                        df_num = df_num.dropna(axis=1, how="all")
                        vals = df_num.values.astype(float)
                    except Exception:
                        skipped_missing_reasons[category].append("astype_float_failed")
                        continue
                if vals.shape[1] != scaler.n_features_in_:
                    skipped_missing_reasons[category].append(f"ncols_mismatch:{vals.shape[1]}")
                    continue
                if vals.shape[0] != X_test.shape[0]:
                    try:
                        test_idx = metadata['indices']['test']
                        vals = vals[test_idx]
                    except Exception:
                        skipped_missing_reasons[category].append("rowcount_mismatch_and_indexing_failed")
                        continue
                vals_norm = scaler.transform(vals)
                X_miss = np.delete(vals_norm, target_idx, axis=1).astype(np.float32)
                y_miss = vals_norm[:, target_idx].astype(np.float32)
                if np.isnan(X_miss).any() or np.isnan(y_miss).any():
                    rmse_m = mae_m = it_m = float("nan")
                else:
                    try:
                        rmse_m, mae_m, it_m = evaluate_models(model_objs, X_miss, y_miss, scaler=scaler, target_idx=target_idx, device=device)
                    except Exception as e:
                        print(f"[WARN] evaluate_models failed for {display} on missing {category} idx={idx}: {e}")
                        rmse_m = mae_m = it_m = float("nan")
                all_results.append({
                    "Method": display,
                    "Imputation_Method": "Missing",
                    "Category": category,
                    "Dataset_Index": idx,
                    "RMSE": float(rmse_m),
                    "MAE": float(mae_m),
                    "Inference_Time": float(it_m),
                    "Training_Time": float(training_times.get(backbone, np.nan))
                })
                processed_missing_counts[category] += 1

    # -------------------------
    # Graph-based imputed graph-sets
    # -------------------------
    print("Evaluating graph-based imputed graph-sets (PCC)...")

    def _has_nonempty_sets(list_of_sets):
        if not list_of_sets:
            return False
        for s in list_of_sets:
            if isinstance(s, list) and len(s) > 0:
                return True
        return False

    filtered_imputed_non = {k: v for k, v in (imputed_graph_sets_non or {}).items() if _has_nonempty_sets(v)}

    if not filtered_imputed_non:
        print("[INFO] No imputed graph sets found after filtering.")

    for key, sets in filtered_imputed_non.items():
        parts = key.split("_", 1)
        category = parts[0] if len(parts) > 0 else ""
        impute_method = parts[1] if len(parts) > 1 else key


        for spec in graph_eval_specs:
            _eval_and_append(
                spec["trained_key"], sets, test_graphs, spec["model_key"],
                category, imputation_method=impute_method, display_name=spec.get("display_name")
            )

    # -------------------------
    # Graph-based missing datasets
    # -------------------------
    print("Evaluating graph-based missing datasets...")

    filtered_missing_non = {"MNAR": {}, "MCAR": {}}
    for topk, scen_map in (missing_graph_sets_non or {}).items():
        for scen, sets in (scen_map or {}).items():
            if _has_nonempty_sets(sets):
                filtered_missing_non.setdefault(topk, {})[scen] = sets

    if not any(filtered_missing_non.values()):
        print("[INFO] No missing graph sets found after filtering.")

    for topk, scen_map in (filtered_missing_non or {}).items():
        for scen, sets in scen_map.items():
            for spec in graph_eval_specs:
                _eval_and_append(
                    spec["trained_key"], sets, test_graphs, spec["model_key"],
                    topk, imputation_method=scen, display_name=spec.get("display_name")
                )
    # -------------------------
    # Save per-set results
    # -------------------------
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

    if bool(getattr(cfg, "SAVE_CV_METRICS", True)):
        cv_cols = ["Trained_Key", "Method", "Fold", "RMSE", "MAE", "Best_Objective", "Training_Time"]
        cv_df = pd.DataFrame(cv_rows)
        for c in cv_cols:
            if c not in cv_df.columns:
                cv_df[c] = np.nan
        if "Graph" in cv_df.columns:
            cv_df = cv_df.drop(columns=["Graph"])
        cv_path = os.path.join(run_dir, "cv_metrics.csv")
        cv_df.to_csv(cv_path, index=False)
        print("Saved CV metrics to", cv_path)
        
    print("\nImputation processing summary (sample):")
    for im in getattr(cfg, "IMPUTATION_METHODS", []):
        cnt = processed_impute_counts.get(im, 0)
        print(f"  {im}: processed_sets={cnt}")

    print("\nMissing datasets processing summary (sample):")
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
            TT_per_dataset=("Training_Time", "mean")
        ).reset_index()

        def agg_stats(vals):
            if vals is None or len(vals) == 0:
                return (np.nan, np.nan)
            arr = np.asarray(vals, dtype=float)
            meanv = float(np.nanmean(arr))
            sdv = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            return (meanv, sdv)

        out_rows = []
        backbone_names = [_display_name(bk) for bk in BACKBONE_KEYS]
        existing_methods = [str(m) for m in df_work["Method"].dropna().astype(str).unique().tolist()]
        for m in sorted(existing_methods):
            if m not in backbone_names:
                backbone_names.append(m)
        for backbone_name in backbone_names:
            method_mask = per_dataset_avg["Method"].str.upper() == backbone_name.upper()
            mask_orig = (per_dataset_avg["Category"].astype(str).str.upper() == "ORIGINAL") & method_mask
            orig_rmse = float(per_dataset_avg.loc[mask_orig, "RMSE_per_dataset"].astype(float).mean()) if mask_orig.any() else np.nan
            orig_mae = float(per_dataset_avg.loc[mask_orig, "MAE_per_dataset"].astype(float).mean()) if mask_orig.any() else np.nan
            orig_it = float(per_dataset_avg.loc[mask_orig, "IT_per_dataset"].astype(float).mean()) if mask_orig.any() else np.nan
            train_time = float(per_dataset_avg.loc[method_mask, "TT_per_dataset"].astype(float).mean()) if method_mask.any() else np.nan

            for impute in getattr(cfg, "IMPUTATION_METHODS", []):
                mask_mnar_missing = method_mask & \
                                    (per_dataset_avg["Category"].str.upper() == "MNAR") & \
                                    (~per_dataset_avg["Imputation_Method"].astype(str).str.upper().isin({m.upper() for m in getattr(cfg, "IMPUTATION_METHODS", [])}))
                mnar_miss_vals_rmse = per_dataset_avg.loc[mask_mnar_missing, "RMSE_per_dataset"].astype(float).values if mask_mnar_missing.any() else np.array([], dtype=float)
                mnar_miss_vals_mae = per_dataset_avg.loc[mask_mnar_missing, "MAE_per_dataset"].astype(float).values if mask_mnar_missing.any() else np.array([], dtype=float)
                mnar_miss_vals_it = per_dataset_avg.loc[mask_mnar_missing, "IT_per_dataset"].astype(float).values if mask_mnar_missing.any() else np.array([], dtype=float)
                mnar_miss_rmse_mean, mnar_miss_rmse_sd = agg_stats(mnar_miss_vals_rmse)
                mnar_miss_mae_mean, mnar_miss_mae_sd = agg_stats(mnar_miss_vals_mae)
                mnar_miss_it_mean, _ = agg_stats(mnar_miss_vals_it)

                mask_mnar_imp = method_mask & \
                                (per_dataset_avg["Category"].str.upper() == "MNAR") & \
                                (per_dataset_avg["Imputation_Method"].astype(str).str.upper() == impute.upper())
                mnar_imp_vals_rmse = per_dataset_avg.loc[mask_mnar_imp, "RMSE_per_dataset"].astype(float).values if mask_mnar_imp.any() else np.array([], dtype=float)
                mnar_imp_vals_mae = per_dataset_avg.loc[mask_mnar_imp, "MAE_per_dataset"].astype(float).values if mask_mnar_imp.any() else np.array([], dtype=float)
                mnar_imp_vals_it = per_dataset_avg.loc[mask_mnar_imp, "IT_per_dataset"].astype(float).values if mask_mnar_imp.any() else np.array([], dtype=float)
                mnar_imp_rmse_mean, mnar_imp_rmse_sd = agg_stats(mnar_imp_vals_rmse)
                mnar_imp_mae_mean, mnar_imp_mae_sd = agg_stats(mnar_imp_vals_mae)
                mnar_imp_it_mean, _ = agg_stats(mnar_imp_vals_it)

                mask_mcar_missing = method_mask & \
                                    (per_dataset_avg["Category"].str.upper() == "MCAR") & \
                                    (~per_dataset_avg["Imputation_Method"].astype(str).str.upper().isin({m.upper() for m in getattr(cfg, "IMPUTATION_METHODS", [])}))
                mcar_miss_vals_rmse = per_dataset_avg.loc[mask_mcar_missing, "RMSE_per_dataset"].astype(float).values if mask_mcar_missing.any() else np.array([], dtype=float)
                mcar_miss_vals_mae = per_dataset_avg.loc[mask_mcar_missing, "MAE_per_dataset"].astype(float).values if mask_mcar_missing.any() else np.array([], dtype=float)
                mcar_miss_vals_it = per_dataset_avg.loc[mask_mcar_missing, "IT_per_dataset"].astype(float).values if mask_mcar_missing.any() else np.array([], dtype=float)
                mcar_miss_rmse_mean, mcar_miss_rmse_sd = agg_stats(mcar_miss_vals_rmse)
                mcar_miss_mae_mean, mcar_miss_mae_sd = agg_stats(mcar_miss_vals_mae)
                mcar_miss_it_mean, _ = agg_stats(mcar_miss_vals_it)

                mask_mcar_imp = method_mask & \
                                (per_dataset_avg["Category"].str.upper() == "MCAR") & \
                                (per_dataset_avg["Imputation_Method"].astype(str).str.upper() == impute.upper())
                mcar_imp_vals_rmse = per_dataset_avg.loc[mask_mcar_imp, "RMSE_per_dataset"].astype(float).values if mask_mcar_imp.any() else np.array([], dtype=float)
                mcar_imp_vals_mae = per_dataset_avg.loc[mask_mcar_imp, "MAE_per_dataset"].astype(float).values if mask_mcar_imp.any() else np.array([], dtype=float)
                mcar_imp_vals_it = per_dataset_avg.loc[mask_mcar_imp, "IT_per_dataset"].astype(float).values if mask_mcar_imp.any() else np.array([], dtype=float)
                mcar_imp_rmse_mean, mcar_imp_rmse_sd = agg_stats(mcar_imp_vals_rmse)
                mcar_imp_mae_mean, mcar_imp_mae_sd = agg_stats(mcar_imp_vals_mae)
                mcar_imp_it_mean, _ = agg_stats(mcar_imp_vals_it)

                out_rows.append({
                    "Method": backbone_name,
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
                    "MCAR_Imputed_Inference_Time_Mean": mcar_imp_it_mean
                })
            out_rows.append({
                "Method": backbone_name,
                "Imputation Method": "Missing",
                "Training_Time": train_time,
                "Original_RMSE": orig_rmse,
                "Original_MAE": orig_mae,
                "Original_Inference_Time": orig_it,
                "MNAR_Missing_RMSE_Mean": mnar_miss_rmse_mean,
                "MNAR_Missing_RMSE_SD": mnar_miss_rmse_sd,
                "MNAR_Missing_MAE_Mean": mnar_miss_mae_mean,
                "MNAR_Missing_MAE_SD": mnar_miss_mae_sd,
                "MNAR_Missing_Inference_Time_Mean": mnar_miss_it_mean,
                "MNAR_Imputed_RMSE_Mean": np.nan,
                "MNAR_Imputed_RMSE_SD": np.nan,
                "MNAR_Imputed_MAE_Mean": np.nan,
                "MNAR_Imputed_MAE_SD": np.nan,
                "MNAR_Imputed_Inference_Time_Mean": np.nan,
                "MCAR_Missing_RMSE_Mean": mcar_miss_rmse_mean,
                "MCAR_Missing_RMSE_SD": mcar_miss_rmse_sd,
                "MCAR_Missing_MAE_Mean": mcar_miss_mae_mean,
                "MCAR_Missing_MAE_SD": mcar_miss_mae_sd,
                "MCAR_Missing_Inference_Time_Mean": mcar_miss_it_mean,
                "MCAR_Imputed_RMSE_Mean": np.nan,
                "MCAR_Imputed_RMSE_SD": np.nan,
                "MCAR_Imputed_MAE_Mean": np.nan,
                "MCAR_Imputed_MAE_SD": np.nan,
                "MCAR_Imputed_Inference_Time_Mean": np.nan
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
        print("Backbone counts (sample):")
        print(combos.head(40).to_string(index=False))

    print("Done. run dir:", run_dir)
    pprint.pprint({k: len(v["models"]) for k, v in trained.items() if isinstance(v, dict) and "models" in v})
    print("Total elapsed (s):", time.time() - t0_global)

    print("\nSkipped imputation reasons (sample):")
    for k, reasons in skipped_impute_reasons.items():
        print(f"  {k}: count={len(reasons)} reasons_sample={reasons[:5]}")

    print("\nSkipped missing reasons (sample):")
    for k, reasons in skipped_missing_reasons.items():
        print(f"  {k}: count={len(reasons)} reasons_sample={reasons[:5]}")

    return run_dir

if __name__ == "__main__":
    main()
