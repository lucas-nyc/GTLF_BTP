
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.utils import pretty_fused_label


SCOPE_SPECS = {
    "RMSE": [
        ("Complete", "Original_RMSE"),
        ("MNAR Missing", "MNAR_Missing_RMSE_Mean"),
        ("MCAR Missing", "MCAR_Missing_RMSE_Mean"),
        ("MNAR Imputed", "MNAR_Imputed_RMSE_Mean"),
        ("MCAR Imputed", "MCAR_Imputed_RMSE_Mean"),
    ],
    "MAE": [
        ("Complete", "Original_MAE"),
        ("MNAR Missing", "MNAR_Missing_MAE_Mean"),
        ("MCAR Missing", "MCAR_Missing_MAE_Mean"),
        ("MNAR Imputed", "MNAR_Imputed_MAE_Mean"),
        ("MCAR Imputed", "MCAR_Imputed_MAE_Mean"),
    ],
}

COMBINED_IMPUTED_SPECS = [
    ("MNAR RMSE", "MNAR_Imputed_RMSE_Mean", "RMSE"),
    ("MCAR RMSE", "MCAR_Imputed_RMSE_Mean", "RMSE"),
    ("MNAR MAE", "MNAR_Imputed_MAE_Mean", "MAE"),
    ("MCAR MAE", "MCAR_Imputed_MAE_Mean", "MAE"),
]

REFERENCE_SCOPE_NAME = "MNAR RMSE"
REFERENCE_SCOPE_COL = "MNAR_Imputed_RMSE_Mean"
BRANCH_ONLY_SUFFIXES = ("_GRAPH_ONLY", "_TABULAR_ONLY")


def _summary_path_from_input(path_str: str) -> tuple[Path, Path]:
    raw = Path(path_str)
    if raw.is_file():
        return raw, raw.parent
    summary_path = raw / "eval_summary_methods_wide.csv"
    return summary_path, raw


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _is_branch_only_method(method_name: str) -> bool:
    name = str(method_name).strip().upper()
    return any(name.endswith(suffix) for suffix in BRANCH_ONLY_SUFFIXES)


def _pretty_method_label(method_name: str) -> str:
    label = pretty_fused_label(method_name)
    return label if label != str(method_name) else str(method_name).strip()


def _load_summary(
    summary_path: Path,
    imputation_method: str | None,
    include_branch_only: bool,
) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if "Method" not in df.columns:
        raise ValueError(f"{summary_path} is missing the 'Method' column.")
    if "Imputation Method" not in df.columns:
        df["Imputation Method"] = ""

    numeric_cols = sorted(
        {
            col
            for col in (
                [c for pairs in SCOPE_SPECS.values() for _, c in pairs]
                + [
                    "MNAR_Imputed_Inference_Time_Mean",
                    "MCAR_Imputed_Inference_Time_Mean",
                ]
            )
        }
    )
    df = _coerce_numeric(df, numeric_cols)
    df["Method"] = df["Method"].astype(str).str.strip()
    df["Imputation Method"] = df["Imputation Method"].fillna("").astype(str).str.strip()

    if not include_branch_only:
        df = df.loc[~df["Method"].map(_is_branch_only_method)].copy()

    if imputation_method:
        mask = df["Imputation Method"].str.upper() == str(imputation_method).strip().upper()
        df = df.loc[mask].copy()
    else:
        nonempty = sorted({x for x in df["Imputation Method"].unique().tolist() if str(x).strip()})
        if len(nonempty) == 1:
            df = df.loc[df["Imputation Method"].str.upper() == nonempty[0].upper()].copy()

    if df.empty:
        raise ValueError(f"No rows remain after filtering: {summary_path}")

    # If multiple rows per method remain, average numeric summary fields over them.
    dup_mask = df.duplicated(subset=["Method"], keep=False)
    if dup_mask.any():
        agg_map = {col: "mean" for col in numeric_cols if col in df.columns}
        agg_map["Imputation Method"] = "first"
        df = df.groupby("Method", as_index=False).agg(agg_map)

    return df


def _topk_frame(df: pd.DataFrame, metric_name: str, top_k: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scope_label, metric_col in SCOPE_SPECS[metric_name]:
        if metric_col not in df.columns:
            continue
        sub = df.loc[df[metric_col].notna(), ["Method", "Imputation Method", metric_col]].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(metric_col, ascending=True).head(int(max(1, top_k))).reset_index(drop=True)
        sub["Scope"] = scope_label
        sub["Metric"] = metric_name
        sub["Rank"] = np.arange(1, len(sub) + 1)
        sub = sub.rename(columns={metric_col: "Value"})
        rows.append(sub[["Metric", "Scope", "Rank", "Method", "Imputation Method", "Value"]])
    if not rows:
        return pd.DataFrame(columns=["Metric", "Scope", "Rank", "Method", "Imputation Method", "Value"])
    return pd.concat(rows, ignore_index=True)


def _reference_topk_frame(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    required_cols = ["Method", "Imputation Method", REFERENCE_SCOPE_COL]
    sub = df.loc[df[REFERENCE_SCOPE_COL].notna(), required_cols].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "Reference_Rank",
                "Method",
                "Imputation Method",
                "MNAR_RMSE",
                "MCAR_RMSE",
                "MNAR_MAE",
                "MCAR_MAE",
                "MNAR_IT",
                "MCAR_IT",
                "MNAR_FPS",
                "MCAR_FPS",
            ]
        )

    sub = sub.sort_values(REFERENCE_SCOPE_COL, ascending=True).head(int(max(1, top_k))).reset_index(drop=True)
    selected_methods = sub["Method"].tolist()
    selected = (
        df.loc[df["Method"].isin(selected_methods)].copy()
        .set_index("Method")
        .loc[selected_methods]
        .reset_index()
    )
    selected["Reference_Rank"] = np.arange(1, len(selected) + 1)
    out = selected[
        [
            "Reference_Rank",
            "Method",
            "Imputation Method",
            "MNAR_Imputed_RMSE_Mean",
            "MCAR_Imputed_RMSE_Mean",
            "MNAR_Imputed_MAE_Mean",
            "MCAR_Imputed_MAE_Mean",
            "MNAR_Imputed_Inference_Time_Mean",
            "MCAR_Imputed_Inference_Time_Mean",
        ]
    ].rename(
        columns={
            "MNAR_Imputed_RMSE_Mean": "MNAR_RMSE",
            "MCAR_Imputed_RMSE_Mean": "MCAR_RMSE",
            "MNAR_Imputed_MAE_Mean": "MNAR_MAE",
            "MCAR_Imputed_MAE_Mean": "MCAR_MAE",
            "MNAR_Imputed_Inference_Time_Mean": "MNAR_IT",
            "MCAR_Imputed_Inference_Time_Mean": "MCAR_IT",
        }
    )
    out["MNAR_FPS"] = np.where(
        pd.to_numeric(out["MNAR_IT"], errors="coerce") > 0.0,
        1.0 / pd.to_numeric(out["MNAR_IT"], errors="coerce"),
        np.nan,
    )
    out["MCAR_FPS"] = np.where(
        pd.to_numeric(out["MCAR_IT"], errors="coerce") > 0.0,
        1.0 / pd.to_numeric(out["MCAR_IT"], errors="coerce"),
        np.nan,
    )
    out["AVG_IT"] = np.nanmean(
        np.column_stack(
            [
                pd.to_numeric(out["MNAR_IT"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(out["MCAR_IT"], errors="coerce").to_numpy(dtype=float),
            ]
        ),
        axis=1,
    )
    out["FPS"] = np.where(
        pd.to_numeric(out["AVG_IT"], errors="coerce") > 0.0,
        1.0 / pd.to_numeric(out["AVG_IT"], errors="coerce"),
        np.nan,
    )
    return out


def _plot_metric_grid(
    topk_df: pd.DataFrame,
    metric_name: str,
    run_name: str,
    out_path: Path,
    top_k: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_flat = list(axes.flat)
    scope_order = [label for label, _ in SCOPE_SPECS[metric_name]]
    color = "#1f77b4" if metric_name == "RMSE" else "#ff7f0e"

    for idx, scope_label in enumerate(scope_order):
        ax = axes_flat[idx]
        sub = topk_df.loc[topk_df["Scope"] == scope_label].copy()
        ax.set_title(scope_label)
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue

        sub = sub.sort_values("Value", ascending=False)
        labels = [f"#{int(r)} {m}" for r, m in zip(sub["Rank"], sub["Method"])]
        bars = ax.barh(labels, sub["Value"], color=color, alpha=0.85)
        ax.set_xlabel(metric_name)
        ax.invert_yaxis()

        xmax = float(sub["Value"].max()) if len(sub) else 0.0
        pad = (0.04 * xmax) if xmax > 0 else 0.02
        ax.set_xlim(0.0, xmax + pad)
        for bar, val in zip(bars, sub["Value"]):
            ax.text(
                float(bar.get_width()) + pad * 0.15,
                float(bar.get_y()) + (float(bar.get_height()) / 2.0),
                f"{float(val):.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )

    for ax in axes_flat[len(scope_order):]:
        ax.set_axis_off()

    fig.suptitle(f"{run_name}: Top-{int(top_k)} {metric_name} by Evaluation Scope", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reference_imputed_grid(
    ref_df: pd.DataFrame,
    run_name: str,
    out_path: Path,
    top_k: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes_flat = list(axes.flat)
    color_map = {"RMSE": "#1f77b4", "MAE": "#ff7f0e"}

    for ax, (scope_label, value_col, metric_name) in zip(axes_flat, COMBINED_IMPUTED_SPECS):
        plot_col = value_col.replace("_Imputed_", "_").replace("_Mean", "")
        if plot_col not in ref_df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue

        sub = ref_df[["Reference_Rank", "Method", plot_col]].copy()
        sub = sub.loc[sub[plot_col].notna()].copy()
        ax.set_title(scope_label)
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue

        labels = [f"#{int(r)} {m}" for r, m in zip(sub["Reference_Rank"], sub["Method"])]
        bars = ax.barh(labels, sub[plot_col], color=color_map[metric_name], alpha=0.85)
        ax.set_xlabel(metric_name)
        ax.invert_yaxis()

        xmax = float(sub[plot_col].max()) if len(sub) else 0.0
        pad = (0.04 * xmax) if xmax > 0 else 0.02
        ax.set_xlim(0.0, xmax + pad)
        for bar, val in zip(bars, sub[plot_col]):
            ax.text(
                float(bar.get_width()) + pad * 0.15,
                float(bar.get_y()) + (float(bar.get_height()) / 2.0),
                f"{float(val):.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )

    fig.suptitle(
        f"{run_name}: Top-{int(top_k)} Imputed Methods (reference = {REFERENCE_SCOPE_NAME})",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reference_rmse_overlay(
    ref_df: pd.DataFrame,
    run_name: str,
    out_path: Path,
    top_k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    sub = ref_df[["Reference_Rank", "Method", "MNAR_RMSE", "MCAR_RMSE"]].copy()
    sub = sub.loc[sub["MNAR_RMSE"].notna() | sub["MCAR_RMSE"].notna()].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    labels = [f"#{int(r)} {m}" for r, m in zip(sub["Reference_Rank"], sub["Method"])]
    y = np.arange(len(sub), dtype=float)
    bar_h = 0.34

    mnar_vals = sub["MNAR_RMSE"].to_numpy(dtype=float)
    mcar_vals = sub["MCAR_RMSE"].to_numpy(dtype=float)

    ax.barh(y - (bar_h / 2.0), mnar_vals, height=bar_h, color="#1f77b4", alpha=0.9, label="MNAR")
    ax.barh(y + (bar_h / 2.0), mcar_vals, height=bar_h, color="#ff7f0e", alpha=0.9, label="MCAR")

    finite_vals = np.concatenate([mnar_vals[np.isfinite(mnar_vals)], mcar_vals[np.isfinite(mcar_vals)]])
    if finite_vals.size > 0:
        xmin = float(np.min(finite_vals))
        xmax = float(np.max(finite_vals))
        span = max(1e-6, xmax - xmin)
        pad = 0.12 * span
        ax.set_xlim(max(0.0, xmin - pad), xmax + pad)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Method")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_title(f"{run_name}: Top-{int(top_k)} Imputed RMSE (reference = {REFERENCE_SCOPE_NAME})")

    x0, x1 = ax.get_xlim()
    text_pad = 0.012 * max(1e-6, x1 - x0)
    for yy, val in zip(y - (bar_h / 2.0), mnar_vals):
        if np.isfinite(val):
            ax.text(float(val) + text_pad, float(yy), f"{float(val):.4f}", va="center", ha="left", fontsize=9)
    for yy, val in zip(y + (bar_h / 2.0), mcar_vals):
        if np.isfinite(val):
            ax.text(float(val) + text_pad, float(yy), f"{float(val):.4f}", va="center", ha="left", fontsize=9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reference_fps_lines(
    ref_df: pd.DataFrame,
    run_name: str,
    out_path: Path,
    top_k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    sub = ref_df[["Reference_Rank", "Method", "MNAR_FPS", "MCAR_FPS"]].copy()
    sub = sub.loc[sub["MNAR_FPS"].notna() | sub["MCAR_FPS"].notna()].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    x = np.asarray([0, 1], dtype=float)
    xticklabels = ["MNAR", "MCAR"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, row in enumerate(sub.itertuples(index=False), start=0):
        y = np.asarray([getattr(row, "MNAR_FPS"), getattr(row, "MCAR_FPS")], dtype=float)
        label = f"#{int(getattr(row, 'Reference_Rank'))} {getattr(row, 'Method')}"
        color = colors[idx % len(colors)]
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=6, color=color, label=label)
        for xx, yy in zip(x, y):
            if np.isfinite(yy):
                ax.text(float(xx), float(yy), f" {float(yy):.2f}", va="bottom", ha="left", fontsize=9, color=color)

    all_vals = np.concatenate(
        [
            pd.to_numeric(sub["MNAR_FPS"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(sub["MCAR_FPS"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size > 0:
        ymin = float(np.min(finite_vals))
        ymax = float(np.max(finite_vals))
        span = max(1e-6, ymax - ymin)
        pad = 0.12 * span
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Imputed Scenario")
    ax.set_ylabel("FPS (1 / inference time)")
    ax.set_title(f"{run_name}: Top-{int(top_k)} Imputed FPS (reference = {REFERENCE_SCOPE_NAME})")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reference_metric_fps_dual(
    ref_df: pd.DataFrame,
    run_name: str,
    out_path: Path,
    top_k: int,
    metric_name: str,
) -> None:
    metric_name = str(metric_name).strip().upper()
    if metric_name not in ("RMSE", "MAE"):
        raise ValueError(f"Unsupported metric for dual plot: {metric_name}")

    mnar_metric_col = f"MNAR_{metric_name}"
    mcar_metric_col = f"MCAR_{metric_name}"
    metric_color_mnar = "#1f77b4"
    metric_color_mcar = "#ff7f0e"
    fps_color = "#d62728"
    title_fontsize = 18
    axis_title_fontsize = 14
    tick_fontsize = 11

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    ax_right = ax.twinx()

    sub = ref_df[
        ["Reference_Rank", "Method", mnar_metric_col, mcar_metric_col, "FPS"]
    ].copy()
    sub = sub.loc[
        sub[mnar_metric_col].notna()
        | sub[mcar_metric_col].notna()
        | sub["FPS"].notna()
    ].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    labels = [_pretty_method_label(m) for m in sub["Method"]]
    x = np.arange(len(sub), dtype=float)
    bar_w = 0.34

    mnar_metric = pd.to_numeric(sub[mnar_metric_col], errors="coerce").to_numpy(dtype=float)
    mcar_metric = pd.to_numeric(sub[mcar_metric_col], errors="coerce").to_numpy(dtype=float)
    fps_vals = pd.to_numeric(sub["FPS"], errors="coerce").to_numpy(dtype=float)

    bars1 = ax.bar(
        x - (bar_w / 2.0),
        mnar_metric,
        width=bar_w,
        color=metric_color_mnar,
        alpha=0.88,
        label="MNAR",
    )
    bars2 = ax.bar(
        x + (bar_w / 2.0),
        mcar_metric,
        width=bar_w,
        color=metric_color_mcar,
        alpha=0.88,
        label="MCAR",
    )

    fps_line = ax_right.plot(
        x,
        fps_vals,
        marker="o",
        linestyle="None",
        markersize=7,
        color=fps_color,
        label="FPS",
    )[0]

    metric_vals = np.concatenate([mnar_metric[np.isfinite(mnar_metric)], mcar_metric[np.isfinite(mcar_metric)]])
    if metric_vals.size > 0:
        ymin = float(np.min(metric_vals))
        ymax = float(np.max(metric_vals))
        span = max(1e-6, ymax - ymin)
        pad = 0.12 * span
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

    finite_fps = fps_vals[np.isfinite(fps_vals)]
    if finite_fps.size > 0:
        ymin_fps = float(np.min(finite_fps))
        ymax_fps = float(np.max(finite_fps))
        span_fps = max(1e-6, ymax_fps - ymin_fps)
        pad_fps = 0.12 * span_fps
        ax_right.set_ylim(max(0.0, ymin_fps - pad_fps), ymax_fps + pad_fps)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=tick_fontsize)
    ax.set_xlabel("Method", fontsize=axis_title_fontsize)
    ax.set_ylabel(metric_name, fontsize=axis_title_fontsize)
    ax_right.set_ylabel("FPS", fontsize=axis_title_fontsize)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax_right.tick_params(axis="y", labelsize=tick_fontsize)

    y0, y1 = ax.get_ylim()
    metric_text_pad = 0.012 * max(1e-6, y1 - y0)
    for bar, val in zip(bars1, mnar_metric):
        if np.isfinite(val):
            ax.text(
                float(bar.get_x()) + (float(bar.get_width()) / 2.0),
                float(val) + metric_text_pad,
                f"{float(val):.4f}",
                va="bottom",
                ha="center",
                fontsize=9,
            )
    for bar, val in zip(bars2, mcar_metric):
        if np.isfinite(val):
            ax.text(
                float(bar.get_x()) + (float(bar.get_width()) / 2.0),
                float(val) + metric_text_pad,
                f"{float(val):.4f}",
                va="bottom",
                ha="center",
                fontsize=9,
            )

    fy0, fy1 = ax_right.get_ylim()
    fps_text_pad = 0.02 * max(1e-6, fy1 - fy0)
    for xx, yy in zip(x, fps_vals):
        if np.isfinite(yy):
            ax_right.text(
                float(xx),
                float(yy) + fps_text_pad,
                f"{float(yy):.2f}",
                va="bottom",
                ha="center",
                fontsize=9,
                color=fps_color,
            )

    handles = [bars1, bars2, fps_line]
    labels_legend = ["MNAR", "MCAR", "FPS"]
    ax.legend(handles, labels_legend, loc="best", frameon=True)
    ax.set_title(
        rf"Top-{int(top_k)} $\overline{{\mathrm{{{metric_name}}}}}$ & FPS",
        fontsize=title_fontsize,
    )

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _run_one(
    summary_input: str,
    top_k: int,
    imputation_method: str | None,
    output_dir: Path | None,
    include_branch_only: bool,
) -> None:
    summary_path, run_dir = _summary_path_from_input(summary_input)
    run_name = run_dir.name if run_dir.name else summary_path.stem
    summary_df = _load_summary(
        summary_path,
        imputation_method=imputation_method,
        include_branch_only=bool(include_branch_only),
    )

    if output_dir is None:
        out_dir = run_dir / "analysis_top3"
    else:
        out_dir = output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in ("RMSE", "MAE"):
        topk_df = _topk_frame(summary_df, metric_name=metric_name, top_k=top_k)
        csv_path = out_dir / f"top{int(top_k)}_{metric_name.lower()}.csv"
        png_path = out_dir / f"top{int(top_k)}_{metric_name.lower()}.png"
        topk_df.to_csv(csv_path, index=False)
        _plot_metric_grid(
            topk_df=topk_df,
            metric_name=metric_name,
            run_name=run_name,
            out_path=png_path,
            top_k=top_k,
        )
        print(f"Saved {metric_name} top-{int(top_k)} table to: {csv_path}")
        print(f"Saved {metric_name} top-{int(top_k)} plot to:  {png_path}")

    ref_df = _reference_topk_frame(summary_df, top_k=top_k)
    ref_csv_path = out_dir / f"top{int(top_k)}_reference_mnar_rmse_imputed.csv"
    ref_png_path = out_dir / f"top{int(top_k)}_reference_mnar_rmse_imputed.png"
    ref_overlay_path = out_dir / f"top{int(top_k)}_reference_mnar_rmse_overlay.png"
    ref_fps_path = out_dir / f"top{int(top_k)}_reference_mnar_fps_imputed.png"
    ref_rmse_dual_path = out_dir / f"top{int(top_k)}_reference_mnar_rmse_fps_dual.png"
    ref_mae_dual_path = out_dir / f"top{int(top_k)}_reference_mnar_mae_fps_dual.png"
    ref_df.to_csv(ref_csv_path, index=False)
    _plot_reference_imputed_grid(
        ref_df=ref_df,
        run_name=run_name,
        out_path=ref_png_path,
        top_k=top_k,
    )
    _plot_reference_rmse_overlay(
        ref_df=ref_df,
        run_name=run_name,
        out_path=ref_overlay_path,
        top_k=top_k,
    )
    _plot_reference_fps_lines(
        ref_df=ref_df,
        run_name=run_name,
        out_path=ref_fps_path,
        top_k=top_k,
    )
    _plot_reference_metric_fps_dual(
        ref_df=ref_df,
        run_name=run_name,
        out_path=ref_rmse_dual_path,
        top_k=top_k,
        metric_name="RMSE",
    )
    _plot_reference_metric_fps_dual(
        ref_df=ref_df,
        run_name=run_name,
        out_path=ref_mae_dual_path,
        top_k=top_k,
        metric_name="MAE",
    )
    print(f"Saved combined imputed reference table to: {ref_csv_path}")
    print(f"Saved combined imputed reference plot to:  {ref_png_path}")
    print(f"Saved RMSE overlay comparison plot to:    {ref_overlay_path}")
    print(f"Saved FPS line comparison plot to:       {ref_fps_path}")
    print(f"Saved RMSE + FPS dual-axis plot to:      {ref_rmse_dual_path}")
    print(f"Saved MAE + FPS dual-axis plot to:       {ref_mae_dual_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot top-k RMSE/MAE methods from eval_summary_methods_wide.csv."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "One or more run directories or direct eval_summary_methods_wide.csv paths. "
            "Example: out/run091 out/'run093(graph_tab_separate)'"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top methods to plot per scope. Default: 3",
    )
    parser.add_argument(
        "--imputation-method",
        type=str,
        default="",
        help=(
            "Optional imputation method filter, e.g. CMILK. "
            "If omitted and the summary contains exactly one non-empty imputation method, it is used automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional root output directory. If omitted, files are written to <run_dir>/analysis_top3. "
            "If provided with multiple runs, each run gets its own subdirectory."
        ),
    )
    parser.add_argument(
        "--include-branch-only",
        action="store_true",
        help="Include *_GRAPH_ONLY and *_TABULAR_ONLY methods. Default excludes them.",
    )
    args = parser.parse_args(argv)

    top_k = int(max(1, args.top_k))
    imputation_method = str(args.imputation_method).strip() or None

    for run_input in args.runs:
        _run_one(
            summary_input=run_input,
            top_k=top_k,
            imputation_method=imputation_method,
            output_dir=args.output_dir,
            include_branch_only=bool(args.include_branch_only),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
