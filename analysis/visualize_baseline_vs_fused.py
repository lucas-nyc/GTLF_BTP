from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils import (
    GRAPH_BASELINE_CANDIDATES,
    TABULAR_BASELINE_CANDIDATES,
    parse_fuse_method as shared_parse_fuse_method,
    pretty_fused_label,
    pretty_graph as shared_pretty_graph,
    pretty_tabular as shared_pretty_tabular,
    resolve_first_available as shared_resolve_first_available,
)


METRIC_COLS = {
    "MNAR": {
        "RMSE": "MNAR_Imputed_RMSE_Mean",
        "MAE": "MNAR_Imputed_MAE_Mean",
        "IT": "MNAR_Imputed_Inference_Time_Mean",
    },
    "MCAR": {
        "RMSE": "MCAR_Imputed_RMSE_Mean",
        "MAE": "MCAR_Imputed_MAE_Mean",
        "IT": "MCAR_Imputed_Inference_Time_Mean",
    },
}


def _summary_path_from_input(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_file():
        return raw
    return raw / "eval_summary_methods_wide.csv"


def _resolve_first_available(candidates: list[str], available: set[str]) -> str | None:
    return shared_resolve_first_available(candidates, available)


def _parse_fuse_method(method_name: str) -> tuple[str, str] | None:
    return shared_parse_fuse_method(method_name)


def _pretty_graph(method_name: str) -> str:
    return shared_pretty_graph(method_name)


def _pretty_tabular(method_name: str) -> str:
    return shared_pretty_tabular(method_name)


def _pretty_fused(graph_prefix: str, tabular_suffix: str) -> str:
    return pretty_fused_label(f"{graph_prefix}_{tabular_suffix}")


def _load_summary(summary_input: str, imputation_method: str | None) -> pd.DataFrame:
    summary_path = _summary_path_from_input(summary_input)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if "Method" not in df.columns:
        raise ValueError(f"{summary_path} is missing the 'Method' column.")
    if "Imputation Method" not in df.columns:
        df["Imputation Method"] = ""

    num_cols = sorted(
        {
            col
            for category in METRIC_COLS.values()
            for col in category.values()
        }
    )
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Method"] = df["Method"].astype(str).str.strip().str.upper()
    df["Imputation Method"] = df["Imputation Method"].fillna("").astype(str).str.strip().str.upper()

    if imputation_method:
        df = df.loc[df["Imputation Method"] == str(imputation_method).strip().upper()].copy()
    else:
        nonempty = sorted({x for x in df["Imputation Method"].unique().tolist() if str(x).strip()})
        if len(nonempty) == 1:
            df = df.loc[df["Imputation Method"] == nonempty[0]].copy()

    dup_mask = df.duplicated(subset=["Method"], keep=False)
    if dup_mask.any():
        agg_map = {col: "mean" for col in num_cols if col in df.columns}
        agg_map["Imputation Method"] = "first"
        df = df.groupby("Method", as_index=False).agg(agg_map)

    if df.empty:
        raise ValueError(f"No rows remain after filtering: {summary_path}")
    return df


def _build_reference_table(baseline_df: pd.DataFrame, fuse_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_baselines = set(baseline_df["Method"].unique().tolist())
    fuse_methods = sorted(fuse_df["Method"].unique().tolist())

    candidates: list[dict[str, object]] = []
    for fuse_method in fuse_methods:
        parsed = _parse_fuse_method(fuse_method)
        if parsed is None:
            continue
        graph_prefix, tabular_suffix = parsed
        graph_baseline = _resolve_first_available(
            GRAPH_BASELINE_CANDIDATES.get(graph_prefix, []),
            available_baselines,
        )
        tabular_baseline = _resolve_first_available(
            TABULAR_BASELINE_CANDIDATES.get(tabular_suffix, []),
            available_baselines,
        )
        if graph_baseline is None or tabular_baseline is None:
            continue
        row = fuse_df.loc[fuse_df["Method"] == fuse_method].head(1)
        if row.empty:
            continue
        mnar_rmse = pd.to_numeric(row["MNAR_Imputed_RMSE_Mean"], errors="coerce").iloc[0]
        if not np.isfinite(mnar_rmse):
            continue
        candidates.append(
            {
                "fuse_method": fuse_method,
                "graph_prefix": graph_prefix,
                "tabular_suffix": tabular_suffix,
                "graph_baseline_method": graph_baseline,
                "tabular_baseline_method": tabular_baseline,
                "reference_mnar_rmse": float(mnar_rmse),
                "reference_label": _pretty_fused(graph_prefix, tabular_suffix),
            }
        )

    ref_df = pd.DataFrame(candidates).sort_values("reference_mnar_rmse", ascending=True).head(int(max(1, top_k))).reset_index(drop=True)
    if ref_df.empty:
        raise ValueError("No fused methods with matched graph and tabular baselines were found.")
    ref_df["reference_rank"] = np.arange(1, len(ref_df) + 1)

    rows: list[dict[str, object]] = []
    role_specs = [
        ("Graph Baseline", "graph_baseline_method"),
        ("Tabular Baseline", "tabular_baseline_method"),
        ("Fused", "fuse_method"),
    ]
    for rec in ref_df.itertuples(index=False):
        for category in ("MNAR", "MCAR"):
            for role_name, field_name in role_specs:
                method_name = getattr(rec, field_name)
                source_df = fuse_df if role_name == "Fused" else baseline_df
                src_row = source_df.loc[source_df["Method"] == method_name].head(1)
                if src_row.empty:
                    continue
                rmse = pd.to_numeric(src_row[METRIC_COLS[category]["RMSE"]], errors="coerce").iloc[0]
                mae = pd.to_numeric(src_row[METRIC_COLS[category]["MAE"]], errors="coerce").iloc[0]
                it = pd.to_numeric(src_row[METRIC_COLS[category]["IT"]], errors="coerce").iloc[0]
                fps = (1.0 / float(it)) if np.isfinite(it) and float(it) > 0.0 else np.nan

                if role_name == "Graph Baseline":
                    pretty_label = _pretty_graph(method_name)
                    role_order = 0
                elif role_name == "Tabular Baseline":
                    pretty_label = _pretty_tabular(method_name)
                    role_order = 1
                else:
                    pretty_label = getattr(rec, "reference_label")
                    role_order = 2

                rows.append(
                    {
                        "reference_rank": int(getattr(rec, "reference_rank")),
                        "reference_label": getattr(rec, "reference_label"),
                        "category": category,
                        "role": role_name,
                        "role_order": role_order,
                        "method_name": method_name,
                        "display_label": pretty_label,
                        "rmse": rmse,
                        "mae": mae,
                        "inference_time": it,
                        "fps": fps,
                    }
                )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError("No comparison rows could be created for the selected references.")
    return ref_df, long_df


def _plot_category_dual(plot_df: pd.DataFrame, category: str, out_path: Path, top_k: int) -> None:
    sub = plot_df.loc[plot_df["category"].astype(str).str.upper() == str(category).strip().upper()].copy()
    if sub.empty:
        raise ValueError(f"No rows found for category: {category}")
    sub = sub.sort_values(["reference_rank", "role_order"]).reset_index(drop=True)

    title_fontsize = 18
    axis_title_fontsize = 14
    tick_fontsize = 11

    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    ax_right = ax.twinx()

    x = np.arange(len(sub), dtype=float)
    bar_w = 0.34

    rmse_vals = pd.to_numeric(sub["rmse"], errors="coerce").to_numpy(dtype=float)
    mae_vals = pd.to_numeric(sub["mae"], errors="coerce").to_numpy(dtype=float)
    fps_vals = pd.to_numeric(sub["fps"], errors="coerce").to_numpy(dtype=float)

    rmse_color = "#1f77b4"
    mae_color = "#ff7f0e"
    fps_color = "#d62728"

    bars_rmse = ax.bar(
        x - (bar_w / 2.0),
        rmse_vals,
        width=bar_w,
        color=rmse_color,
        alpha=0.88,
        label="RMSE",
    )
    bars_mae = ax.bar(
        x + (bar_w / 2.0),
        mae_vals,
        width=bar_w,
        color=mae_color,
        alpha=0.88,
        label="MAE",
    )
    fps_points = ax_right.plot(
        x,
        fps_vals,
        marker="o",
        linestyle="None",
        markersize=7,
        color=fps_color,
        label="FPS",
        zorder=3,
    )[0]

    for start in range(0, len(sub), 3):
        gx = x[start:start + 3]
        gy = fps_vals[start:start + 3]
        mask = np.isfinite(gy)
        if int(mask.sum()) >= 2:
            ax_right.plot(
                gx[mask],
                gy[mask],
                color=fps_color,
                linewidth=1.6,
                alpha=0.85,
                zorder=2,
            )

    metric_vals = np.concatenate([rmse_vals[np.isfinite(rmse_vals)], mae_vals[np.isfinite(mae_vals)]])
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
        lower_pad_fps = 0.12 * span_fps
        upper_pad_fps = 0.30 * span_fps
        ax_right.set_ylim(max(0.0, ymin_fps - lower_pad_fps), ymax_fps + upper_pad_fps)

    ax.set_xticks(x)
    ax.set_xticklabels(sub["display_label"].tolist(), rotation=0, ha="center", fontsize=tick_fontsize)
    ax.set_xlabel("Method", fontsize=axis_title_fontsize)
    ax.set_ylabel(r"$\overline{\mathrm{RMSE}}$ / $\overline{\mathrm{MAE}}$", fontsize=axis_title_fontsize)
    ax_right.set_ylabel("FPS", fontsize=axis_title_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax_right.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    y0, y1 = ax.get_ylim()
    metric_text_pad = 0.012 * max(1e-6, y1 - y0)
    for bar, val in zip(bars_rmse, rmse_vals):
        if np.isfinite(val):
            ax.text(
                float(bar.get_x()) + (float(bar.get_width()) / 2.0),
                float(val) + metric_text_pad,
                f"{float(val):.4f}",
                va="bottom",
                ha="center",
                fontsize=9,
            )
    for bar, val in zip(bars_mae, mae_vals):
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

    # Light separators between reference groups of 3 methods.
    for idx in range(3, len(sub), 3):
        ax.axvline(float(idx) - 0.5, color="#cccccc", linewidth=1.0, linestyle=":")

    handles = [bars_rmse, bars_mae, fps_points]
    labels = ["RMSE", "MAE", "FPS"]
    ax.legend(handles, labels, loc="upper right", frameon=True)
    ax.set_title(
        f"{category.upper()}: Top-{int(top_k)} Fused Methods vs Baseline",
        fontsize=title_fontsize,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize baseline vs fused metrics for the top-k fused MNAR-RMSE reference methods."
    )
    parser.add_argument(
        "--baseline-summary",
        type=str,
        required=True,
        help="Path to the baseline run directory or its eval_summary_methods_wide.csv.",
    )
    parser.add_argument(
        "--fuse-summary",
        type=str,
        required=True,
        help="Path to the fused run directory or its eval_summary_methods_wide.csv.",
    )
    parser.add_argument(
        "--imputation-method",
        type=str,
        default="",
        help="Optional imputation method filter, e.g. CMILK.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of fused reference methods to use. Default: 3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <fuse_run>/analysis_baseline_vs_fused",
    )
    args = parser.parse_args(argv)

    baseline_df = _load_summary(args.baseline_summary, imputation_method=(str(args.imputation_method).strip() or None))
    fuse_df = _load_summary(args.fuse_summary, imputation_method=(str(args.imputation_method).strip() or None))
    ref_df, long_df = _build_reference_table(baseline_df=baseline_df, fuse_df=fuse_df, top_k=int(max(1, args.top_k)))

    fuse_summary_path = _summary_path_from_input(args.fuse_summary)
    if args.output_dir is None:
        out_dir = fuse_summary_path.parent / "analysis_baseline_vs_fused"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_csv = out_dir / "baseline_vs_fused_topk_reference_methods.csv"
    long_csv = out_dir / "baseline_vs_fused_topk_metrics_long.csv"
    mnar_png = out_dir / "baseline_vs_fused_topk_mnar.png"
    mcar_png = out_dir / "baseline_vs_fused_topk_mcar.png"

    ref_df.to_csv(ref_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    _plot_category_dual(long_df, category="MNAR", out_path=mnar_png, top_k=int(max(1, args.top_k)))
    _plot_category_dual(long_df, category="MCAR", out_path=mcar_png, top_k=int(max(1, args.top_k)))

    print(f"Saved reference table to:   {ref_csv}")
    print(f"Saved long metrics table to:{long_csv}")
    print(f"Saved MNAR comparison plot: {mnar_png}")
    print(f"Saved MCAR comparison plot: {mcar_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
