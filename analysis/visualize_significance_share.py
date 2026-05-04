from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEST_SPECS = [
    ("Paired t-test", "paired_t_pvalue"),
    ("Wilcoxon", "wilcoxon_pvalue"),
]


def _collect_summary_paths(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            found.append(path)
            continue
        if path.is_dir():
            local = sorted(path.glob("paired_stats_summary_*.csv"))
            if not local:
                local = sorted(path.glob("**/paired_stats_summary_*.csv"))
            found.extend(local)
            continue
        raise FileNotFoundError(f"Input path not found: {path}")
    unique = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    if not unique:
        raise FileNotFoundError("No paired_stats_summary_*.csv files were found.")
    return unique


def _infer_metric(path: Path, df: pd.DataFrame) -> str:
    if "metric" in df.columns:
        vals = [str(v).strip().upper() for v in df["metric"].dropna().unique().tolist() if str(v).strip()]
        if len(vals) == 1:
            return vals[0]
    m = re.search(r"paired_stats_summary_(.+)\.csv$", path.name, flags=re.IGNORECASE)
    if m:
        return str(m.group(1)).strip().upper()
    return path.stem.upper()


def _load_combined_summary(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        metric = _infer_metric(path, df)
        df = df.copy()
        df["metric"] = metric
        df["source_csv"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        raise ValueError("No rows were loaded from the provided summary CSVs.")
    return out


def _build_significance_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_order = []
    for metric in df["metric"].dropna().astype(str).str.upper().tolist():
        if metric not in metric_order:
            metric_order.append(metric)

    for metric in metric_order:
        sub = df.loc[df["metric"].astype(str).str.upper() == metric].copy()
        total_pairs = int(len(sub))
        if total_pairs <= 0:
            continue
        for test_name, p_col in TEST_SPECS:
            if p_col not in sub.columns:
                continue
            pvals = pd.to_numeric(sub[p_col], errors="coerce")
            significant_mask = pvals < float(alpha)
            significant_count = int(significant_mask.fillna(False).sum())
            missing_count = int(pvals.isna().sum())
            nonsignificant_count = int(total_pairs - significant_count)
            significant_pct = (100.0 * significant_count / total_pairs) if total_pairs > 0 else float("nan")
            nonsignificant_pct = (100.0 * nonsignificant_count / total_pairs) if total_pairs > 0 else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "test": test_name,
                    "pvalue_column": p_col,
                    "alpha": float(alpha),
                    "total_pairs": total_pairs,
                    "significant_count": significant_count,
                    "nonsignificant_count": nonsignificant_count,
                    "missing_pvalue_count": missing_count,
                    "significant_pct": significant_pct,
                    "nonsignificant_pct": nonsignificant_pct,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid significance summary rows could be created.")
    return out


def _plot_stacked_share(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["metric"].astype(str).str.upper() + "\n" + plot_df["test"].astype(str)

    x = np.arange(len(plot_df), dtype=float)
    sig = pd.to_numeric(plot_df["significant_pct"], errors="coerce").to_numpy(dtype=float)
    nonsig = pd.to_numeric(plot_df["nonsignificant_pct"], errors="coerce").to_numpy(dtype=float)

    sig_color = "#2ca02c"
    nonsig_color = "#bdbdbd"

    bars_sig = ax.bar(x, sig, color=sig_color, width=0.68, label=r"$p < 0.05$")
    bars_non = ax.bar(x, nonsig, bottom=sig, color=nonsig_color, width=0.68, label=r"$p \geq 0.05$ / NA")

    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Share of Analysis Pairs (%)", fontsize=13)
    ax.set_xlabel("Metric and Statistical Test", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"].tolist(), fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title(title, fontsize=17)
    ax.legend(loc="upper right", fontsize=11, frameon=True)

    for idx, (bar_sig, bar_non) in enumerate(zip(bars_sig, bars_non)):
        sig_pct = float(sig[idx]) if np.isfinite(sig[idx]) else 0.0
        total = int(plot_df.iloc[idx]["total_pairs"])
        sig_count = int(plot_df.iloc[idx]["significant_count"])
        txt = f"{sig_pct:.1f}%\n({sig_count}/{total})"
        if sig_pct >= 12.0:
            ax.text(
                float(bar_sig.get_x()) + (float(bar_sig.get_width()) / 2.0),
                float(bar_sig.get_height()) / 2.0,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                float(bar_sig.get_x()) + (float(bar_sig.get_width()) / 2.0),
                float(bar_sig.get_height()) + 3.0,
                txt,
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize the share of significant paired-comparison results."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more paired_stats_summary_*.csv files, or directories containing them. "
            "Example: out/run093(graph_tab_separate)/analysis_vs_run091"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold. Default: 0.05",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: the first input directory.",
    )
    args = parser.parse_args(argv)

    paths = _collect_summary_paths(args.inputs)
    combined_df = _load_combined_summary(paths)
    summary_df = _build_significance_summary(combined_df, alpha=float(args.alpha))

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        first = paths[0]
        out_dir = first.parent if first.is_file() else first
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "significance_share_summary.csv"
    png_path = out_dir / "significance_share_stacked.png"

    summary_df.to_csv(csv_path, index=False)
    _plot_stacked_share(
        summary_df=summary_df,
        out_path=png_path,
        title=r"Share of Significant Analysis Pairs ($p < 0.05$)",
    )

    print(f"Loaded {len(paths)} summary file(s).")
    print(f"Saved summary CSV to:  {csv_path}")
    print(f"Saved stacked plot to: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
