
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy import stats

from utils.utils import (
    GRAPH_BASELINE_CANDIDATES,
    TABULAR_BASELINE_CANDIDATES,
    is_missing_imputation_label as shared_is_missing_imputation_label,
    parse_fuse_method as shared_parse_fuse_method,
    resolve_first_available as shared_resolve_first_available,
)

KEY_COLS = ["Category", "Imputation_Method", "Dataset_Index"]
MISSING_IMPUTATION_LABELS = {"", "MISSING", "NONE", "NAN", "NON_OCCLUDED", "FACE_MASK", "GLASSES"}

def _resolve_first_available(candidates: Iterable[str], available: set[str]) -> str | None:
    return shared_resolve_first_available(candidates, available)


def _parse_fuse_method(method_name: str) -> tuple[str, str] | None:
    return shared_parse_fuse_method(method_name)


def _is_missing_imputation_label(label: str) -> bool:
    if str(label).strip().upper() in MISSING_IMPUTATION_LABELS:
        return True
    return bool(shared_is_missing_imputation_label(label))


def _filter_to_imputation_scope(
    df: pd.DataFrame,
    include_missing: bool,
    include_original: bool,
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    category_upper = work["Category"].astype(str).str.upper()
    missing_mask = work["Imputation_Method"].astype(str).map(_is_missing_imputation_label)
    original_mask = category_upper == "ORIGINAL"
    keep_mask = pd.Series(True, index=work.index, dtype=bool)
    if not include_missing:
        keep_mask = keep_mask & (~missing_mask)
    if not include_original:
        keep_mask = keep_mask & (~original_mask)
    return work.loc[keep_mask].copy()


def _load_eval_table(
    csv_path: Path,
    metric: str,
    include_missing: bool = False,
    include_original: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Method", *KEY_COLS, metric}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    out = df.copy()
    out["Method"] = out["Method"].astype(str).str.strip().str.upper()
    out["Category"] = out["Category"].fillna("").astype(str).str.strip().str.upper()
    out["Imputation_Method"] = out["Imputation_Method"].fillna("").astype(str).str.strip().str.upper()
    out["Dataset_Index"] = pd.to_numeric(out["Dataset_Index"], errors="coerce")
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=["Dataset_Index", metric]).copy()
    out["Dataset_Index"] = out["Dataset_Index"].astype(int)
    return _filter_to_imputation_scope(
        out,
        include_missing=bool(include_missing),
        include_original=bool(include_original),
    )


def _method_metric_frame(df: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    sub = df.loc[df["Method"] == method, KEY_COLS + [metric]].copy()
    if sub.empty:
        return sub
    dup_mask = sub.duplicated(subset=KEY_COLS, keep=False)
    if dup_mask.any():
        sub = sub.groupby(KEY_COLS, as_index=False)[metric].mean()
    return sub


def _paired_frame(
    baseline_df: pd.DataFrame,
    fuse_df: pd.DataFrame,
    baseline_method: str,
    fuse_method: str,
    metric: str,
    lower_is_better: bool,
) -> pd.DataFrame:
    base = _method_metric_frame(baseline_df, baseline_method, metric).rename(
        columns={metric: "baseline_value"}
    )
    fused = _method_metric_frame(fuse_df, fuse_method, metric).rename(
        columns={metric: "fuse_value"}
    )
    merged = base.merge(fused, on=KEY_COLS, how="inner", validate="one_to_one")
    if lower_is_better:
        merged["paired_improvement"] = merged["baseline_value"] - merged["fuse_value"]
    else:
        merged["paired_improvement"] = merged["fuse_value"] - merged["baseline_value"]
    return merged


def _paired_stats_from_frame(paired: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    baseline = paired["baseline_value"].to_numpy(dtype=float)
    fused = paired["fuse_value"].to_numpy(dtype=float)
    diff = paired["paired_improvement"].to_numpy(dtype=float)
    n = int(diff.size)
    mean_baseline = float(np.mean(baseline)) if n else float("nan")
    mean_fuse = float(np.mean(fused)) if n else float("nan")
    mean_improvement = float(np.mean(diff)) if n else float("nan")
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
    se_diff = float(sd_diff / math.sqrt(n)) if n > 1 and np.isfinite(sd_diff) else float("nan")

    if n > 1 and np.isfinite(sd_diff):
        t_crit = float(stats.t.ppf(1.0 - (alpha / 2.0), df=n - 1))
        ci_low = mean_improvement - (t_crit * se_diff)
        ci_high = mean_improvement + (t_crit * se_diff)
    elif n == 1:
        ci_low = mean_improvement
        ci_high = mean_improvement
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    if n == 0:
        t_stat = float("nan")
        t_p = float("nan")
    elif n == 1:
        t_stat = float("nan")
        t_p = float("nan")
    elif np.isclose(sd_diff, 0.0, atol=1e-15):
        if np.isclose(mean_improvement, 0.0, atol=1e-15):
            t_stat = 0.0
            t_p = 1.0
        else:
            t_stat = math.copysign(float("inf"), mean_improvement)
            t_p = 0.0
    else:
        t_res = stats.ttest_rel(baseline, fused, nan_policy="omit")
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

    if n == 0:
        wilcoxon_stat = float("nan")
        wilcoxon_p = float("nan")
    elif np.allclose(diff, 0.0, atol=1e-15):
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    else:
        try:
            w_res = stats.wilcoxon(
                baseline,
                fused,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="auto",
            )
        except TypeError:
            w_res = stats.wilcoxon(
                baseline,
                fused,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
            )
        wilcoxon_stat = float(w_res.statistic)
        wilcoxon_p = float(w_res.pvalue)

    if n == 0:
        cohen_d = float("nan")
    elif n == 1:
        cohen_d = float("nan")
    elif np.isclose(sd_diff, 0.0, atol=1e-15):
        if np.isclose(mean_improvement, 0.0, atol=1e-15):
            cohen_d = 0.0
        else:
            cohen_d = math.copysign(float("inf"), mean_improvement)
    else:
        cohen_d = float(mean_improvement / sd_diff)

    paired_t_significant = bool(np.isfinite(t_p) and (float(t_p) < float(alpha)))
    wilcoxon_significant = bool(np.isfinite(wilcoxon_p) and (float(wilcoxon_p) < float(alpha)))

    return {
        "n_pairs": n,
        "mean_baseline": mean_baseline,
        "mean_fuse": mean_fuse,
        "mean_paired_improvement": mean_improvement,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "paired_t_statistic": t_stat,
        "paired_t_pvalue": t_p,
        "paired_t_significant": paired_t_significant,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "wilcoxon_significant": wilcoxon_significant,
        "cohens_d": cohen_d,
    }


def build_summary(
    baseline_df: pd.DataFrame,
    fuse_df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    alpha: float,
    paired_categories: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_baselines = set(baseline_df["Method"].unique().tolist())
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    fuse_methods = sorted(fuse_df["Method"].unique().tolist())
    for paired_category in paired_categories:
        baseline_cat = baseline_df.loc[baseline_df["Category"] == paired_category].copy()
        fuse_cat = fuse_df.loc[fuse_df["Category"] == paired_category].copy()
        if baseline_cat.empty or fuse_cat.empty:
            continue

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

            comparisons = [
                ("graph_baseline", graph_baseline),
                ("tabular_baseline", tabular_baseline),
            ]
            for comparison_role, baseline_method in comparisons:
                if baseline_method is None:
                    continue
                paired = _paired_frame(
                    baseline_df=baseline_cat,
                    fuse_df=fuse_cat,
                    baseline_method=baseline_method,
                    fuse_method=fuse_method,
                    metric=metric,
                    lower_is_better=lower_is_better,
                )
                if paired.empty:
                    continue

                stats_row = _paired_stats_from_frame(paired, alpha=alpha)
                stats_row.update(
                    {
                        "paired_category": paired_category,
                        "comparison_role": comparison_role,
                        "baseline_method": baseline_method,
                        "fuse_method": fuse_method,
                        "graph_component": graph_prefix,
                        "tabular_component": tabular_suffix,
                        "metric": metric,
                        "comparison_label": f"{baseline_method} vs {fuse_method}",
                    }
                )
                summary_rows.append(stats_row)

                detail = paired.copy()
                detail["paired_category"] = paired_category
                detail["comparison_role"] = comparison_role
                detail["baseline_method"] = baseline_method
                detail["fuse_method"] = fuse_method
                detail["graph_component"] = graph_prefix
                detail["tabular_component"] = tabular_suffix
                detail["metric"] = metric
                detail_frames.append(detail)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df[
            [
                "paired_category",
                "comparison_role",
                "baseline_method",
                "fuse_method",
                "graph_component",
                "tabular_component",
                "metric",
                "n_pairs",
                "mean_baseline",
                "mean_fuse",
                "mean_paired_improvement",
                "ci_low",
                "ci_high",
                "paired_t_statistic",
                "paired_t_pvalue",
                "paired_t_significant",
                "wilcoxon_statistic",
                "wilcoxon_pvalue",
                "wilcoxon_significant",
                "cohens_d",
                "comparison_label",
            ]
        ].sort_values(["paired_category", "fuse_method", "comparison_role"]).reset_index(drop=True)

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_df, detail_df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and fusion eval_per_set.csv files with paired statistics."
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("out/run096/eval_per_set.csv"),
        help="Path to the baseline eval_per_set.csv file.",
    )
    parser.add_argument(
        "--fuse-csv",
        type=Path,
        default=Path(r"out/run095/eval_per_set.csv"),
        help="Path to the fusion eval_per_set.csv file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="RMSE",
        help="Metric column to compare. Default: RMSE",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated summary tables. Defaults to <fuse_run>/analysis_vs_<baseline_run>.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Interpret larger metric values as better. Default assumes lower-is-better.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the paired-improvement confidence interval. Default: 0.05",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help=(
            "Include raw missing-data rows in the paired comparison. "
            "By default, only Original and true imputation-method rows are compared."
        ),
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help=(
            "Also include Original rows in the paired comparison. "
            "By default, only imputed MNAR/MCAR rows are tested."
        ),
    )
    args = parser.parse_args(argv)

    baseline_csv = args.baseline_csv
    fuse_csv = args.fuse_csv
    metric = str(args.metric).strip()
    lower_is_better = not bool(args.higher_is_better)

    if not baseline_csv.exists():
        raise SystemExit(f"Baseline CSV not found: {baseline_csv}")
    if not fuse_csv.exists():
        raise SystemExit(f"Fusion CSV not found: {fuse_csv}")

    baseline_df = _load_eval_table(
        baseline_csv,
        metric=metric,
        include_missing=bool(args.include_missing),
        include_original=bool(args.include_original),
    )
    fuse_df = _load_eval_table(
        fuse_csv,
        metric=metric,
        include_missing=bool(args.include_missing),
        include_original=bool(args.include_original),
    )

    shared_categories = set(baseline_df["Category"].unique().tolist()) & set(fuse_df["Category"].unique().tolist())
    preferred_order = ["MNAR", "MCAR", "ORIGINAL"]
    paired_categories = [c for c in preferred_order if c in shared_categories]
    paired_categories.extend(sorted(c for c in shared_categories if c not in set(preferred_order)))

    summary_df, detail_df = build_summary(
        baseline_df=baseline_df,
        fuse_df=fuse_df,
        metric=metric,
        lower_is_better=lower_is_better,
        alpha=float(args.alpha),
        paired_categories=paired_categories,
    )

    if args.output_dir is None:
        output_dir = fuse_csv.parent / f"analysis_vs_{baseline_csv.parent.name}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"paired_stats_summary_{metric.lower()}.csv"
    detail_path = output_dir / f"paired_stats_detail_{metric.lower()}.csv"
    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    print(f"Baseline CSV: {baseline_csv}")
    print(f"Fusion CSV:    {fuse_csv}")
    print(f"Metric:        {metric}")
    print(
        "Scope:         imputed only"
        if (not bool(args.include_missing)) and (not bool(args.include_original))
        else (
            "Scope:         imputed + original"
            if (not bool(args.include_missing)) and bool(args.include_original)
            else "Scope:         imputed + original/raw missing as requested"
        )
    )
    print(f"Categories:    {', '.join(paired_categories) if paired_categories else '(none)'}")
    print(f"Improvement:   baseline - fuse" if lower_is_better else "Improvement:   fuse - baseline")
    print(f"Saved summary: {summary_path}")
    print(f"Saved detail:  {detail_path}")

    if summary_df.empty:
        print("No comparable baseline/fusion pairs were found.")
        return 0

    preview = summary_df[
        [
            "comparison_role",
            "baseline_method",
            "fuse_method",
            "n_pairs",
            "mean_paired_improvement",
            "ci_low",
            "ci_high",
            "paired_t_statistic",
            "paired_t_pvalue",
            "paired_t_significant",
            "wilcoxon_statistic",
            "wilcoxon_pvalue",
            "wilcoxon_significant",
            "cohens_d",
        ]
    ]
    print()
    print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
