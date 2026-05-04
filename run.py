import argparse
import shlex
import sys
from pathlib import Path


TASKS = ["baseline", "fusion", "stats", "visualization"]
DEFAULT_ORDER = list(TASKS)


def _split_arg_string(value):
    value = str(value or "").strip()
    if not value:
        return []
    parts = shlex.split(value, posix=False)
    return [
        part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"') else part
        for part in parts
    ]


def _restore_stdio():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def _normalize_tasks(raw_tasks, run_all=False):
    if run_all:
        return list(DEFAULT_ORDER)
    if not raw_tasks:
        return list(DEFAULT_ORDER)

    normalized = []
    for task in raw_tasks:
        key = str(task).strip().lower()
        if key == "all":
            return list(DEFAULT_ORDER)
        if key not in TASKS:
            valid = ", ".join(TASKS + ["all"])
            raise SystemExit(f"Unknown task '{task}'. Valid tasks: {valid}")
        if key not in normalized:
            normalized.append(key)
    return [task for task in DEFAULT_ORDER if task in normalized]


def _eval_csv_from_run(run_dir):
    if not run_dir:
        return None
    return Path(run_dir) / "eval_per_set.csv"


def _summary_from_run(run_dir):
    if not run_dir:
        return None
    return Path(run_dir)


def _default_stats_output_dir(baseline_run, fusion_run):
    if not baseline_run or not fusion_run:
        return None
    return Path(fusion_run) / f"analysis_vs_{Path(baseline_run).name}"


def _run_baseline(arg_string):
    from pipelines import baselines_pipeline

    argv = _split_arg_string(arg_string)
    print("\n[RUN] baseline", " ".join(argv))
    try:
        return baselines_pipeline.main(argv)
    finally:
        _restore_stdio()


def _run_fusion(arg_string):
    from pipelines import gtlf_pipeline

    argv = _split_arg_string(arg_string)
    print("\n[RUN] fusion", " ".join(argv))
    try:
        return gtlf_pipeline.main(argv)
    finally:
        _restore_stdio()


def _run_stats(arg_string, baseline_run=None, fusion_run=None, baseline_csv=None, fusion_csv=None):
    from analysis import statistical_analysis_impl

    argv = _split_arg_string(arg_string)
    if not argv:
        baseline_csv = Path(baseline_csv) if baseline_csv else _eval_csv_from_run(baseline_run)
        fusion_csv = Path(fusion_csv) if fusion_csv else _eval_csv_from_run(fusion_run)
        if baseline_csv is None or fusion_csv is None:
            raise SystemExit(
                "Stats need CSV inputs. Run baseline and fusion in the same command, "
                "or pass --baseline-csv/--fusion-csv, or pass --stats-args."
            )
        argv = ["--baseline-csv", str(baseline_csv), "--fuse-csv", str(fusion_csv)]

    print("\n[RUN] statistical_analysis", " ".join(argv))
    try:
        return statistical_analysis_impl.main(argv)
    finally:
        _restore_stdio()


def _run_visualization(argv):
    from analysis import visualize_baseline_vs_fused, visualize_significance_share, visualize_top3

    print("\n[RUN] visualization", " ".join(argv))
    try:
        if not argv:
            raise SystemExit("Visualization needs a command: top3, significance-share, or baseline-vs-fused.")
        command, *rest = argv
        if command == "top3":
            return visualize_top3.main(rest)
        if command == "significance-share":
            return visualize_significance_share.main(rest)
        if command == "baseline-vs-fused":
            return visualize_baseline_vs_fused.main(rest)
        raise SystemExit(
            "Unknown visualization command "
            f"'{command}'. Valid commands: top3, significance-share, baseline-vs-fused."
        )
    finally:
        _restore_stdio()


def _run_default_visualizations(baseline_run=None, fusion_run=None, imputation_method="CMILK", stats_output_dir=None):
    run_inputs = [str(p) for p in (_summary_from_run(baseline_run), _summary_from_run(fusion_run)) if p]
    if run_inputs:
        _run_visualization(["top3", *run_inputs, "--imputation-method", str(imputation_method)])

    if baseline_run and fusion_run:
        _run_visualization([
            "baseline-vs-fused",
            "--baseline-summary",
            str(baseline_run),
            "--fuse-summary",
            str(fusion_run),
            "--imputation-method",
            str(imputation_method),
        ])

    if stats_output_dir and Path(stats_output_dir).exists():
        _run_visualization(["significance-share", str(stats_output_dir)])


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Unified GTLF-BTP runner. Run baseline, fusion, statistical analysis, "
            "visualization, all tasks, or any combination."
        )
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        default=None,
        help="Tasks to run: baseline, fusion, stats, visualization, or all.",
    )
    parser.add_argument("--all", action="store_true", help="Run baseline, fusion, stats, and visualization.")
    parser.add_argument(
        "--baseline-args",
        default="",
        help='Quoted arguments passed to baselines_pipeline, for example "--no-cv".',
    )
    parser.add_argument(
        "--fusion-args",
        default="",
        help='Quoted arguments passed to gtlf_pipeline, for example "--epochs 50 --no-cv".',
    )
    parser.add_argument(
        "--stats-args",
        default="",
        help='Quoted arguments passed to statistical_analysis, for example "--metric MAE".',
    )
    parser.add_argument(
        "--visualization-args",
        action="append",
        default=[],
        help=(
            "Quoted visualization command. Can be repeated, for example "
            '"top3 out/run001" or "baseline-vs-fused --baseline-summary out/run001 --fuse-summary out/run002".'
        ),
    )
    parser.add_argument("--baseline-run", default=None, help="Existing baseline run directory for stats/visualization.")
    parser.add_argument("--fusion-run", default=None, help="Existing fusion run directory for stats/visualization.")
    parser.add_argument("--baseline-csv", default=None, help="Existing baseline eval_per_set.csv for stats.")
    parser.add_argument("--fusion-csv", default=None, help="Existing fusion eval_per_set.csv for stats.")
    parser.add_argument(
        "--imputation-method",
        default="CMILK",
        help="Imputation method used by default visualizations. Default: CMILK.",
    )
    parser.add_argument(
        "--skip-default-visualizations",
        action="store_true",
        help="When visualization is selected without --visualization-args, do not auto-generate default plots.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    tasks = _normalize_tasks(args.tasks, run_all=bool(args.all))

    baseline_run = args.baseline_run
    fusion_run = args.fusion_run
    stats_output_dir = None

    if "baseline" in tasks:
        baseline_run = _run_baseline(args.baseline_args)

    if "fusion" in tasks:
        fusion_run = _run_fusion(args.fusion_args)

    if "stats" in tasks:
        _run_stats(
            args.stats_args,
            baseline_run=baseline_run,
            fusion_run=fusion_run,
            baseline_csv=args.baseline_csv,
            fusion_csv=args.fusion_csv,
        )
        stats_output_dir = _default_stats_output_dir(baseline_run, fusion_run)

    if "visualization" in tasks:
        if args.visualization_args:
            for viz_args in args.visualization_args:
                _run_visualization(_split_arg_string(viz_args))
        elif not bool(args.skip_default_visualizations):
            _run_default_visualizations(
                baseline_run=baseline_run,
                fusion_run=fusion_run,
                imputation_method=args.imputation_method,
                stats_output_dir=stats_output_dir,
            )

    print("\n[DONE] Selected tasks completed.")
    if baseline_run:
        print("Baseline run:", baseline_run)
    if fusion_run:
        print("Fusion run:", fusion_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
