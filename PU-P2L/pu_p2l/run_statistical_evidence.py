from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import MARGINAL_METHODS, read_csv, write_csv


DEFAULT_BASELINES = ["MaxLoss", "GREATS", "EL2N", "GraNdLast", "RHO-PretrainRef"]
DEFAULT_METRICS = [
    "test_inappropriate_risk",
    "test_error",
    "certified_bound",
    "effective_compression_size",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "group_revisit_rate",
    "unique_group_fraction",
    "runtime_sec",
    "selection_runtime_sec",
]
DEFAULT_CONDITION_FIELDS = [
    "dataset",
    "noise_rate",
    "pretrain_fraction",
    "pretrain_training_mode",
    "es_budget",
    "reference_method",
    "reference_es_budget",
]
LOWER_IS_BETTER = {
    "test_inappropriate_risk",
    "test_error",
    "certified_bound",
    "pac_bayes_bound",
    "self_selected_bound",
    "ada_bound",
    "compression_size",
    "remaining_bad",
    "effective_compression_size",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "pairwise_feature_cosine",
    "mean_support_redundancy",
    "max_support_redundancy",
    "local_redundancy_hit_rate",
    "residual_redundancy_hit_rate",
    "strong_redundancy_hit_rate",
    "group_revisit_rate",
    "max_group_selection_fraction",
    "runtime_sec",
    "selection_runtime_sec",
    "time_budget_overrun_sec",
    "time_budget_ratio",
}
HIGHER_IS_BETTER = {
    "stop_reached",
    "mode_entropy",
    "minority_mode_fraction",
    "spectral_entropy",
    "mean_selected_residual_novelty",
    "unique_group_fraction",
}
T_CRIT_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired statistical evidence from PU-P2L result CSVs. "
            "Rows are paired by seed within each experimental condition."
        )
    )
    parser.add_argument("--results-dir", type=str, action="append", default=[])
    parser.add_argument("--results", type=str, nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, default="results/PU-R/statistical_evidence")
    parser.add_argument("--target-method", type=str, default="PU-R")
    parser.add_argument("--baselines", type=str, nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--condition-fields", type=str, nargs="+", default=DEFAULT_CONDITION_FIELDS)
    parser.add_argument(
        "--row-filter",
        type=str,
        action="append",
        default=[],
        help=(
            "Keep only rows matching field=value[,value...] before pairing. "
            "May be passed more than once, e.g. --row-filter pretrain_fraction=0.0 "
            "--row-filter es_budget=100,200."
        ),
    )
    parser.add_argument("--include-step", action="store_true", help="Also pair by step for ES trace CSVs.")
    parser.add_argument("--include-marginal", action="store_true")
    parser.add_argument("--min-pairs", type=int, default=2)
    return parser.parse_args()


def to_float(value: Any) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_row_filters(filters: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in filters:
        if "=" not in item:
            raise ValueError(f"Invalid --row-filter {item!r}; expected field=value[,value...].")
        field, raw_values = item.split("=", 1)
        field = field.strip()
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not field or not values:
            raise ValueError(f"Invalid --row-filter {item!r}; expected field=value[,value...].")
        parsed.setdefault(field, []).extend(values)
    return parsed


def values_match(row_value: Any, allowed_values: list[str]) -> bool:
    row_text = "" if row_value is None else str(row_value)
    row_float = to_float(row_text)
    for allowed in allowed_values:
        allowed_float = to_float(allowed)
        if not math.isnan(row_float) and not math.isnan(allowed_float):
            if abs(row_float - allowed_float) <= 1e-12:
                return True
        elif row_text == allowed:
            return True
    return False


def filter_rows(rows: list[dict[str, str]], filters: dict[str, list[str]]) -> list[dict[str, str]]:
    if not filters:
        return rows
    return [
        row
        for row in rows
        if all(values_match(row.get(field, ""), allowed_values) for field, allowed_values in filters.items())
    ]


def discover_result_paths(results_dirs: list[str], results: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in results:
        path = Path(item)
        if path.is_dir():
            paths.extend(path.rglob("results.csv"))
        elif path.name == "results.csv":
            paths.append(path)
    for item in results_dirs:
        paths.extend(Path(item).rglob("results.csv"))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    return [Path(path) for path in unique]


def t_critical_975(df: int) -> float:
    if df <= 0:
        return math.nan
    if df <= 30:
        return T_CRIT_975[df]
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def infer_condition_fields(rows: list[dict[str, str]], requested: list[str], include_step: bool) -> list[str]:
    available = {key for row in rows for key, value in row.items() if value not in ("", None)}
    fields = [field for field in requested if field in available]
    if include_step and "step" in available and "step" not in fields:
        fields.append("step")
    return fields


def key_from_row(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in fields)


def format_condition(fields: list[str], key: tuple[str, ...]) -> str:
    if not fields:
        return "all"
    return ", ".join(f"{field}={value}" for field, value in zip(fields, key))


def direction_for_metric(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher"
    return "lower" if metric in LOWER_IS_BETTER else "lower"


def paired_rows_for_source(
    source_path: Path,
    rows: list[dict[str, str]],
    condition_fields: list[str],
    target_method: str,
    baselines: list[str],
    metrics: list[str],
    min_pairs: int,
    include_marginal: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_method_key_seed: dict[tuple[str, tuple[str, ...], str], dict[str, str]] = {}
    source_name = str(source_path)
    source_dir = str(source_path.parent)
    for row in rows:
        method = row.get("method", "")
        if not include_marginal and method in MARGINAL_METHODS:
            continue
        seed = row.get("seed", "")
        if seed == "":
            continue
        key = key_from_row(row, condition_fields)
        rows_by_method_key_seed[(method, key, seed)] = row

    condition_keys = sorted({key for _, key, _ in rows_by_method_key_seed})
    by_condition: list[dict[str, Any]] = []
    all_diffs: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_target_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    all_baseline_values: dict[tuple[str, str], list[float]] = defaultdict(list)

    for key in condition_keys:
        seeds = sorted({seed for method, cond, seed in rows_by_method_key_seed if cond == key and method == target_method})
        for baseline in baselines:
            if not include_marginal and baseline in MARGINAL_METHODS:
                continue
            for metric in metrics:
                paired: list[tuple[float, float]] = []
                paired_seeds: list[str] = []
                for seed in seeds:
                    target_row = rows_by_method_key_seed.get((target_method, key, seed))
                    baseline_row = rows_by_method_key_seed.get((baseline, key, seed))
                    if target_row is None or baseline_row is None:
                        continue
                    target_value = to_float(target_row.get(metric, ""))
                    baseline_value = to_float(baseline_row.get(metric, ""))
                    if math.isnan(target_value) or math.isnan(baseline_value):
                        continue
                    paired.append((target_value, baseline_value))
                    paired_seeds.append(seed)
                if len(paired) < min_pairs:
                    continue
                row = paired_summary(
                    source_name,
                    source_dir,
                    condition_fields,
                    key,
                    target_method,
                    baseline,
                    metric,
                    paired,
                    paired_seeds,
                )
                by_condition.append(row)
                all_diffs[(baseline, metric)].extend([target - base for target, base in paired])
                all_target_values[(baseline, metric)].extend([target for target, _ in paired])
                all_baseline_values[(baseline, metric)].extend([base for _, base in paired])

    overall: list[dict[str, Any]] = []
    for (baseline, metric), diffs in sorted(all_diffs.items()):
        if len(diffs) < min_pairs:
            continue
        paired = list(zip(all_target_values[(baseline, metric)], all_baseline_values[(baseline, metric)]))
        row = paired_summary(
            source_name,
            source_dir,
            [],
            (),
            target_method,
            baseline,
            metric,
            paired,
            [str(idx) for idx in range(len(paired))],
        )
        row["condition"] = "overall_across_conditions"
        overall.append(row)

    return by_condition, overall


def paired_summary(
    source_name: str,
    source_dir: str,
    condition_fields: list[str],
    condition_key: tuple[str, ...],
    target_method: str,
    baseline: str,
    metric: str,
    paired: list[tuple[float, float]],
    paired_seeds: list[str],
) -> dict[str, Any]:
    target_values = np.asarray([target for target, _ in paired], dtype=np.float64)
    baseline_values = np.asarray([base for _, base in paired], dtype=np.float64)
    diffs = target_values - baseline_values
    n = int(len(diffs))
    diff_mean = float(np.mean(diffs))
    diff_std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    diff_se = diff_std / math.sqrt(n) if n > 1 else 0.0
    crit = t_critical_975(n - 1)
    ci_low = diff_mean - crit * diff_se if n > 1 else diff_mean
    ci_high = diff_mean + crit * diff_se if n > 1 else diff_mean
    direction = direction_for_metric(metric)
    if direction == "higher":
        wins = target_values > baseline_values
        losses = target_values < baseline_values
    else:
        wins = target_values < baseline_values
        losses = target_values > baseline_values
    baseline_mean = float(np.mean(baseline_values))
    relative_change = 100.0 * diff_mean / abs(baseline_mean) if abs(baseline_mean) > 1e-12 else math.nan
    return {
        "source": source_name,
        "source_dir": source_dir,
        "condition": format_condition(condition_fields, condition_key),
        "condition_fields": ",".join(condition_fields),
        **{field: value for field, value in zip(condition_fields, condition_key)},
        "target_method": target_method,
        "baseline_method": baseline,
        "metric": metric,
        "direction": direction,
        "n_pairs": n,
        "paired_seeds": " ".join(paired_seeds),
        "target_mean": float(np.mean(target_values)),
        "baseline_mean": baseline_mean,
        "diff_mean": diff_mean,
        "diff_se": diff_se,
        "diff_ci95_low": float(ci_low),
        "diff_ci95_high": float(ci_high),
        "relative_change_percent": relative_change,
        "target_win_rate": float(np.mean(wins)),
        "target_loss_rate": float(np.mean(losses)),
        "target_tie_rate": float(np.mean(target_values == baseline_values)),
        "target_better_all_pairs": int(bool(np.all(wins))),
        "target_better_ci95": int((ci_low > 0.0) if direction == "higher" else (ci_high < 0.0)),
        "baseline_better_ci95": int((ci_high < 0.0) if direction == "higher" else (ci_low > 0.0)),
    }


def field_order(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "source",
        "source_dir",
        "condition",
        "condition_fields",
        "dataset",
        "noise_rate",
        "pretrain_fraction",
        "pretrain_training_mode",
        "es_budget",
        "reference_method",
        "reference_es_budget",
        "step",
        "target_method",
        "baseline_method",
        "metric",
        "direction",
        "n_pairs",
        "paired_seeds",
        "target_mean",
        "baseline_mean",
        "diff_mean",
        "diff_se",
        "diff_ci95_low",
        "diff_ci95_high",
        "relative_change_percent",
        "target_win_rate",
        "target_loss_rate",
        "target_tie_rate",
        "target_better_all_pairs",
        "target_better_ci95",
        "baseline_better_ci95",
    ]
    fields: list[str] = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def write_markdown_report(path: Path, overall_rows: list[dict[str, Any]], target_method: str) -> None:
    priority_metrics = [
        "test_inappropriate_risk",
        "certified_bound",
        "effective_compression_size",
        "noise_hit_rate",
        "group_revisit_rate",
        "unique_group_fraction",
    ]
    lines = [
        "# Paired Statistical Evidence",
        "",
        f"Target method: `{target_method}`.",
        "",
        "Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. "
        "For lower-is-better metrics, negative values favor the target method.",
        "",
    ]
    def short_source_dir(source_dir: str) -> str:
        path_obj = Path(source_dir)
        parts = path_obj.parts
        if "results" in parts:
            idx = parts.index("results")
            return str(Path(*parts[idx:]))
        return str(path_obj)

    for metric in priority_metrics:
        rows = [row for row in overall_rows if row["metric"] == metric]
        if not rows:
            continue
        lines.extend([f"## {metric}", ""])
        for row in sorted(rows, key=lambda item: (item["source_dir"], item["baseline_method"])):
            ci = f"[{row['diff_ci95_low']:.4g}, {row['diff_ci95_high']:.4g}]"
            if row["target_better_ci95"]:
                verdict = "favors target"
            elif row["baseline_better_ci95"]:
                verdict = "favors baseline"
            else:
                verdict = "inconclusive"
            lines.append(
                f"- `{short_source_dir(row['source_dir'])}` vs `{row['baseline_method']}`: "
                f"mean diff {row['diff_mean']:.4g}, 95% CI {ci}, "
                f"win rate {row['target_win_rate']:.2f}, n={row['n_pairs']} ({verdict})."
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    result_paths = discover_result_paths(args.results_dir, args.results)
    if not result_paths:
        raise FileNotFoundError("No results.csv files found. Pass --results-dir or --results.")
    row_filters = parse_row_filters(args.row_filter)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_condition_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    for path in result_paths:
        rows = filter_rows(read_csv(path), row_filters)
        if not rows:
            continue
        condition_fields = infer_condition_fields(rows, args.condition_fields, args.include_step)
        condition_rows, source_overall_rows = paired_rows_for_source(
            path,
            rows,
            condition_fields,
            args.target_method,
            args.baselines,
            args.metrics,
            args.min_pairs,
            args.include_marginal,
        )
        by_condition_rows.extend(condition_rows)
        overall_rows.extend(source_overall_rows)

    if not by_condition_rows and not overall_rows:
        raise ValueError("No paired comparisons were available for the requested target, baselines, and metrics.")

    if by_condition_rows:
        write_csv(output_dir / "paired_by_condition.csv", field_order(by_condition_rows), by_condition_rows)
    if overall_rows:
        write_csv(output_dir / "paired_overall.csv", field_order(overall_rows), overall_rows)
        write_markdown_report(output_dir / "paired_report.md", overall_rows, args.target_method)


if __name__ == "__main__":
    main()
