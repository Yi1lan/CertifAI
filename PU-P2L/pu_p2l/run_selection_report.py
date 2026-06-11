from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import write_csv
from .run_selection_visualization import plot_selection_group


GROUP_FIELDS = [
    "dataset",
    "method",
    "noise_rate",
    "pretrain_fraction",
    "pretrain_training_mode",
    "budget",
]

METRICS = [
    "certified_bound",
    "effective_compression_size",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "pairwise_feature_cosine",
    "group_revisit_rate",
    "unique_group_fraction",
    "runtime_sec",
]

TABLE_METRICS = [
    ("noise_hit_rate", "Noise-hit"),
    ("duplicate_hit_rate", "Duplicate-hit"),
    ("pairwise_feature_cosine", "Pairwise cosine"),
    ("group_revisit_rate", "Group revisit"),
    ("effective_compression_size", "$k_{\\mathrm{eff}}$"),
    ("certified_bound", "P2L-ES bound"),
]

NEW_SELECTION_METRICS = [
    ("noise_hit_rate", "Noise-hit"),
    ("duplicate_hit_rate", "Duplicate-hit"),
    ("group_revisit_rate", "Group revisit"),
    ("unique_group_fraction", "Unique group fraction"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mean/std selection-ablation report and representative "
            "selection plots from a run_selection_visualization output folder."
        )
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--methods", type=str, nargs="*", default=[])
    parser.add_argument("--target-methods", type=str, nargs="+", default=["PU-R", "PU-R-Manifold"])
    parser.add_argument("--example-budget", type=int, default=100)
    parser.add_argument("--best-metric", type=str, default="noise_hit_rate")
    parser.add_argument("--best-direction", type=str, choices=["min", "max"], default="min")
    parser.add_argument("--background-limit", type=int, default=6000)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def finite(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def summarize_results(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in GROUP_FIELDS)].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out: dict[str, Any] = {field: key[idx] for idx, field in enumerate(GROUP_FIELDS)}
        out["n"] = len(group)
        for metric in METRICS:
            values = finite([to_float(row, metric) for row in group])
            if values:
                arr = np.asarray(values, dtype=np.float64)
                out[f"{metric}_mean"] = float(np.mean(arr))
                out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        summary.append(out)
    return summary


def group_revisit_rate(group_ids: list[int]) -> float:
    valid = np.asarray([group_id for group_id in group_ids if group_id >= 0], dtype=np.int64)
    if len(valid) == 0:
        return 0.0
    _, counts = np.unique(valid, return_counts=True)
    return float(np.sum(np.maximum(counts - 1, 0)) / len(valid))


def unique_group_fraction(group_ids: list[int]) -> float:
    valid = np.asarray([group_id for group_id in group_ids if group_id >= 0], dtype=np.int64)
    if len(valid) == 0:
        return 0.0
    return float(len(np.unique(valid)) / len(valid))


def new_selection_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    seed_group_fields = [*GROUP_FIELDS, "seed"]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(float(row.get("initial_support", "0") or 0)) == 1:
            continue
        key = tuple(str(row.get(field, "")) for field in seed_group_fields)
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        group_ids = [int(float(row.get("group_id", "-1") or -1)) for row in group]
        output.append(
            {
                **{field: key[idx] for idx, field in enumerate(seed_group_fields)},
                "n_selected": len(group),
                "noise_hit_rate": float(np.mean([to_float(row, "is_noisy") for row in group])),
                "duplicate_hit_rate": float(np.mean([to_float(row, "is_duplicate") for row in group])),
                "group_revisit_rate": group_revisit_rate(group_ids),
                "unique_group_fraction": unique_group_fraction(group_ids),
            }
        )
    return output


def summarize_new_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field, "")) for field in GROUP_FIELDS)].append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out: dict[str, Any] = {field: key[idx] for idx, field in enumerate(GROUP_FIELDS)}
        out["n"] = len(group)
        for metric, _ in NEW_SELECTION_METRICS:
            values = finite([to_float(row, metric) for row in group])
            if values:
                arr = np.asarray(values, dtype=np.float64)
                out[f"{metric}_mean"] = float(np.mean(arr))
                out[f"{metric}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = ""
                out[f"{metric}_std"] = ""
        summary.append(out)
    return summary


def summary_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def method_order(rows: list[dict[str, str]], explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    preferred = ["MaxLoss", "GREATS", "PU-R", "PU-R-Manifold"]
    observed = []
    for row in rows:
        method = row.get("method", "")
        if method and method not in observed:
            observed.append(method)
    return [method for method in preferred if method in observed] + [
        method for method in observed if method not in preferred
    ]


def choose_best_examples(
    rows: list[dict[str, str]],
    targets: list[str],
    budget: int,
    metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    examples = []
    multiplier = 1.0 if direction == "min" else -1.0
    for target in targets:
        candidates = [
            row
            for row in rows
            if row.get("method") == target
            and int(float(row.get("budget", "0") or 0)) == int(budget)
            and not math.isnan(to_float(row, metric))
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                multiplier * to_float(row, metric),
                to_float(row, "certified_bound"),
                to_float(row, "effective_compression_size"),
                int(float(row.get("seed", "0") or 0)),
            )
        )
        best = dict(candidates[0])
        best["target_method"] = target
        best["best_metric"] = metric
        best["best_direction"] = direction
        examples.append(best)
    return examples


def row_matches_group(row: dict[str, str], example: dict[str, Any]) -> bool:
    for field in ["dataset", "seed", "noise_rate", "pretrain_fraction"]:
        if str(row.get(field, "")) != str(example.get(field, "")):
            return False
    return True


def safe_name(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("+", "plus").replace("-", "_").lower()


def plot_best_examples(
    input_dir: Path,
    output_dir: Path,
    rows: list[dict[str, str]],
    examples: list[dict[str, Any]],
    methods: list[str],
    background_limit: int,
) -> None:
    if not examples:
        return
    pool_rows = read_rows(input_dir / "pool_projection.csv")
    selected_rows = read_rows(input_dir / "selected_points.csv")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for example in examples:
        budget = int(float(example["budget"]))
        target = str(example["target_method"])
        pool_group = [row for row in pool_rows if row_matches_group(row, example)]
        selected_group = [row for row in selected_rows if row_matches_group(row, example)]
        summary_group = [row for row in rows if row_matches_group(row, example)]
        title = (
            f"Representative {target}: seed={example['seed']}, "
            f"noise={float(example['noise_rate']):g}, budget={budget}"
        )
        filename = (
            f"selection_best_{safe_name(target)}_seed_{int(float(example['seed']))}_"
            f"budget_{budget}.png"
        )
        plot_selection_group(
            pool_group,
            selected_group,
            summary_group,
            methods,
            budget,
            background_limit,
            plots_dir / filename,
            title,
        )


def fmt_mean_std(row: dict[str, Any], metric: str) -> str:
    mean = row.get(f"{metric}_mean", "")
    std = row.get(f"{metric}_std", "")
    if mean == "" or std == "":
        return "--"
    return f"{float(mean):.3f} $\\pm$ {float(std):.3f}"


def write_latex_table(path: Path, summary: list[dict[str, Any]], methods: list[str]) -> None:
    ordered = sorted(
        summary,
        key=lambda row: (float(row["budget"]), methods.index(row["method"]) if row["method"] in methods else 999),
    )
    with path.open("w") as handle:
        handle.write("\\begin{tabular}{ll" + "c" * len(TABLE_METRICS) + "}\\n")
        handle.write("\\toprule\\n")
        headers = ["Budget", "Method", *[label for _, label in TABLE_METRICS]]
        handle.write(" & ".join(headers) + r" \\" + "\n")
        handle.write("\\midrule\\n")
        for row in ordered:
            cells = [str(int(float(row["budget"]))), str(row["method"])]
            cells.extend(fmt_mean_std(row, metric) for metric, _ in TABLE_METRICS)
            handle.write(" & ".join(cells) + r" \\" + "\n")
        handle.write("\\bottomrule\\n")
        handle.write("\\end{tabular}\\n")


def write_new_selection_latex_table(path: Path, summary: list[dict[str, Any]], methods: list[str]) -> None:
    ordered = sorted(
        summary,
        key=lambda row: (float(row["budget"]), methods.index(row["method"]) if row["method"] in methods else 999),
    )
    with path.open("w") as handle:
        handle.write("\\begin{tabular}{ll" + "c" * len(NEW_SELECTION_METRICS) + "}\\n")
        handle.write("\\toprule\\n")
        headers = ["Budget", "Method", *[label for _, label in NEW_SELECTION_METRICS]]
        handle.write(" & ".join(headers) + r" \\" + "\n")
        handle.write("\\midrule\\n")
        for row in ordered:
            cells = [str(int(float(row["budget"]))), str(row["method"])]
            cells.extend(fmt_mean_std(row, metric) for metric, _ in NEW_SELECTION_METRICS)
            handle.write(" & ".join(cells) + r" \\" + "\n")
        handle.write("\\bottomrule\\n")
        handle.write("\\end{tabular}\\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "selection_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = input_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    rows = read_rows(results_path)
    methods = method_order(rows, args.methods)
    summary = summarize_results(rows)
    write_csv(output_dir / "selection_mean_sd.csv", summary_fields(summary), summary)
    write_latex_table(output_dir / "selection_mean_sd_table.tex", summary, methods)

    selected_path = input_dir / "selected_points.csv"
    if selected_path.exists():
        selected_rows = read_rows(selected_path)
        new_rows = new_selection_seed_rows(selected_rows)
        new_summary = summarize_new_selection(new_rows)
        write_csv(output_dir / "new_selection_seed_metrics.csv", summary_fields(new_rows), new_rows)
        write_csv(output_dir / "new_selection_mean_sd.csv", summary_fields(new_summary), new_summary)
        write_new_selection_latex_table(output_dir / "new_selection_mean_sd_table.tex", new_summary, methods)

    examples = choose_best_examples(
        rows,
        args.target_methods,
        args.example_budget,
        args.best_metric,
        args.best_direction,
    )
    if examples:
        write_csv(output_dir / "selection_best_examples.csv", list(examples[0].keys()), examples)
        plot_best_examples(input_dir, output_dir, rows, examples, methods, args.background_limit)


if __name__ == "__main__":
    main()
