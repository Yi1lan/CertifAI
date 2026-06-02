from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RESULT_FIELDS = [
    "method",
    "dataset",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "n_cert",
    "n_pretrain",
    "compression_size",
    "remaining_bad",
    "effective_compression_size",
    "certified_bound",
    "test_error",
    "test_inappropriate_risk",
    "pac_bayes_bound",
    "pac_bayes_empirical_risk",
    "pac_bayes_mc_upper",
    "pac_bayes_kl",
    "runtime_sec",
    "stop_reached",
    "train_calls",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "pairwise_feature_cosine",
    "mean_support_redundancy",
    "max_support_redundancy",
    "mean_selected_residual_novelty",
    "local_redundancy_hit_rate",
    "residual_redundancy_hit_rate",
    "strong_redundancy_hit_rate",
    "mode_entropy",
    "minority_mode_fraction",
    "spectral_entropy",
    "dynamic_mu",
]

SUMMARY_NUMERIC_FIELDS = [
    "compression_size",
    "remaining_bad",
    "effective_compression_size",
    "certified_bound",
    "test_error",
    "test_inappropriate_risk",
    "pac_bayes_bound",
    "pac_bayes_empirical_risk",
    "pac_bayes_mc_upper",
    "pac_bayes_kl",
    "runtime_sec",
    "stop_reached",
    "hit_limit",
    "train_calls",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "pairwise_feature_cosine",
    "mean_support_redundancy",
    "max_support_redundancy",
    "mean_selected_residual_novelty",
    "local_redundancy_hit_rate",
    "residual_redundancy_hit_rate",
    "strong_redundancy_hit_rate",
    "mode_entropy",
    "minority_mode_fraction",
    "spectral_entropy",
    "dynamic_mu",
]

MARGINAL_METHODS = {
    "Marginal",
    "Marginal+Residual",
    "Marginal-Redundancy",
    "Marginal+Residual-Redundancy",
}


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: normalize_value(row.get(field)) for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def to_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def summarize(rows: list[dict[str, Any]], group_fields: list[str], numeric_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        out = {field: key[idx] for idx, field in enumerate(group_fields)}
        out["n"] = len(group)
        for field in numeric_fields:
            values = [to_float(row, field) for row in group]
            values = [value for value in values if not math.isnan(value)]
            if values:
                arr = np.asarray(values, dtype=np.float64)
                out[f"{field}_mean"] = float(np.mean(arr))
                out[f"{field}_se"] = (
                    float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
                )
            else:
                out[f"{field}_mean"] = ""
                out[f"{field}_se"] = ""
        output.append(out)
    return output


def summary_fields(summary: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in summary:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def without_marginal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("method") not in MARGINAL_METHODS]


def write_summary_views(
    output_dir: Path,
    rows: list[dict[str, Any]],
    group_fields: list[str],
    numeric_fields: list[str],
) -> None:
    summary = summarize(rows, group_fields=group_fields, numeric_fields=numeric_fields)
    fields = summary_fields(summary)
    write_csv(output_dir / "summary.csv", fields, summary)
    write_csv(output_dir / "summary_with_marginal.csv", fields, summary)

    filtered = without_marginal_rows(rows)
    summary_no_marginal = summarize(filtered, group_fields=group_fields, numeric_fields=numeric_fields)
    fields_no_marginal = summary_fields(summary_no_marginal)
    write_csv(output_dir / "summary_without_marginal.csv", fields_no_marginal, summary_no_marginal)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "with_marginal.csv", fields, summary)
    write_csv(tables_dir / "without_marginal.csv", fields_no_marginal, summary_no_marginal)
