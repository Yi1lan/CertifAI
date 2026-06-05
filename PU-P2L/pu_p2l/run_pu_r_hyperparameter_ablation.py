from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import make_pretrain_split
from .io_utils import SUMMARY_NUMERIC_FIELDS, write_csv, write_json, write_summary_views
from .model import resolve_device
from .run_boundary import add_dataset_args, add_pac_bayes_args, add_score_args, build_config, make_dataset_from_args
from .run_es_budget_boundary import ES_BUDGET_FIELDS
from .runner import METHODS, RunConfig, run_p2l_es_budgets
from .scores import ScoreConfig

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ABLATION_FIELDS = [
    *ES_BUDGET_FIELDS,
    "variant",
    "sweep_name",
    "sweep_value",
    "mu",
    "global_redundancy_weight",
    "residual_rank",
    "gamma_value",
]

PLOT_COLORS = ["#274753", "#297270", "#299d8f", "#8ab07c", "#e7c66b", "#f3a361", "#e66d50"]


def format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-factor PU-R hyperparameter ablations.")
    parser.add_argument("--output-dir", type=str, default="results/PU-R/ablations/pu_r_hyperparameters")
    parser.add_argument("--dataset-name", type=str, default="fashion_mnist")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.3])
    parser.add_argument("--pretrain-fractions", type=float, nargs="+", default=[0.3])
    parser.add_argument("--es-budgets", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--baseline-methods", type=str, nargs="*", default=[])

    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--duplicate-groups", type=int, default=40)
    parser.add_argument("--duplicates-per-group", type=int, default=10)
    parser.add_argument("--ambiguous-fraction", type=float, default=0.35)
    parser.add_argument("--cluster-std", type=float, default=0.45)
    parser.add_argument("--band-std", type=float, default=0.35)
    parser.add_argument("--duplicate-std", type=float, default=0.015)
    add_dataset_args(parser)

    parser.add_argument("--model-name", type=str, default="auto", choices=["auto", "small_mlp", "mnist_fcn", "cifar_resnet18"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout-prob", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--p2l-epochs-per-iter", type=int, default=1)
    parser.add_argument("--p2l-lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=-math.log(0.5))
    parser.add_argument("--delta", type=float, default=0.035)
    parser.add_argument("--max-total-support", type=int, default=1000)
    parser.add_argument("--initial-per-class", type=int, default=2)
    parser.add_argument("--greats-probe-size", type=int, default=64)
    add_pac_bayes_args(parser, default_samples=0)
    add_score_args(parser)

    parser.add_argument("--mu-values", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0, 1.5])
    parser.add_argument(
        "--redundancy-weight-values",
        type=float,
        nargs="*",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--residual-rank-values", type=int, nargs="*", default=[0, 32, 64, 128])
    parser.add_argument("--gamma-values", type=float, nargs="*", default=[])
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def make_variant_config(base: RunConfig, sweep_name: str, sweep_value: float) -> RunConfig:
    score: ScoreConfig = base.score
    if sweep_name == "mu":
        score = replace(score, mu=float(sweep_value))
        return replace(base, score=score)
    if sweep_name == "global_redundancy_weight":
        score = replace(score, global_redundancy_weight=float(sweep_value))
        return replace(base, score=score)
    if sweep_name == "residual_rank":
        score = replace(score, residual_rank=int(sweep_value))
        return replace(base, score=score)
    if sweep_name == "gamma":
        score = replace(score, gamma=float(sweep_value))
        return replace(base, gamma=float(sweep_value), score=score)
    raise ValueError(f"Unknown sweep name: {sweep_name}")


def variants_from_args(args: argparse.Namespace) -> list[tuple[str, float, str]]:
    variants: list[tuple[str, float, str]] = []
    for value in args.mu_values:
        variants.append(("mu", float(value), f"mu={value:g}"))
    for value in args.redundancy_weight_values:
        variants.append(("global_redundancy_weight", float(value), f"lambda={value:g}"))
    for value in args.residual_rank_values:
        variants.append(("residual_rank", float(value), f"rank={value:g}"))
    for value in args.gamma_values:
        variants.append(("gamma", float(value), f"gamma={value:g}"))
    if not variants:
        raise ValueError("At least one hyperparameter value must be supplied.")
    return variants


def mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), se


def metric_groups(
    rows: list[dict[str, Any]],
    sweep_name: str,
    noise_rate: float,
    pretrain_fraction: float,
    es_budget: int,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    values: dict[float, list[float]] = {}
    for row in rows:
        if row.get("sweep_name") != sweep_name:
            continue
        if abs(float(row.get("noise_rate", 0.0)) - noise_rate) > 1e-12:
            continue
        if abs(float(row.get("pretrain_fraction", 0.0)) - pretrain_fraction) > 1e-12:
            continue
        if int(row.get("es_budget", -1)) != int(es_budget):
            continue
        value = row.get(metric)
        if value in ("", None):
            continue
        y = float(value)
        if np.isnan(y):
            continue
        x = float(row["sweep_value"])
        values.setdefault(x, []).append(y)
    xs = sorted(values)
    means: list[float] = []
    ses: list[float] = []
    for x in xs:
        mean, se = mean_se(values[x])
        means.append(mean)
        ses.append(se)
    return xs, means, ses


def plot_line_with_band(
    ax: plt.Axes,
    xs: list[float],
    means: list[float],
    ses: list[float],
    label: str,
    color: str,
) -> None:
    x_arr = np.asarray(xs, dtype=np.float64)
    mean_arr = np.asarray(means, dtype=np.float64)
    se_arr = np.asarray(ses, dtype=np.float64)
    ax.plot(x_arr, mean_arr, marker="o", linewidth=2.0, color=color, label=label)
    ax.fill_between(x_arr, mean_arr - se_arr, mean_arr + se_arr, color=color, alpha=0.18, linewidth=0)


def plot_hyperparameter_ablation(rows: list[dict[str, Any]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sweep_names = sorted({str(row["sweep_name"]) for row in rows if row.get("sweep_name") != "baseline"})
    noise_rates = sorted({float(row["noise_rate"]) for row in rows})
    pretrain_fractions = sorted({float(row["pretrain_fraction"]) for row in rows})
    budgets = sorted({int(row["es_budget"]) for row in rows})
    metric_sets = [
        (
            "bound_risk",
            [
                ("test_inappropriate_risk", "Clean test risk"),
                ("certified_bound", "P2L-ES bound"),
                ("effective_compression_size", "Effective compression size"),
            ],
        ),
        (
            "selection_diagnostics",
            [
                ("noise_hit_rate", "Noise-hit rate"),
                ("duplicate_hit_rate", "Duplicate-hit rate"),
                ("pairwise_feature_cosine", "Pairwise feature cosine"),
                ("mean_selected_residual_novelty", "Mean residual novelty"),
            ],
        ),
    ]

    for sweep_name in sweep_names:
        for noise_rate in noise_rates:
            for pretrain_fraction in pretrain_fractions:
                for plot_name, metrics in metric_sets:
                    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 2.8 * len(metrics)), sharex=True)
                    if len(metrics) == 1:
                        axes = [axes]
                    any_curve = False
                    for metric_index, (metric, ylabel) in enumerate(metrics):
                        ax = axes[metric_index]
                        for budget_index, budget in enumerate(budgets):
                            xs, means, ses = metric_groups(
                                rows, sweep_name, noise_rate, pretrain_fraction, budget, metric
                            )
                            if not xs:
                                continue
                            any_curve = True
                            color = PLOT_COLORS[budget_index % len(PLOT_COLORS)]
                            plot_line_with_band(ax, xs, means, ses, f"ES={budget}", color)
                        ax.set_ylabel(ylabel)
                        ax.grid(alpha=0.25)
                        ax.legend(fontsize=9)
                    axes[-1].set_xlabel(sweep_name.replace("_", " "))
                    fig.suptitle(
                        f"PU-R hyperparameter ablation: {sweep_name}, noise={noise_rate:g}, "
                        f"pretrain={pretrain_fraction:g}",
                        fontsize=12,
                    )
                    fig.tight_layout()
                    if any_curve:
                        filename = (
                            f"{sweep_name}_{plot_name}_noise_{format_float_for_filename(noise_rate)}_"
                            f"pretrain_{format_float_for_filename(pretrain_fraction)}.png"
                        )
                        fig.savefig(plots_dir / filename, dpi=180)
                    plt.close(fig)


def main() -> None:
    args = parse_args()
    unknown_baselines = sorted(set(args.baseline_methods) - set(METHODS))
    if unknown_baselines:
        raise ValueError(f"Unknown baseline methods: {unknown_baselines}. Valid methods: {METHODS}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    base_config = build_config(args)
    variants = variants_from_args(args)
    write_json(output_dir / "config.json", vars(args))

    tasks = [
        (seed, noise_rate, pretrain_fraction, sweep_name, sweep_value, variant)
        for seed in args.seeds
        for noise_rate in args.noise_rates
        for pretrain_fraction in args.pretrain_fractions
        for sweep_name, sweep_value, variant in variants
    ]
    progress = tqdm(tasks, desc="PU-R hyperparameter ablation") if tqdm is not None else tasks
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, float, float], Any] = {}

    for seed, noise_rate, pretrain_fraction, sweep_name, sweep_value, variant in progress:
        key = (seed, noise_rate, pretrain_fraction)
        if key not in cache:
            bundle = make_dataset_from_args(args, seed, noise_rate)
            cache[key] = make_pretrain_split(bundle, pretrain_fraction, seed)
        config = make_variant_config(base_config, sweep_name, sweep_value)
        result_rows = run_p2l_es_budgets(
            "PU-R",
            seed,
            noise_rate,
            pretrain_fraction,
            cache[key],
            config,
            device,
            args.es_budgets,
        )
        for row in result_rows:
            row["dataset"] = args.dataset_name
            row["variant"] = variant
            row["sweep_name"] = sweep_name
            row["sweep_value"] = sweep_value
            row["mu"] = config.score.mu
            row["global_redundancy_weight"] = config.score.global_redundancy_weight
            row["residual_rank"] = config.score.residual_rank
            row["gamma_value"] = config.gamma
        rows.extend(result_rows)

    for method in args.baseline_methods:
        baseline_tasks = [
            (seed, noise_rate, pretrain_fraction)
            for seed in args.seeds
            for noise_rate in args.noise_rates
            for pretrain_fraction in args.pretrain_fractions
        ]
        baseline_progress = tqdm(baseline_tasks, desc=f"{method} baseline") if tqdm is not None else baseline_tasks
        for seed, noise_rate, pretrain_fraction in baseline_progress:
            key = (seed, noise_rate, pretrain_fraction)
            if key not in cache:
                bundle = make_dataset_from_args(args, seed, noise_rate)
                cache[key] = make_pretrain_split(bundle, pretrain_fraction, seed)
            result_rows = run_p2l_es_budgets(
                method,
                seed,
                noise_rate,
                pretrain_fraction,
                cache[key],
                base_config,
                device,
                args.es_budgets,
            )
            for row in result_rows:
                row["dataset"] = args.dataset_name
                row["variant"] = method
                row["sweep_name"] = "baseline"
                row["sweep_value"] = ""
                row["mu"] = base_config.score.mu
                row["global_redundancy_weight"] = base_config.score.global_redundancy_weight
                row["residual_rank"] = base_config.score.residual_rank
                row["gamma_value"] = base_config.gamma
            rows.extend(result_rows)

    write_csv(output_dir / "results.csv", ABLATION_FIELDS, rows)
    write_summary_views(
        output_dir,
        rows,
        group_fields=[
            "dataset",
            "method",
            "variant",
            "sweep_name",
            "sweep_value",
            "noise_rate",
            "pretrain_fraction",
            "pretrain_training_mode",
            "es_budget",
        ],
        numeric_fields=["step", *SUMMARY_NUMERIC_FIELDS, "mu", "global_redundancy_weight", "residual_rank", "gamma_value"],
    )
    if not args.no_plots:
        plot_hyperparameter_ablation(rows, output_dir / "plots")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
