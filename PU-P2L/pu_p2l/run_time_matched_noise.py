from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .data import make_pretrain_split
from .io_utils import SUMMARY_NUMERIC_FIELDS, write_csv, write_json, write_summary_views
from .model import resolve_device
from .plotting import plot_time_matched_noise
from .runner import METHODS, run_p2l_time_budget
from .run_boundary import add_dataset_args, add_pac_bayes_args, add_score_args, build_config, make_dataset_from_args

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


TIME_MATCHED_METHODS = ["MaxLoss", "Marginal", "EL2N", "GraNdLast", "RHO-PretrainRef", "PU-R", "GREATS"]
TIME_MATCHED_FIELDS = [
    "method",
    "dataset",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "pretrain_training_mode",
    "reference_method",
    "reference_es_budget",
    "es_budget",
    "target_step",
    "time_budget_sec",
    "selection_runtime_sec",
    "time_budget_overrun_sec",
    "time_budget_ratio",
    "stopping_condition",
    "time_budget_hit",
    "target_step_hit",
    "step",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a time-matched PU-P2L noise sweep. The reference method is first run "
            "to a fixed ES budget; each other method then receives the same selection-loop wall-clock time."
        )
    )
    parser.add_argument("--output-dir", type=str, default="results/PU-R/FashionMNIST/time_matched/pur_es50")
    parser.add_argument("--dataset-name", type=str, default="fashion_mnist")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--pretrain-fractions", type=float, nargs="+", default=[0.3])
    parser.add_argument("--reference-method", type=str, default="PU-R")
    parser.add_argument("--reference-es-budget", type=int, default=50)
    parser.add_argument("--methods", type=str, nargs="+", default=TIME_MATCHED_METHODS)

    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--duplicate-groups", type=int, default=40)
    parser.add_argument("--duplicates-per-group", type=int, default=10)
    parser.add_argument("--ambiguous-fraction", type=float, default=0.35)
    parser.add_argument("--cluster-std", type=float, default=0.45)
    parser.add_argument("--band-std", type=float, default=0.35)
    parser.add_argument("--duplicate-std", type=float, default=0.015)
    add_dataset_args(parser)

    parser.add_argument(
        "--model-name",
        type=str,
        default="auto",
        choices=["auto", "small_mlp", "mnist_fcn", "cifar_resnet18"],
    )
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
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {METHODS}")
    if args.reference_method not in METHODS:
        raise ValueError(f"Unknown reference method: {args.reference_method}. Valid methods: {METHODS}")
    if args.reference_es_budget < 0:
        raise ValueError("--reference-es-budget must be non-negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = build_config(args)
    write_json(output_dir / "config.json", vars(args))

    tasks = [
        (seed, noise_rate, pretrain_fraction)
        for seed in args.seeds
        for noise_rate in args.noise_rates
        for pretrain_fraction in args.pretrain_fractions
    ]
    comparison_methods = [method for method in args.methods if method != args.reference_method]
    total_runs = len(tasks) * (1 + len(comparison_methods))
    progress = tqdm(total=total_runs, desc="PU-P2L time-matched noise") if tqdm is not None else None
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, float, float], Any] = {}

    try:
        for seed, noise_rate, pretrain_fraction in tasks:
            key = (seed, noise_rate, pretrain_fraction)
            if key not in cache:
                bundle = make_dataset_from_args(args, seed, noise_rate)
                cache[key] = make_pretrain_split(bundle, pretrain_fraction, seed)
            split = cache[key]

            reference_row = run_p2l_time_budget(
                args.reference_method,
                seed,
                noise_rate,
                pretrain_fraction,
                split,
                config,
                device,
                reference_method=args.reference_method,
                reference_es_budget=args.reference_es_budget,
                target_step=args.reference_es_budget,
            )
            reference_row["dataset"] = args.dataset_name
            rows.append(reference_row)
            if progress is not None:
                progress.update(1)

            time_budget_sec = float(reference_row["selection_runtime_sec"])
            for method in comparison_methods:
                row = run_p2l_time_budget(
                    method,
                    seed,
                    noise_rate,
                    pretrain_fraction,
                    split,
                    config,
                    device,
                    reference_method=args.reference_method,
                    reference_es_budget=args.reference_es_budget,
                    time_budget_sec=time_budget_sec,
                )
                row["dataset"] = args.dataset_name
                rows.append(row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    write_csv(output_dir / "results.csv", TIME_MATCHED_FIELDS, rows)
    write_summary_views(
        output_dir,
        rows,
        group_fields=[
            "dataset",
            "reference_method",
            "reference_es_budget",
            "method",
            "noise_rate",
            "pretrain_fraction",
            "pretrain_training_mode",
        ],
        numeric_fields=[
            "step",
            "target_step",
            "time_budget_sec",
            "selection_runtime_sec",
            "time_budget_overrun_sec",
            "time_budget_ratio",
            "time_budget_hit",
            "target_step_hit",
            *SUMMARY_NUMERIC_FIELDS,
        ],
    )
    if not args.no_plots:
        plot_time_matched_noise(output_dir / "results.csv", output_dir / "plots")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
