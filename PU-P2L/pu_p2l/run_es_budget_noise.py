from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .data import make_pretrain_split
from .io_utils import summarize, write_csv, write_json
from .model import resolve_device
from .plotting import plot_es_budget_noise
from .runner import METHODS, run_p2l_es_budgets
from .run_boundary import add_pac_bayes_args, build_config, make_dataset_from_args
from .run_es_budget_boundary import ES_BUDGET_FIELDS, ES_BUDGET_METHODS

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-ES PU-P2L noise-sweep experiments.")
    parser.add_argument("--output-dir", type=str, default="results/synthetic_redundancy_hard/es_budget_noise")
    parser.add_argument("--dataset-name", type=str, default="synthetic_redundancy_hard")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--pretrain-fraction", type=float, default=0.0)
    parser.add_argument("--es-budgets", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--methods", type=str, nargs="+", default=ES_BUDGET_METHODS)

    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--duplicate-groups", type=int, default=40)
    parser.add_argument("--duplicates-per-group", type=int, default=10)
    parser.add_argument("--ambiguous-fraction", type=float, default=0.35)
    parser.add_argument("--cluster-std", type=float, default=0.45)
    parser.add_argument("--band-std", type=float, default=0.35)
    parser.add_argument("--duplicate-std", type=float, default=0.015)

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
    parser.add_argument("--max-total-support", type=int, default=600)
    parser.add_argument("--initial-per-class", type=int, default=2)
    parser.add_argument("--greats-probe-size", type=int, default=64)
    add_pac_bayes_args(parser, default_samples=0)

    parser.add_argument("--r-h", type=int, default=5)
    parser.add_argument("--r-consensus", type=int, default=10)
    parser.add_argument("--c-loss", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--lambda-redundancy", type=float, default=1.0)
    parser.add_argument("--global-redundancy-weight", type=float, default=1.5)
    parser.add_argument("--consensus-weight", type=float, default=1.25)
    parser.add_argument("--noise-penalty", type=float, default=2.5)
    parser.add_argument("--residual-rank", type=int, default=0)
    parser.add_argument("--residual-tol", type=float, default=1e-8)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {METHODS}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = build_config(args)
    write_json(output_dir / "config.json", vars(args))

    tasks = [
        (seed, noise_rate, method)
        for seed in args.seeds
        for noise_rate in args.noise_rates
        for method in args.methods
    ]
    progress = tqdm(tasks, desc="PU-P2L fixed-ES noise") if tqdm is not None else tasks
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, float], Any] = {}
    for seed, noise_rate, method in progress:
        key = (seed, noise_rate)
        if key not in cache:
            bundle = make_dataset_from_args(args, seed, noise_rate)
            cache[key] = make_pretrain_split(bundle, args.pretrain_fraction, seed)
        budget_rows = run_p2l_es_budgets(
            method,
            seed,
            noise_rate,
            args.pretrain_fraction,
            cache[key],
            config,
            device,
            args.es_budgets,
        )
        for row in budget_rows:
            row["dataset"] = args.dataset_name
        rows.extend(budget_rows)

    write_csv(output_dir / "results.csv", ES_BUDGET_FIELDS, rows)
    summary = summarize(
        rows,
        group_fields=["dataset", "method", "noise_rate", "pretrain_fraction", "es_budget"],
        numeric_fields=[
            "step",
            "compression_size",
            "remaining_bad",
            "effective_compression_size",
            "certified_bound",
            "test_error",
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
        ],
    )
    summary_fields: list[str] = []
    for row in summary:
        for field in row:
            if field not in summary_fields:
                summary_fields.append(field)
    write_csv(output_dir / "summary.csv", summary_fields, summary)
    if not args.no_plots:
        plot_es_budget_noise(output_dir / "results.csv", output_dir / "plots")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
