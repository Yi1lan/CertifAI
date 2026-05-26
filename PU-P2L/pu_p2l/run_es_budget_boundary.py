from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .data import make_pretrain_split, make_redundancy_dataset
from .io_utils import summarize, write_csv, write_json
from .model import resolve_device
from .plotting import plot_es_budget_boundary
from .runner import CERTIFIED_METHODS, run_p2l_es_budgets
from .run_boundary import build_config

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ES_BUDGET_METHODS = ["MaxLoss", "PU-C", "PU-F", "PU-G"]
ES_BUDGET_FIELDS = [
    "method",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "es_budget",
    "step",
    "n_cert",
    "n_pretrain",
    "compression_size",
    "remaining_bad",
    "effective_compression_size",
    "certified_bound",
    "test_error",
    "runtime_sec",
    "stop_reached",
    "hit_limit",
    "train_calls",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "pairwise_feature_cosine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-ES PU-P2L boundary experiments.")
    parser.add_argument("--output-dir", type=str, default="results/pu_p2l_es_budget_boundary_hard")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.4])
    parser.add_argument(
        "--pretrain-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
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

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--p2l-epochs-per-iter", type=int, default=1)
    parser.add_argument("--p2l-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=-math.log(0.5))
    parser.add_argument("--delta", type=float, default=0.035)
    parser.add_argument("--max-total-support", type=int, default=600)
    parser.add_argument("--initial-per-class", type=int, default=2)
    parser.add_argument("--greats-probe-size", type=int, default=64)

    parser.add_argument("--r-h", type=int, default=5)
    parser.add_argument("--r-consensus", type=int, default=10)
    parser.add_argument("--c-loss", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--lambda-redundancy", type=float, default=1.0)
    parser.add_argument("--global-redundancy-weight", type=float, default=1.5)
    parser.add_argument("--consensus-weight", type=float, default=1.25)
    parser.add_argument("--noise-penalty", type=float, default=2.5)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.methods) - CERTIFIED_METHODS)
    if unknown:
        raise ValueError(f"Unknown or non-certified methods: {unknown}. Valid methods: {ES_BUDGET_METHODS}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = build_config(args)
    write_json(output_dir / "config.json", vars(args))

    tasks = [
        (seed, noise_rate, pretrain_fraction, method)
        for seed in args.seeds
        for noise_rate in args.noise_rates
        for pretrain_fraction in args.pretrain_fractions
        for method in args.methods
    ]
    progress = tqdm(tasks, desc="PU-P2L fixed-ES boundary") if tqdm is not None else tasks
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, float, float], Any] = {}
    for seed, noise_rate, pretrain_fraction, method in progress:
        key = (seed, noise_rate, pretrain_fraction)
        if key not in cache:
            bundle = make_redundancy_dataset(
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                duplicate_groups=args.duplicate_groups,
                duplicates_per_group=args.duplicates_per_group,
                noise_rate=noise_rate,
                ambiguous_fraction=args.ambiguous_fraction,
                cluster_std=args.cluster_std,
                band_std=args.band_std,
                duplicate_std=args.duplicate_std,
            )
            cache[key] = make_pretrain_split(bundle, pretrain_fraction, seed)
        rows.extend(
            run_p2l_es_budgets(
                method,
                seed,
                noise_rate,
                pretrain_fraction,
                cache[key],
                config,
                device,
                args.es_budgets,
            )
        )

    write_csv(output_dir / "results.csv", ES_BUDGET_FIELDS, rows)
    summary = summarize(
        rows,
        group_fields=["method", "noise_rate", "pretrain_fraction", "es_budget"],
        numeric_fields=[
            "step",
            "compression_size",
            "remaining_bad",
            "effective_compression_size",
            "certified_bound",
            "test_error",
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
        plot_es_budget_boundary(output_dir / "results.csv", output_dir / "plots")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
