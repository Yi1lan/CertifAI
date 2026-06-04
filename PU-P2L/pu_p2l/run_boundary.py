from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .data import make_experiment_dataset, make_pretrain_split, pac_bayes_enabled_for_dataset
from .io_utils import RESULT_FIELDS, SUMMARY_NUMERIC_FIELDS, write_csv, write_json, write_summary_views
from .model import resolve_device
from .plotting import plot_boundary
from .runner import DEFAULT_METHODS, METHODS, RunConfig, run_p2l_method
from .scores import ScoreConfig

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean PU-P2L boundary experiments.")
    parser.add_argument("--output-dir", type=str, default="results/synthetic_redundancy_hard/boundary")
    parser.add_argument("--dataset-name", type=str, default="synthetic_redundancy_hard")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.4])
    parser.add_argument(
        "--pretrain-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)

    parser.add_argument("--n-train", type=int, default=3000)
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
    parser.add_argument("--max-total-support", type=int, default=600)
    parser.add_argument("--initial-per-class", type=int, default=2)
    parser.add_argument("--greats-probe-size", type=int, default=64)
    add_pac_bayes_args(parser, default_samples=0)

    add_score_args(parser)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def add_pac_bayes_args(parser: argparse.ArgumentParser, default_samples: int) -> None:
    parser.add_argument(
        "--pac-bayes-samples",
        type=int,
        default=default_samples,
        help="Number of posterior samples for PAC-Bayes. Use 0 to disable; synthetic redundancy always forces 0.",
    )
    parser.add_argument("--pac-bayes-delta", type=float, default=None)
    parser.add_argument("--pac-bayes-delta-test", type=float, default=0.01)
    parser.add_argument("--pac-bayes-prior-sigma", type=float, default=0.05)
    parser.add_argument("--pac-bayes-posterior-sigma", type=float, default=0.05)
    parser.add_argument("--pac-bayes-train-epochs", type=int, default=0)
    parser.add_argument("--pac-bayes-lr", type=float, default=1e-3)
    parser.add_argument("--pac-bayes-batch-size", type=int, default=0)
    parser.add_argument("--pac-bayes-kl-weight", type=float, default=1.0)
    parser.add_argument("--pac-bayes-scope", type=str, default="head", choices=["head", "all"])


def add_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrain-training-mode",
        type=str,
        default="warm_start",
        choices=["warm_start", "support", "warm_start_and_support"],
        help=(
            "How pretrain data is used by P2L. 'warm_start' preserves the current PU-P2L behavior: "
            "train once on pretrain data, then iterate using selected certification support only. "
            "'support' matches the original P2L role: pretrain data is included in every iterative "
            "P2L training update but is not charged in the compression size. "
            "'warm_start_and_support' does both."
        ),
    )
    parser.add_argument("--c-loss", type=float, default=3.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="PU-R-Vol entropy boost. Larger values emphasize residual novelty when support spectral entropy is low.",
    )
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--lambda-redundancy", type=float, default=1.0)
    parser.add_argument("--global-redundancy-weight", type=float, default=1.5)
    parser.add_argument("--residual-rank", type=int, default=0)
    parser.add_argument("--residual-tol", type=float, default=1e-8)
    parser.add_argument("--manifold-k", type=int, default=10)
    parser.add_argument("--manifold-tau", type=float, default=0.5)
    parser.add_argument("--manifold-eigenvectors", type=int, default=16)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode-imbalance",
        type=float,
        default=0.85,
        help=(
            "Mode-A probability for mode_mnist, boundary_duplicate_mnist, boundary_duplicate_fashion_mnist, "
            "volume_duplicate_fashion_mnist, volume_gap_fashion_mnist, manifold_duplicate_fashion_mnist, "
            "and manifold_orbit_fashion_mnist."
        ),
    )
    parser.add_argument(
        "--boundary-augmentation",
        type=int,
        default=1,
        help=(
            "Augmentation multiplier marker for boundary, volume-gap, and manifold-orbit duplicate datasets; "
            "values >1 augment Mode A."
        ),
    )
    parser.add_argument(
        "--rotation-angles",
        type=float,
        nargs="+",
        default=[-60.0, -30.0, 0.0, 30.0, 60.0],
        help="Fixed rotation angles for rotated and manifold duplicate datasets.",
    )


def build_config(args: argparse.Namespace) -> RunConfig:
    pac_bayes_delta = getattr(args, "pac_bayes_delta", None)
    pac_bayes_samples = getattr(args, "pac_bayes_samples", 0)
    if not pac_bayes_enabled_for_dataset(getattr(args, "dataset_name", "")):
        pac_bayes_samples = 0
        setattr(args, "pac_bayes_samples", 0)
    return RunConfig(
        model_name=args.model_name,
        hidden_dim=args.hidden_dim,
        dropout_prob=args.dropout_prob,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        pretrain_training_mode=getattr(args, "pretrain_training_mode", "warm_start"),
        p2l_epochs_per_iter=args.p2l_epochs_per_iter,
        p2l_lr=args.p2l_lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        delta=args.delta,
        max_total_support=args.max_total_support,
        initial_per_class=args.initial_per_class,
        greats_probe_size=args.greats_probe_size,
        pac_bayes_samples=pac_bayes_samples,
        pac_bayes_delta=args.delta if pac_bayes_delta is None else pac_bayes_delta,
        pac_bayes_delta_test=getattr(args, "pac_bayes_delta_test", 0.01),
        pac_bayes_prior_sigma=getattr(args, "pac_bayes_prior_sigma", 1.0),
        pac_bayes_posterior_sigma=getattr(args, "pac_bayes_posterior_sigma", 0.05),
        pac_bayes_train_epochs=getattr(args, "pac_bayes_train_epochs", 0),
        pac_bayes_lr=getattr(args, "pac_bayes_lr", 1e-3),
        pac_bayes_batch_size=getattr(args, "pac_bayes_batch_size", 0),
        pac_bayes_kl_weight=getattr(args, "pac_bayes_kl_weight", 1.0),
        pac_bayes_scope=getattr(args, "pac_bayes_scope", "head"),
        score=ScoreConfig(
            gamma=args.gamma,
            c_loss=args.c_loss,
            alpha=args.alpha,
            mu=args.mu,
            lambda_redundancy=args.lambda_redundancy,
            global_redundancy_weight=args.global_redundancy_weight,
            residual_rank=getattr(args, "residual_rank", 0),
            residual_tol=getattr(args, "residual_tol", 1e-8),
            manifold_k=getattr(args, "manifold_k", 10),
            manifold_tau=getattr(args, "manifold_tau", 0.5),
            manifold_eigenvectors=getattr(args, "manifold_eigenvectors", 16),
        ),
    )


def make_dataset_from_args(args: argparse.Namespace, seed: int, noise_rate: float):
    return make_experiment_dataset(
        dataset_name=args.dataset_name,
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
        data_dir=args.data_dir,
        download=args.download_data,
        mode_imbalance=getattr(args, "mode_imbalance", 0.85),
        boundary_augmentation=getattr(args, "boundary_augmentation", 1),
        rotation_angles=getattr(args, "rotation_angles", None),
    )


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
        (seed, noise_rate, pretrain_fraction, method)
        for seed in args.seeds
        for noise_rate in args.noise_rates
        for pretrain_fraction in args.pretrain_fractions
        for method in args.methods
    ]
    progress = tqdm(tasks, desc="clean PU-P2L boundary") if tqdm is not None else tasks
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, float, float], Any] = {}
    for seed, noise_rate, pretrain_fraction, method in progress:
        key = (seed, noise_rate, pretrain_fraction)
        if key not in cache:
            bundle = make_dataset_from_args(args, seed, noise_rate)
            cache[key] = make_pretrain_split(bundle, pretrain_fraction, seed)
        row = run_p2l_method(method, seed, noise_rate, pretrain_fraction, cache[key], config, device)
        row["dataset"] = args.dataset_name
        rows.append(row)

    write_csv(output_dir / "results.csv", RESULT_FIELDS, rows)
    write_summary_views(
        output_dir,
        rows,
        group_fields=["dataset", "method", "noise_rate", "pretrain_fraction", "pretrain_training_mode"],
        numeric_fields=SUMMARY_NUMERIC_FIELDS,
    )
    if not args.no_plots:
        plot_boundary(output_dir / "results.csv", output_dir / "plots")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
