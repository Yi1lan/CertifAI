from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bounds import p2l_bound
from .data import deterministic_initial_support, make_pretrain_split, stable_seed
from .io_utils import write_csv, write_json, write_summary_views
from .model import compute_losses, eval_error, eval_inappropriate_risk, resolve_device, train_model
from .adaptive_generalization_bounds import ada_clipped_gaussian_bound, self_selected_generalization_bound
from .plotting import plot_generalization_bounds
from .run_boundary import add_dataset_args, add_pac_bayes_args, add_score_args, build_config, make_dataset_from_args
from .runner import (
    CERTIFIED_METHODS,
    METHODS,
    choose_next,
    deterministic_probe,
    frozen_reference_model,
    make_run_model,
    pac_bayes_parameter_names,
    pac_bayes_stats,
    p2l_training_data,
    parameter_vector,
    selected_set_diagnostics,
    set_all_seeds,
    use_pretrain_warm_start,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


GENERALIZATION_FIELDS = [
    "method",
    "dataset",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "pretrain_training_mode",
    "n_cert",
    "n_pretrain",
    "compression_size",
    "remaining_bad",
    "effective_compression_size",
    "selection_steps",
    "certified_bound",
    "test_error",
    "test_inappropriate_risk",
    "pool_empirical_error",
    "support_empirical_error",
    "pac_bayes_bound",
    "pac_bayes_empirical_risk",
    "pac_bayes_mc_upper",
    "pac_bayes_kl",
    "self_selected_bound",
    "self_selected_bound_raw",
    "self_selected_empirical_risk",
    "self_selected_initial_gap",
    "self_selected_reciprocal_gap",
    "self_selected_total_gap",
    "self_selected_lipschitz_sample",
    "self_selected_log_cover_constant",
    "self_selected_concentration_rate_constant",
    "self_selected_data_diameter",
    "self_selected_dimension",
    "self_selected_wasserstein_p",
    "ada_bound",
    "ada_bound_raw",
    "ada_empirical_risk",
    "ada_alpha",
    "ada_query_count",
    "ada_initial_size",
    "ada_final_size",
    "ada_sigma",
    "ada_beta",
    "ada_epsilon",
    "ada_delta_sum",
    "ada_gamma_star",
    "runtime_sec",
    "stop_reached",
    "train_calls",
    "loss_eval_count",
    "noise_hit_rate",
    "duplicate_hit_rate",
    "pairwise_feature_cosine",
    "mean_support_redundancy",
    "max_support_redundancy",
    "mean_selected_residual_novelty",
    "local_redundancy_hit_rate",
    "residual_redundancy_hit_rate",
    "strong_redundancy_hit_rate",
    "group_revisit_rate",
    "unique_group_fraction",
    "max_group_selection_fraction",
    "mode_entropy",
    "minority_mode_fraction",
    "spectral_entropy",
    "dynamic_mu",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MNIST generalization-bound comparisons against external paper bounds."
    )
    parser.add_argument("--output-dir", type=str, default="results/genearlization_bound/mnist")
    parser.add_argument("--dataset-name", type=str, default="mnist")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.4])
    parser.add_argument(
        "--pretrain-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    parser.add_argument("--methods", type=str, nargs="+", default=["PU-R-Vol"])

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
    add_pac_bayes_args(parser, default_samples=50)

    add_score_args(parser)

    parser.add_argument("--bound-delta", type=float, default=None)
    parser.add_argument("--ssd-empirical-risk", type=str, default="support", choices=["support", "pool"])
    parser.add_argument("--ssd-loss-lipschitz", type=float, default=1.0)
    parser.add_argument("--ssd-wasserstein-p", type=float, default=1.0)
    parser.add_argument("--ssd-dimension", type=str, default="auto")
    parser.add_argument("--ssd-data-diameter", type=str, default="auto")
    parser.add_argument("--ssd-changed-per-iter", type=int, default=1)
    parser.add_argument("--ssd-sample-lipschitz", type=str, default="auto")
    parser.add_argument("--ssd-log-cover-constant", type=float, default=None)
    parser.add_argument("--ssd-concentration-rate-constant", type=float, default=None)

    parser.add_argument("--ada-beta-prime", type=float, default=None)
    parser.add_argument("--ada-query-count-mode", type=str, default="train_calls", choices=["one", "train_calls", "selection_steps", "loss_evaluations"])
    parser.add_argument("--ada-initial-size-mode", type=str, default="cert", choices=["cert", "pretrain", "total"])
    parser.add_argument("--ada-sigma-min", type=float, default=1e-4)
    parser.add_argument("--ada-sigma-max", type=float, default=1.0)
    parser.add_argument("--ada-sigma-grid-size", type=int, default=14)
    parser.add_argument("--ada-beta-min", type=float, default=1e-12)
    parser.add_argument("--ada-beta-max", type=float, default=0.5)
    parser.add_argument("--ada-beta-grid-size", type=int, default=14)
    parser.add_argument("--ada-epsilon-max", type=float, default=2.0)
    parser.add_argument("--ada-epsilon-grid-size", type=int, default=14)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_auto_float(value: str, computed: float) -> float:
    if value.strip().lower() == "auto":
        return float(computed)
    return float(value)


def parse_auto_optional_float(value: str, computed: float | None) -> float | None:
    if value.strip().lower() == "auto":
        return computed
    return float(value)


def mnist_domain_diameter(input_shape: tuple[int, ...]) -> float:
    feature_dim = int(np.prod(input_shape))
    pixel_range = 1.0 / 0.3081
    return float(math.sqrt(feature_dim * pixel_range * pixel_range + 1.0))


def observed_domain_diameter(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 1.0
    flat = x.reshape(x.shape[0], -1)
    feature_ranges = np.ptp(flat, axis=0)
    label_range = float(np.max(y) - np.min(y)) if len(y) else 0.0
    return float(math.sqrt(float(np.sum(feature_ranges * feature_ranges)) + label_range * label_range))


def auto_dimension(split) -> float:
    return float(int(np.prod(split.pool.x.shape[1:])) + 1)


def auto_diameter(dataset_name: str, split) -> float:
    if dataset_name.strip().lower() in {"mnist", "binary_mnist", "binarymnist"}:
        return mnist_domain_diameter(tuple(split.pool.x.shape[1:]))
    return observed_domain_diameter(split.pool.x, split.pool.y)


def ada_sizes(args: argparse.Namespace, split) -> tuple[int, int]:
    n_cert = len(split.pool.y)
    n_pretrain = len(split.y_pretrain)
    if args.ada_initial_size_mode == "cert":
        return max(1, n_cert), max(1, n_cert)
    if args.ada_initial_size_mode == "pretrain":
        return max(1, n_pretrain), max(1, n_pretrain + n_cert)
    return max(1, n_pretrain + n_cert), max(1, n_pretrain + n_cert)


def ada_query_count(args: argparse.Namespace, train_calls: int, selection_steps: int, loss_eval_count: int) -> int:
    if args.ada_query_count_mode == "one":
        return 1
    if args.ada_query_count_mode == "selection_steps":
        return max(1, selection_steps + 1)
    if args.ada_query_count_mode == "loss_evaluations":
        return max(1, int(loss_eval_count))
    return max(1, int(train_calls))


def adaptive_bound_stats(
    args: argparse.Namespace,
    split,
    model,
    support: list[int],
    initial_support_size: int,
    train_calls: int,
    loss_eval_count: int,
    device: torch.device,
    config,
) -> dict[str, Any]:
    support_arr = np.asarray(support, dtype=np.int64)
    if len(support_arr):
        support_empirical_error = eval_error(
            model,
            split.pool.x[support_arr],
            split.pool.y[support_arr],
            device,
            config.inference_batch_size,
        )
    else:
        support_empirical_error = 1.0
    pool_empirical_error = eval_error(model, split.pool.x, split.pool.y, device, config.inference_batch_size)

    selection_steps = max(0, len(support) - initial_support_size)
    bound_delta = args.delta if args.bound_delta is None else args.bound_delta
    ssd_dimension = parse_auto_float(args.ssd_dimension, auto_dimension(split))
    ssd_diameter = parse_auto_float(args.ssd_data_diameter, auto_diameter(args.dataset_name, split))
    default_sample_lipschitz = None
    ssd_sample_lipschitz = parse_auto_optional_float(args.ssd_sample_lipschitz, default_sample_lipschitz)
    ssd_empirical = support_empirical_error if args.ssd_empirical_risk == "support" else pool_empirical_error
    self_selected = self_selected_generalization_bound(
        ssd_empirical,
        len(split.pool.y),
        bound_delta,
        selection_steps,
        loss_lipschitz=args.ssd_loss_lipschitz,
        data_diameter=ssd_diameter,
        dimension=ssd_dimension,
        wasserstein_p=args.ssd_wasserstein_p,
        changed_per_iter=args.ssd_changed_per_iter,
        sample_lipschitz=ssd_sample_lipschitz,
        log_cover_constant=args.ssd_log_cover_constant,
        concentration_rate_constant=args.ssd_concentration_rate_constant,
    )

    ada_initial, ada_final = ada_sizes(args, split)
    ada_k = ada_query_count(args, train_calls, selection_steps, loss_eval_count)
    ada_beta_prime = bound_delta if args.ada_beta_prime is None else args.ada_beta_prime
    ada = ada_clipped_gaussian_bound(
        pool_empirical_error,
        final_size=ada_final,
        initial_size=ada_initial,
        query_count=ada_k,
        beta_prime=ada_beta_prime,
        sigma_min=args.ada_sigma_min,
        sigma_max=args.ada_sigma_max,
        sigma_grid_size=args.ada_sigma_grid_size,
        beta_min=args.ada_beta_min,
        beta_max=args.ada_beta_max,
        beta_grid_size=args.ada_beta_grid_size,
        epsilon_max=args.ada_epsilon_max,
        epsilon_grid_size=args.ada_epsilon_grid_size,
    )

    return {
        "pool_empirical_error": pool_empirical_error,
        "support_empirical_error": support_empirical_error,
        "selection_steps": selection_steps,
        "self_selected_bound": self_selected.bound,
        "self_selected_bound_raw": self_selected.raw_bound,
        "self_selected_empirical_risk": self_selected.empirical_risk,
        "self_selected_initial_gap": self_selected.initial_gap,
        "self_selected_reciprocal_gap": self_selected.reciprocal_gap,
        "self_selected_total_gap": self_selected.total_gap,
        "self_selected_lipschitz_sample": self_selected.lipschitz_sample,
        "self_selected_log_cover_constant": self_selected.log_cover_constant,
        "self_selected_concentration_rate_constant": self_selected.concentration_rate_constant,
        "self_selected_data_diameter": self_selected.data_diameter,
        "self_selected_dimension": self_selected.dimension,
        "self_selected_wasserstein_p": self_selected.wasserstein_p,
        "ada_bound": ada.bound,
        "ada_bound_raw": ada.raw_bound,
        "ada_empirical_risk": ada.empirical_risk,
        "ada_alpha": ada.alpha,
        "ada_query_count": ada.query_count,
        "ada_initial_size": ada.initial_size,
        "ada_final_size": ada.final_size,
        "ada_sigma": ada.sigma,
        "ada_beta": ada.beta,
        "ada_epsilon": ada.epsilon,
        "ada_delta_sum": ada.delta_sum,
        "ada_gamma_star": ada.gamma_star,
    }


def run_generalization_method(
    method: str,
    seed: int,
    noise_rate: float,
    pretrain_fraction: float,
    split,
    args: argparse.Namespace,
    config,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    set_all_seeds(stable_seed(seed, method, int(pretrain_fraction * 10_000)))
    model = make_run_model(
        stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), split, config, device
    )

    if use_pretrain_warm_start(config) and len(split.y_pretrain):
        train_model(
            model,
            split.x_pretrain,
            split.y_pretrain,
            epochs=config.pretrain_epochs,
            lr=config.pretrain_lr,
            batch_size=config.batch_size,
            device=device,
            weight_decay=config.weight_decay,
            optimizer_name=config.optimizer,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    prior_vector = parameter_vector(model, pac_bayes_parameter_names(model, config.pac_bayes_scope))
    reference_model = frozen_reference_model(model) if method == "RHO-PretrainRef" else None

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    initial_support_size = len(support)
    support_set = set(support)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y))
    stop_reached = False
    remaining_bad = len(split.pool.y)
    train_calls = 0
    loss_eval_count = 0

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, *split.pool.x.shape[1:]), dtype=np.float32)
        probe_y = np.empty((0,), dtype=np.int64)

    while True:
        train_x, train_y = p2l_training_data(split, support, config)
        if len(train_y):
            train_model(
                model,
                train_x,
                train_y,
                epochs=config.p2l_epochs_per_iter,
                lr=config.p2l_lr,
                batch_size=config.batch_size,
                device=device,
                weight_decay=config.weight_decay,
                optimizer_name=config.optimizer,
                momentum=config.momentum,
                nesterov=config.nesterov,
            )
            train_calls += 1

        non_support = np.asarray([idx for idx in range(len(split.pool.y)) if idx not in support_set], dtype=np.int64)
        if len(non_support) == 0:
            remaining_bad = 0
            stop_reached = True
            break

        losses = compute_losses(
            model, split.pool.x[non_support], split.pool.y[non_support], device, config.inference_batch_size
        )
        loss_eval_count += int(len(non_support))
        bad_local = np.flatnonzero(losses > config.gamma)
        remaining_bad = int(len(bad_local))
        if remaining_bad == 0:
            stop_reached = True
            break
        if len(support) >= limit:
            break

        bad_candidates = non_support[bad_local]
        bad_losses = losses[bad_local]
        chosen = choose_next(
            method,
            model,
            split.pool,
            support,
            bad_candidates,
            bad_losses,
            config,
            device,
            probe_x,
            probe_y,
            reference_model,
        )
        support.append(chosen)
        support_set.add(chosen)

    compression_size = len(support)
    effective_size = compression_size if stop_reached else compression_size + remaining_bad
    certified_bound = p2l_bound(effective_size, len(split.pool.y), config.delta) if method in CERTIFIED_METHODS else None
    test_error = eval_error(model, split.x_test, split.y_test, device, config.inference_batch_size)
    test_inappropriate_risk = eval_inappropriate_risk(
        model, split.x_test, split.y_test, config.gamma, device, config.inference_batch_size
    )
    pac_stats = pac_bayes_stats(
        model,
        prior_vector,
        split,
        config,
        device,
        stable_seed(seed, f"{method}-pac-bayes-generalization", int(pretrain_fraction * 10_000)),
    )
    adaptive_stats = adaptive_bound_stats(
        args,
        split,
        model,
        support,
        initial_support_size,
        train_calls,
        loss_eval_count,
        device,
        config,
    )
    diagnostics = selected_set_diagnostics(model, split.pool, support, device, config.inference_batch_size, config.score)
    runtime_sec = time.perf_counter() - started

    return {
        "method": method,
        "seed": seed,
        "noise_rate": noise_rate,
        "pretrain_fraction": pretrain_fraction,
        "pretrain_training_mode": config.pretrain_training_mode,
        "n_cert": len(split.pool.y),
        "n_pretrain": len(split.y_pretrain),
        "compression_size": compression_size,
        "remaining_bad": remaining_bad,
        "effective_compression_size": effective_size,
        "certified_bound": certified_bound,
        "test_error": test_error,
        "test_inappropriate_risk": test_inappropriate_risk,
        **pac_stats,
        **adaptive_stats,
        "runtime_sec": runtime_sec,
        "stop_reached": int(stop_reached),
        "train_calls": train_calls,
        "loss_eval_count": loss_eval_count,
        **diagnostics,
    }


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
    progress = tqdm(tasks, desc="MNIST generalization bounds") if tqdm is not None else tasks
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, float, float], Any] = {}
    for seed, noise_rate, pretrain_fraction, method in progress:
        key = (seed, noise_rate, pretrain_fraction)
        if key not in cache:
            bundle = make_dataset_from_args(args, seed, noise_rate)
            cache[key] = make_pretrain_split(bundle, pretrain_fraction, seed)
        row = run_generalization_method(method, seed, noise_rate, pretrain_fraction, cache[key], args, config, device)
        row["dataset"] = args.dataset_name
        rows.append(row)

    write_csv(output_dir / "results.csv", GENERALIZATION_FIELDS, rows)
    numeric_fields = [
        field
        for field in GENERALIZATION_FIELDS
        if field not in {"method", "dataset", "seed", "noise_rate", "pretrain_fraction", "pretrain_training_mode"}
    ]
    write_summary_views(
        output_dir,
        rows,
        group_fields=["dataset", "method", "noise_rate", "pretrain_fraction", "pretrain_training_mode"],
        numeric_fields=numeric_fields,
    )
    if not args.no_plots:
        plot_generalization_bounds(output_dir / "results.csv", output_dir / "plots")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
