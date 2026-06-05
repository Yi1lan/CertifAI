from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .bounds import p2l_bound
from .data import SplitBundle, deterministic_initial_support, make_pretrain_split, stable_seed
from .io_utils import SUMMARY_NUMERIC_FIELDS, write_csv, write_json, write_summary_views
from .model import compute_losses, model_stats, resolve_device, train_model
from .plotting import COLORS, format_float_for_filename
from .run_boundary import add_dataset_args, add_pac_bayes_args, add_score_args, build_config, make_dataset_from_args
from .runner import (
    CERTIFIED_METHODS,
    METHODS,
    RunConfig,
    choose_next,
    deterministic_probe,
    frozen_reference_model,
    make_run_model,
    p2l_training_data,
    selected_set_diagnostics,
    set_all_seeds,
    use_pretrain_warm_start,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def unique_fields(fields: list[str]) -> list[str]:
    output: list[str] = []
    for field in fields:
        if field not in output:
            output.append(field)
    return output


SUMMARY_FIELDS = [
    "method",
    "dataset",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "pretrain_training_mode",
    "budget",
    "step",
    "n_cert",
    "n_pretrain",
    "compression_size",
    "remaining_bad",
    "effective_compression_size",
    "certified_bound",
    "runtime_sec",
    "stop_reached",
    "hit_limit",
    "train_calls",
    *SUMMARY_NUMERIC_FIELDS,
]
SUMMARY_FIELDS = unique_fields(SUMMARY_FIELDS)

SELECTION_FIELDS = [
    "method",
    "dataset",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "pretrain_training_mode",
    "budget",
    "step",
    "pool_index",
    "sample_id",
    "selected_order",
    "initial_support",
    "y_train",
    "true_y",
    "is_noisy",
    "is_duplicate",
    "group_id",
    "projection",
    "projection_source",
    "projection_x",
    "projection_y",
]

POOL_FIELDS = [
    "dataset",
    "seed",
    "noise_rate",
    "pretrain_fraction",
    "pool_index",
    "sample_id",
    "y_train",
    "true_y",
    "is_noisy",
    "is_duplicate",
    "group_id",
    "projection",
    "projection_source",
    "projection_x",
    "projection_y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise PU-P2L selected points under a fixed projection.")
    parser.add_argument("--output-dir", type=str, default="results/PU-R/ablations/selection_visualization")
    parser.add_argument("--dataset-name", type=str, default="boundary_duplicate_fashion_mnist")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.3])
    parser.add_argument("--pretrain-fractions", type=float, nargs="+", default=[0.3])
    parser.add_argument("--methods", type=str, nargs="+", default=["MaxLoss", "GREATS", "PU-R"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--projection", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--projection-source", type=str, default="raw", choices=["raw", "embedding"])
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--background-limit", type=int, default=6000)

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
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def pca_projection(features: np.ndarray) -> np.ndarray:
    matrix = features.reshape(features.shape[0], -1).astype(np.float64)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    coords = matrix @ vh[:2].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return coords.astype(np.float64)


def tsne_projection(features: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("t-SNE projection requires scikit-learn. Use --projection pca if it is unavailable.") from exc

    matrix = features.reshape(features.shape[0], -1).astype(np.float64)
    matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    safe_perplexity = min(float(perplexity), max(5.0, (len(matrix) - 1) / 3.0))
    projector = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return projector.fit_transform(matrix).astype(np.float64)


def projection_features(split: SplitBundle, config: RunConfig, device: torch.device, seed: int, source: str) -> np.ndarray:
    if source == "raw":
        return split.pool.x.reshape(len(split.pool.y), -1)
    if source != "embedding":
        raise ValueError(f"Unknown projection source: {source}")

    model = make_run_model(stable_seed(seed, "projection-model"), split, config, device)
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
    initial_support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    train_x, train_y = p2l_training_data(split, initial_support, config)
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
    return model_stats(model, split.pool.x, split.pool.y, device, config.inference_batch_size).embeddings


def project_pool(
    split: SplitBundle,
    config: RunConfig,
    device: torch.device,
    seed: int,
    projection: str,
    source: str,
    tsne_perplexity: float,
) -> np.ndarray:
    features = projection_features(split, config, device, seed, source)
    if projection == "pca":
        return pca_projection(features)
    if projection == "tsne":
        return tsne_projection(features, tsne_perplexity, stable_seed(seed, "selection-tsne"))
    raise ValueError(f"Unknown projection: {projection}")


def run_selection_snapshots(
    method: str,
    seed: int,
    noise_rate: float,
    pretrain_fraction: float,
    split: SplitBundle,
    config: RunConfig,
    device: torch.device,
    budgets: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    reference_model = frozen_reference_model(model) if method == "RHO-PretrainRef" else None

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    initial_support_size = len(support)
    support_set = set(support)
    budgets = sorted(set(max(0, int(budget)) for budget in budgets))
    max_budget = max(budgets)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y), initial_support_size + max_budget)
    train_calls = 0
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    recorded: set[int] = set()

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, *split.pool.x.shape[1:]), dtype=np.float32)
        probe_y = np.empty((0,), dtype=np.int64)

    def append_snapshot(budget: int, step: int, remaining_bad: int, stop_reached: bool, hit_limit: bool) -> None:
        effective_size = len(support) if stop_reached else len(support) + remaining_bad
        bound = p2l_bound(effective_size, len(split.pool.y), config.delta) if method in CERTIFIED_METHODS else None
        diagnostics = selected_set_diagnostics(model, split.pool, support, device, config.inference_batch_size, config.score)
        summary_rows.append(
            {
                "method": method,
                "seed": seed,
                "noise_rate": noise_rate,
                "pretrain_fraction": pretrain_fraction,
                "pretrain_training_mode": config.pretrain_training_mode,
                "budget": budget,
                "es_budget": budget,
                "step": step,
                "n_cert": len(split.pool.y),
                "n_pretrain": len(split.y_pretrain),
                "compression_size": len(support),
                "remaining_bad": remaining_bad,
                "effective_compression_size": effective_size,
                "certified_bound": bound,
                "runtime_sec": time.perf_counter() - started,
                "stop_reached": int(stop_reached),
                "hit_limit": int(hit_limit),
                "train_calls": train_calls,
                **diagnostics,
            }
        )
        for pos, pool_index in enumerate(support):
            selected_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "noise_rate": noise_rate,
                    "pretrain_fraction": pretrain_fraction,
                    "pretrain_training_mode": config.pretrain_training_mode,
                    "budget": budget,
                    "step": step,
                    "pool_index": int(pool_index),
                    "sample_id": int(split.pool.sample_id[pool_index]),
                    "selected_order": 0 if pos < initial_support_size else pos - initial_support_size + 1,
                    "initial_support": int(pos < initial_support_size),
                    "y_train": int(split.pool.y[pool_index]),
                    "true_y": int(split.pool.true_y[pool_index]),
                    "is_noisy": int(split.pool.is_noisy[pool_index]),
                    "is_duplicate": int(split.pool.is_duplicate[pool_index]),
                    "group_id": int(split.pool.group_id[pool_index]),
                }
            )
        recorded.add(budget)

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

        step = max(0, len(support) - initial_support_size)
        non_support = np.asarray([idx for idx in range(len(split.pool.y)) if idx not in support_set], dtype=np.int64)
        stop_reached = False
        remaining_bad = 0
        bad_local = np.array([], dtype=np.int64)
        losses = np.array([], dtype=np.float64)

        if len(non_support) == 0:
            stop_reached = True
        else:
            losses = compute_losses(
                model, split.pool.x[non_support], split.pool.y[non_support], device, config.inference_batch_size
            )
            bad_local = np.flatnonzero(losses > config.gamma)
            remaining_bad = int(len(bad_local))
            stop_reached = remaining_bad == 0

        hit_limit = (not stop_reached) and len(support) >= limit
        for budget in budgets:
            if budget not in recorded and budget <= step:
                append_snapshot(budget, step, remaining_bad, stop_reached, hit_limit)

        if stop_reached or hit_limit or step >= max_budget:
            for budget in budgets:
                if budget not in recorded:
                    append_snapshot(budget, step, remaining_bad, stop_reached, hit_limit)
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

    return summary_rows, selected_rows


def pool_projection_rows(
    dataset_name: str,
    split: SplitBundle,
    seed: int,
    noise_rate: float,
    pretrain_fraction: float,
    coords: np.ndarray,
    projection: str,
    source: str,
) -> list[dict[str, Any]]:
    rows = []
    for pool_index in range(len(split.pool.y)):
        rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "noise_rate": noise_rate,
                "pretrain_fraction": pretrain_fraction,
                "pool_index": pool_index,
                "sample_id": int(split.pool.sample_id[pool_index]),
                "y_train": int(split.pool.y[pool_index]),
                "true_y": int(split.pool.true_y[pool_index]),
                "is_noisy": int(split.pool.is_noisy[pool_index]),
                "is_duplicate": int(split.pool.is_duplicate[pool_index]),
                "group_id": int(split.pool.group_id[pool_index]),
                "projection": projection,
                "projection_source": source,
                "projection_x": float(coords[pool_index, 0]),
                "projection_y": float(coords[pool_index, 1]),
            }
        )
    return rows


def attach_projection(
    rows: list[dict[str, Any]],
    coords: np.ndarray,
    projection: str,
    source: str,
) -> None:
    for row in rows:
        pool_index = int(row["pool_index"])
        row["projection"] = projection
        row["projection_source"] = source
        row["projection_x"] = float(coords[pool_index, 0])
        row["projection_y"] = float(coords[pool_index, 1])


def float_or_none(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def plot_selection_group(
    pool_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    methods: list[str],
    budget: int,
    background_limit: int,
    path: Path,
    title: str,
) -> None:
    if not pool_rows:
        return
    rng = np.random.default_rng(0)
    background_rows = pool_rows
    if background_limit > 0 and len(pool_rows) > background_limit:
        indices = rng.choice(len(pool_rows), size=background_limit, replace=False)
        background_rows = [pool_rows[int(idx)] for idx in indices]

    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.4), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]
    label_values = sorted({int(row["true_y"]) for row in background_rows})
    cmap = plt.get_cmap("tab10")
    summary_lookup = {(row["method"], int(row["budget"])): row for row in summary_rows}

    for ax, method in zip(axes, methods):
        for label in label_values:
            label_rows = [row for row in background_rows if int(row["true_y"]) == label]
            ax.scatter(
                [float(row["projection_x"]) for row in label_rows],
                [float(row["projection_y"]) for row in label_rows],
                s=8,
                color=cmap(label % 10),
                alpha=0.16,
                linewidth=0,
            )

        method_rows = [
            row for row in selected_rows if row["method"] == method and int(row["budget"]) == int(budget)
        ]
        initial_rows = [row for row in method_rows if int(row["initial_support"]) == 1]
        selected_new = [row for row in method_rows if int(row["initial_support"]) == 0]
        color = COLORS.get(method, "#333333")
        if initial_rows:
            ax.scatter(
                [float(row["projection_x"]) for row in initial_rows],
                [float(row["projection_y"]) for row in initial_rows],
                s=34,
                marker="x",
                color="#111111",
                linewidth=1.0,
                alpha=0.85,
            )
        if selected_new:
            ax.scatter(
                [float(row["projection_x"]) for row in selected_new],
                [float(row["projection_y"]) for row in selected_new],
                s=30,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.90,
            )
        noisy_rows = [row for row in selected_new if int(row["is_noisy"]) == 1]
        duplicate_rows = [row for row in selected_new if int(row["is_duplicate"]) == 1]
        if duplicate_rows:
            ax.scatter(
                [float(row["projection_x"]) for row in duplicate_rows],
                [float(row["projection_y"]) for row in duplicate_rows],
                s=52,
                facecolors="none",
                edgecolors="#111111",
                linewidth=0.8,
                alpha=0.90,
            )
        if noisy_rows:
            ax.scatter(
                [float(row["projection_x"]) for row in noisy_rows],
                [float(row["projection_y"]) for row in noisy_rows],
                s=42,
                marker="x",
                color="#e66d50",
                linewidth=1.0,
                alpha=0.95,
            )

        summary = summary_lookup.get((method, int(budget)), {})
        noise_hit = float_or_none(summary, "noise_hit_rate")
        dup_hit = float_or_none(summary, "duplicate_hit_rate")
        subtitle = method
        if noise_hit is not None and dup_hit is not None:
            subtitle += f"\nnoise-hit={noise_hit:.2f}, dup-hit={dup_hit:.2f}"
        ax.set_title(subtitle, fontsize=10)
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("Projection dimension 2")
    for ax in axes:
        ax.set_xlabel("Projection dimension 1")

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="selected", markerfacecolor="#555555", markersize=7),
        Line2D([0], [0], marker="x", color="#111111", label="initial support", linestyle="None", markersize=7),
        Line2D([0], [0], marker="x", color="#e66d50", label="noisy selected", linestyle="None", markersize=7),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111111",
            label="duplicate selected",
            linestyle="None",
            markerfacecolor="none",
            markersize=7,
        ),
    ]
    axes[-1].legend(handles=legend_items, loc="best", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_selection_snapshots(
    pool_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    methods: list[str],
    budgets: list[int],
    background_limit: int,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(
        {
            (row["dataset"], int(row["seed"]), float(row["noise_rate"]), float(row["pretrain_fraction"]))
            for row in pool_rows
        },
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    for dataset_name, seed, noise_rate, pretrain_fraction in keys:
        pool_group = [
            row
            for row in pool_rows
            if row["dataset"] == dataset_name
            and int(row["seed"]) == seed
            and abs(float(row["noise_rate"]) - noise_rate) <= 1e-12
            and abs(float(row["pretrain_fraction"]) - pretrain_fraction) <= 1e-12
        ]
        selected_group = [
            row
            for row in selected_rows
            if row["dataset"] == dataset_name
            and int(row["seed"]) == seed
            and abs(float(row["noise_rate"]) - noise_rate) <= 1e-12
            and abs(float(row["pretrain_fraction"]) - pretrain_fraction) <= 1e-12
        ]
        summary_group = [
            row
            for row in summary_rows
            if row["dataset"] == dataset_name
            and int(row["seed"]) == seed
            and abs(float(row["noise_rate"]) - noise_rate) <= 1e-12
            and abs(float(row["pretrain_fraction"]) - pretrain_fraction) <= 1e-12
        ]
        for budget in budgets:
            title = (
                f"{dataset_name}, seed={seed}, noise={noise_rate:g}, "
                f"pretrain={pretrain_fraction:g}, budget={budget}"
            )
            filename = (
                f"selection_projection_{dataset_name}_seed_{seed}_noise_{format_float_for_filename(noise_rate)}_"
                f"pretrain_{format_float_for_filename(pretrain_fraction)}_budget_{budget}.png"
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


def main() -> None:
    args = parse_args()
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {METHODS}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = build_config(args)
    budgets = sorted(set(max(0, int(budget)) for budget in args.budgets))
    write_json(output_dir / "config.json", vars(args))

    tasks = [
        (seed, noise_rate, pretrain_fraction)
        for seed in args.seeds
        for noise_rate in args.noise_rates
        for pretrain_fraction in args.pretrain_fractions
    ]
    progress = tqdm(tasks, desc="selection projection setup") if tqdm is not None else tasks
    pool_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for seed, noise_rate, pretrain_fraction in progress:
        bundle = make_dataset_from_args(args, seed, noise_rate)
        split = make_pretrain_split(bundle, pretrain_fraction, seed)
        coords = project_pool(
            split,
            config,
            device,
            seed,
            args.projection,
            args.projection_source,
            args.tsne_perplexity,
        )
        pool_rows.extend(
            pool_projection_rows(
                args.dataset_name,
                split,
                seed,
                noise_rate,
                pretrain_fraction,
                coords,
                args.projection,
                args.projection_source,
            )
        )
        method_iter = tqdm(args.methods, desc=f"selection seed={seed} noise={noise_rate:g}") if tqdm is not None else args.methods
        for method in method_iter:
            method_summary, method_selected = run_selection_snapshots(
                method,
                seed,
                noise_rate,
                pretrain_fraction,
                split,
                config,
                device,
                budgets,
            )
            attach_projection(method_selected, coords, args.projection, args.projection_source)
            for row in method_summary:
                row["dataset"] = args.dataset_name
            for row in method_selected:
                row["dataset"] = args.dataset_name
            summary_rows.extend(method_summary)
            selected_rows.extend(method_selected)

    write_csv(output_dir / "pool_projection.csv", POOL_FIELDS, pool_rows)
    write_csv(output_dir / "selected_points.csv", SELECTION_FIELDS, selected_rows)
    write_csv(output_dir / "results.csv", SUMMARY_FIELDS, summary_rows)
    write_summary_views(
        output_dir,
        summary_rows,
        group_fields=["dataset", "method", "noise_rate", "pretrain_fraction", "pretrain_training_mode", "budget"],
        numeric_fields=["step", *SUMMARY_NUMERIC_FIELDS],
    )
    if not args.no_plots:
        plot_selection_snapshots(
            pool_rows,
            selected_rows,
            summary_rows,
            args.methods,
            budgets,
            args.background_limit,
            output_dir / "plots",
        )


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    main()
