from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .bounds import p2l_bound
from .data import CertPool, SplitBundle, deterministic_initial_support, stable_seed, stratified_indices
from .model import SmallMLP, compute_losses, eval_error, make_model, model_stats, train_model
from .scores import (
    ScoreConfig,
    cosine_matrix,
    score_greats_reference,
    score_pu_c,
    score_pu_f_or_g,
    tie_break_argmax,
)


CERTIFIED_METHODS = {"MaxLoss", "PU-C", "PU-F", "PU-G"}
REFERENCE_METHODS = {"GREATS"}
METHODS = ["MaxLoss", "PU-C", "PU-F", "PU-G", "GREATS"]


@dataclass(frozen=True)
class RunConfig:
    hidden_dim: int
    batch_size: int
    pretrain_epochs: int
    pretrain_lr: float
    p2l_epochs_per_iter: int
    p2l_lr: float
    weight_decay: float
    gamma: float
    delta: float
    max_total_support: int
    initial_per_class: int
    score: ScoreConfig
    greats_probe_size: int


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic_probe(pool: CertPool, split: SplitBundle, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(split.y_pretrain) > 0:
        take = min(count, len(split.y_pretrain))
        idx = stratified_indices(split.y_pretrain, take, stable_seed(seed, "greats-pretrain-probe"))
        return split.x_pretrain[idx], split.y_pretrain[idx]
    take = min(count, len(pool.y))
    idx = stratified_indices(pool.y, take, stable_seed(seed, "greats-cert-probe"))
    return pool.x[idx], pool.y[idx]


def run_p2l_method(
    method: str,
    seed: int,
    noise_rate: float,
    pretrain_fraction: float,
    split: SplitBundle,
    config: RunConfig,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    set_all_seeds(stable_seed(seed, method, int(pretrain_fraction * 10_000)))
    model = make_model(stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), config.hidden_dim, device)

    if len(split.y_pretrain):
        train_model(
            model,
            split.x_pretrain,
            split.y_pretrain,
            epochs=config.pretrain_epochs,
            lr=config.pretrain_lr,
            batch_size=config.batch_size,
            device=device,
            weight_decay=config.weight_decay,
        )

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    support_set = set(support)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y))
    stop_reached = False
    remaining_bad = len(split.pool.y)
    train_calls = 0

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, split.pool.x.shape[1]), dtype=np.float32)
        probe_y = np.empty((0,), dtype=np.int64)

    while True:
        if support:
            support_arr = np.asarray(support, dtype=np.int64)
            train_model(
                model,
                split.pool.x[support_arr],
                split.pool.y[support_arr],
                epochs=config.p2l_epochs_per_iter,
                lr=config.p2l_lr,
                batch_size=config.batch_size,
                device=device,
                weight_decay=config.weight_decay,
            )
            train_calls += 1

        non_support = np.asarray([idx for idx in range(len(split.pool.y)) if idx not in support_set], dtype=np.int64)
        if len(non_support) == 0:
            remaining_bad = 0
            stop_reached = True
            break

        losses = compute_losses(model, split.pool.x[non_support], split.pool.y[non_support], device)
        bad_local = np.flatnonzero(losses > config.gamma)
        remaining_bad = int(len(bad_local))
        if remaining_bad == 0:
            stop_reached = True
            break
        if len(support) >= limit:
            break

        bad_candidates = non_support[bad_local]
        bad_losses = losses[bad_local]
        chosen = choose_next(method, model, split.pool, support, bad_candidates, bad_losses, config, device, probe_x, probe_y)
        support.append(chosen)
        support_set.add(chosen)

    compression_size = len(support)
    effective_size = compression_size if stop_reached else compression_size + remaining_bad
    bound = p2l_bound(effective_size, len(split.pool.y), config.delta) if method in CERTIFIED_METHODS else None
    test_error = eval_error(model, split.x_test, split.y_test, device)
    diagnostics = selected_set_diagnostics(model, split.pool, support, device)
    runtime_sec = time.perf_counter() - started

    return {
        "method": method,
        "seed": seed,
        "noise_rate": noise_rate,
        "pretrain_fraction": pretrain_fraction,
        "n_cert": len(split.pool.y),
        "n_pretrain": len(split.y_pretrain),
        "compression_size": compression_size,
        "remaining_bad": remaining_bad,
        "effective_compression_size": effective_size,
        "certified_bound": bound,
        "test_error": test_error,
        "runtime_sec": runtime_sec,
        "stop_reached": int(stop_reached),
        "train_calls": train_calls,
        **diagnostics,
    }


def run_p2l_trace(
    method: str,
    seed: int,
    noise_rate: float,
    pretrain_fraction: float,
    split: SplitBundle,
    config: RunConfig,
    device: torch.device,
    record_every: int,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    set_all_seeds(stable_seed(seed, method, int(pretrain_fraction * 10_000)))
    model = make_model(stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), config.hidden_dim, device)

    if len(split.y_pretrain):
        train_model(
            model,
            split.x_pretrain,
            split.y_pretrain,
            epochs=config.pretrain_epochs,
            lr=config.pretrain_lr,
            batch_size=config.batch_size,
            device=device,
            weight_decay=config.weight_decay,
        )

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    initial_support_size = len(support)
    support_set = set(support)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y))
    train_calls = 0
    rows: list[dict[str, Any]] = []

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, split.pool.x.shape[1]), dtype=np.float32)
        probe_y = np.empty((0,), dtype=np.int64)

    record_every = max(1, int(record_every))

    while True:
        if support:
            support_arr = np.asarray(support, dtype=np.int64)
            train_model(
                model,
                split.pool.x[support_arr],
                split.pool.y[support_arr],
                epochs=config.p2l_epochs_per_iter,
                lr=config.p2l_lr,
                batch_size=config.batch_size,
                device=device,
                weight_decay=config.weight_decay,
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
            losses = compute_losses(model, split.pool.x[non_support], split.pool.y[non_support], device)
            bad_local = np.flatnonzero(losses > config.gamma)
            remaining_bad = int(len(bad_local))
            stop_reached = remaining_bad == 0

        hit_limit = (not stop_reached) and len(support) >= limit
        should_record = step == 0 or step % record_every == 0 or stop_reached or hit_limit
        if should_record:
            effective_size = len(support) if stop_reached else len(support) + remaining_bad
            bound = p2l_bound(effective_size, len(split.pool.y), config.delta) if method in CERTIFIED_METHODS else None
            test_error = eval_error(model, split.x_test, split.y_test, device)
            diagnostics = selected_set_diagnostics(model, split.pool, support, device)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "noise_rate": noise_rate,
                    "pretrain_fraction": pretrain_fraction,
                    "step": step,
                    "n_cert": len(split.pool.y),
                    "n_pretrain": len(split.y_pretrain),
                    "compression_size": len(support),
                    "remaining_bad": remaining_bad,
                    "effective_compression_size": effective_size,
                    "certified_bound": bound,
                    "test_error": test_error,
                    "runtime_sec": time.perf_counter() - started,
                    "stop_reached": int(stop_reached),
                    "hit_limit": int(hit_limit),
                    "train_calls": train_calls,
                    **diagnostics,
                }
            )

        if stop_reached or hit_limit:
            break

        bad_candidates = non_support[bad_local]
        bad_losses = losses[bad_local]
        chosen = choose_next(method, model, split.pool, support, bad_candidates, bad_losses, config, device, probe_x, probe_y)
        support.append(chosen)
        support_set.add(chosen)

    return rows


def run_p2l_es_budgets(
    method: str,
    seed: int,
    noise_rate: float,
    pretrain_fraction: float,
    split: SplitBundle,
    config: RunConfig,
    device: torch.device,
    budgets: list[int],
) -> list[dict[str, Any]]:
    budgets = sorted(set(max(0, int(budget)) for budget in budgets))
    if not budgets:
        return []

    started = time.perf_counter()
    set_all_seeds(stable_seed(seed, method, int(pretrain_fraction * 10_000)))
    model = make_model(stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), config.hidden_dim, device)

    if len(split.y_pretrain):
        train_model(
            model,
            split.x_pretrain,
            split.y_pretrain,
            epochs=config.pretrain_epochs,
            lr=config.pretrain_lr,
            batch_size=config.batch_size,
            device=device,
            weight_decay=config.weight_decay,
        )

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    initial_support_size = len(support)
    support_set = set(support)
    max_budget = max(budgets)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y), initial_support_size + max_budget)
    train_calls = 0
    rows: list[dict[str, Any]] = []
    recorded: set[int] = set()

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, split.pool.x.shape[1]), dtype=np.float32)
        probe_y = np.empty((0,), dtype=np.int64)

    def append_snapshot(
        budget: int,
        step: int,
        remaining_bad: int,
        stop_reached: bool,
        hit_limit: bool,
    ) -> None:
        effective_size = len(support) if stop_reached else len(support) + remaining_bad
        bound = p2l_bound(effective_size, len(split.pool.y), config.delta) if method in CERTIFIED_METHODS else None
        test_error = eval_error(model, split.x_test, split.y_test, device)
        diagnostics = selected_set_diagnostics(model, split.pool, support, device)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "noise_rate": noise_rate,
                "pretrain_fraction": pretrain_fraction,
                "es_budget": budget,
                "step": step,
                "n_cert": len(split.pool.y),
                "n_pretrain": len(split.y_pretrain),
                "compression_size": len(support),
                "remaining_bad": remaining_bad,
                "effective_compression_size": effective_size,
                "certified_bound": bound,
                "test_error": test_error,
                "runtime_sec": time.perf_counter() - started,
                "stop_reached": int(stop_reached),
                "hit_limit": int(hit_limit),
                "train_calls": train_calls,
                **diagnostics,
            }
        )
        recorded.add(budget)

    while True:
        if support:
            support_arr = np.asarray(support, dtype=np.int64)
            train_model(
                model,
                split.pool.x[support_arr],
                split.pool.y[support_arr],
                epochs=config.p2l_epochs_per_iter,
                lr=config.p2l_lr,
                batch_size=config.batch_size,
                device=device,
                weight_decay=config.weight_decay,
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
            losses = compute_losses(model, split.pool.x[non_support], split.pool.y[non_support], device)
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
        chosen = choose_next(method, model, split.pool, support, bad_candidates, bad_losses, config, device, probe_x, probe_y)
        support.append(chosen)
        support_set.add(chosen)

    return rows


def selected_set_diagnostics(
    model: SmallMLP,
    pool: CertPool,
    selected: list[int],
    device: torch.device,
) -> dict[str, float]:
    if not selected:
        return {
            "noise_hit_rate": 0.0,
            "duplicate_hit_rate": 0.0,
            "pairwise_feature_cosine": 0.0,
        }
    selected_arr = np.asarray(selected, dtype=np.int64)
    stats = model_stats(model, pool.x[selected_arr], pool.y[selected_arr], device)
    if len(selected_arr) < 2:
        pairwise_feature_cosine = 0.0
    else:
        sim = cosine_matrix(stats.embeddings, stats.embeddings)
        pairwise_feature_cosine = float(np.mean(sim[np.triu_indices_from(sim, k=1)]))
    return {
        "noise_hit_rate": float(np.mean(pool.is_noisy[selected_arr])),
        "duplicate_hit_rate": float(np.mean(pool.is_duplicate[selected_arr])),
        "pairwise_feature_cosine": pairwise_feature_cosine,
    }


def choose_next(
    method: str,
    model: SmallMLP,
    pool: CertPool,
    support: list[int],
    candidate: np.ndarray,
    candidate_losses: np.ndarray,
    config: RunConfig,
    device: torch.device,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
) -> int:
    if method == "MaxLoss" or not support:
        return tie_break_argmax(candidate, candidate_losses, pool.sample_id)

    support_arr = np.asarray(support, dtype=np.int64)
    support_stats = model_stats(model, pool.x[support_arr], pool.y[support_arr], device)
    cand_stats = model_stats(model, pool.x[candidate], pool.y[candidate], device)

    if method == "PU-C":
        scores = score_pu_c(candidate, candidate_losses, support_stats, cand_stats, config.score)
    elif method == "PU-F":
        scores = score_pu_f_or_g(
            candidate, candidate_losses, support_arr, support_stats, cand_stats, pool, config.score, False
        )
    elif method == "PU-G":
        scores = score_pu_f_or_g(
            candidate, candidate_losses, support_arr, support_stats, cand_stats, pool, config.score, True
        )
    elif method == "GREATS":
        probe_stats = model_stats(model, probe_x, probe_y, device)
        scores = score_greats_reference(cand_stats, probe_stats, support_stats, config.score.lambda_redundancy)
    else:
        raise ValueError(f"Unknown method: {method}")
    return tie_break_argmax(candidate, scores, pool.sample_id)
