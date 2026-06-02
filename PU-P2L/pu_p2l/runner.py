from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .bounds import gaussian_kl_isotropic, p2l_bound, pac_bayes_bound
from .data import CertPool, SplitBundle, deterministic_initial_support, stable_seed, stratified_indices
from .model import compute_losses, eval_error, eval_inappropriate_risk, make_model, model_stats, train_model
from .scores import (
    ScoreConfig,
    cosine_matrix,
    normalized_spectral_entropy,
    row_normed,
    score_ablation,
    score_marginal,
    score_greats_reference,
    score_pu_r,
    score_pu_r_manifold,
    score_pu_r_vol,
    support_span_basis,
    tie_break_argmax,
)


ABLATION_METHODS = {
    "ClippedLoss",
    "ResidualOnly",
    "RedundancyOnly",
    "Loss+Residual",
    "Loss-Redundancy",
    "PU-C-style",
    "Marginal+Residual",
    "Marginal-Redundancy",
    "Marginal+Residual-Redundancy",
}
CERTIFIED_METHODS = {"MaxLoss", "Marginal", "PU-R", "PU-R-Vol", "PU-R-Manifold", *ABLATION_METHODS}
REFERENCE_METHODS = {"GREATS"}
METHODS = [
    "MaxLoss",
    "Marginal",
    "PU-R",
    "PU-R-Vol",
    "PU-R-Manifold",
    "ClippedLoss",
    "ResidualOnly",
    "RedundancyOnly",
    "Loss+Residual",
    "Loss-Redundancy",
    "PU-C-style",
    "Marginal+Residual",
    "Marginal-Redundancy",
    "Marginal+Residual-Redundancy",
    "GREATS",
]
DEFAULT_METHODS = ["MaxLoss", "Marginal", "PU-R", "PU-R-Vol", "PU-R-Manifold", "GREATS"]


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    hidden_dim: int
    dropout_prob: float
    batch_size: int
    inference_batch_size: int
    pretrain_epochs: int
    pretrain_lr: float
    p2l_epochs_per_iter: int
    p2l_lr: float
    optimizer: str
    momentum: float
    nesterov: bool
    weight_decay: float
    gamma: float
    delta: float
    max_total_support: int
    initial_per_class: int
    score: ScoreConfig
    greats_probe_size: int
    pac_bayes_samples: int
    pac_bayes_delta: float
    pac_bayes_delta_test: float
    pac_bayes_prior_sigma: float
    pac_bayes_posterior_sigma: float
    pac_bayes_train_epochs: int
    pac_bayes_lr: float
    pac_bayes_batch_size: int
    pac_bayes_kl_weight: float
    pac_bayes_scope: str


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


def split_num_classes(split: SplitBundle) -> int:
    labels = [split.pool.y, split.y_pretrain, split.y_test]
    max_label = max((int(np.max(y)) for y in labels if len(y)), default=1)
    return max_label + 1


def make_run_model(seed: int, split: SplitBundle, config: RunConfig, device: torch.device) -> nn.Module:
    return make_model(
        seed=seed,
        model_name=config.model_name,
        input_shape=tuple(split.pool.x.shape[1:]),
        num_classes=split_num_classes(split),
        hidden_dim=config.hidden_dim,
        dropout_prob=config.dropout_prob,
        device=device,
    )


def parameter_items(model: nn.Module, names: set[str] | None = None) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if names is None or name in names]


def parameter_vector(model: nn.Module, names: set[str] | None = None) -> torch.Tensor:
    return parameters_to_vector([param.detach() for _, param in parameter_items(model, names)]).detach().cpu()


def set_parameter_vector(model: nn.Module, vector: torch.Tensor, device: torch.device) -> None:
    vector_to_parameters(vector.detach().to(device), model.parameters())


def pac_bayes_parameter_names(model: nn.Module, scope: str) -> set[str]:
    names = [name for name, _ in model.named_parameters()]
    if scope == "all":
        return set(names)
    if scope != "head":
        raise ValueError("PAC-Bayes scope must be 'head' or 'all'.")

    head_names = [
        name
        for name in names
        if name.startswith("head.") or name.startswith("l4.") or name.startswith("fc.")
    ]
    if head_names:
        return set(head_names)
    return set(names[-2:])


def vector_to_named_parameters(
    model: nn.Module,
    vector: torch.Tensor,
    selected_names: set[str],
) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in model.named_parameters():
        if name in selected_names:
            numel = param.numel()
            params[name] = vector[offset : offset + numel].view_as(param)
            offset += numel
        else:
            params[name] = param.detach()
    if offset != vector.numel():
        raise ValueError("PAC-Bayes parameter vector size does not match selected parameters.")
    return params


def torch_gaussian_kl_isotropic(
    posterior_mean: torch.Tensor,
    prior_mean: torch.Tensor,
    posterior_sigma: float,
    prior_sigma: float,
) -> torch.Tensor:
    dim = posterior_mean.numel()
    variance_ratio = (posterior_sigma / prior_sigma) ** 2
    log_ratio = 2.0 * np.log(prior_sigma / posterior_sigma)
    diff_sq = torch.sum((posterior_mean - prior_mean) ** 2)
    return 0.5 * (
        dim * (variance_ratio - 1.0 + log_ratio) + diff_sq / (prior_sigma**2)
    )


def pac_bayes_stats(
    model: nn.Module,
    prior_vector: torch.Tensor,
    split: SplitBundle,
    config: RunConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float | None]:
    if config.pac_bayes_samples <= 0:
        return {
            "pac_bayes_bound": None,
            "pac_bayes_empirical_risk": None,
            "pac_bayes_mc_upper": None,
            "pac_bayes_kl": None,
        }

    if config.pac_bayes_prior_sigma <= 0 or config.pac_bayes_posterior_sigma <= 0:
        raise ValueError("PAC-Bayes Gaussian sigmas must be positive.")

    torch.manual_seed(seed)
    model.eval()
    selected_names = pac_bayes_parameter_names(model, config.pac_bayes_scope)
    prior_selected = prior_vector.to(device)
    posterior_mean = parameter_vector(model, selected_names).to(device).detach().clone().requires_grad_(True)
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
    x_pool, y_pool = split.pool.x, split.pool.y
    x_t = torch.as_tensor(x_pool, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_pool, dtype=torch.long, device=device)
    n = len(y_pool)

    def logits_from_vector(vector: torch.Tensor, x_batch: torch.Tensor) -> torch.Tensor:
        params = vector_to_named_parameters(model, vector, selected_names)
        return functional_call(model, (params, buffers), (x_batch,))

    train_batch_size = config.pac_bayes_batch_size if config.pac_bayes_batch_size > 0 else config.batch_size
    train_batch_size = max(1, min(train_batch_size, max(n, 1)))
    if n and config.pac_bayes_train_epochs > 0:
        optimizer = torch.optim.Adam([posterior_mean], lr=config.pac_bayes_lr)
        for _ in range(config.pac_bayes_train_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, train_batch_size):
                idx = perm[start : start + train_batch_size]
                optimizer.zero_grad(set_to_none=True)
                sampled_vector = posterior_mean + config.pac_bayes_posterior_sigma * torch.randn_like(posterior_mean)
                logits = logits_from_vector(sampled_vector, x_t[idx])
                empirical_loss = F.cross_entropy(logits, y_t[idx])
                kl_t = torch_gaussian_kl_isotropic(
                    posterior_mean,
                    prior_selected,
                    config.pac_bayes_posterior_sigma,
                    config.pac_bayes_prior_sigma,
                )
                objective = empirical_loss + config.pac_bayes_kl_weight * kl_t / max(n, 1)
                objective.backward()
                optimizer.step()

    posterior_vector = posterior_mean.detach().cpu()
    kl = gaussian_kl_isotropic(
        posterior_vector,
        prior_selected.detach().cpu(),
        posterior_sigma=config.pac_bayes_posterior_sigma,
        prior_sigma=config.pac_bayes_prior_sigma,
    )

    empirical_errors: list[float] = []
    eval_batch_size = max(1, config.inference_batch_size)
    with torch.no_grad():
        for _ in range(config.pac_bayes_samples):
            sampled_vector = posterior_mean.detach() + config.pac_bayes_posterior_sigma * torch.randn_like(
                posterior_mean
            )
            correct = 0
            total = 0
            for start in range(0, n, eval_batch_size):
                logits = logits_from_vector(sampled_vector, x_t[start : start + eval_batch_size])
                pred = logits.argmax(dim=1)
                target = y_t[start : start + eval_batch_size]
                correct += int((pred == target).sum().item())
                total += len(target)
            empirical_errors.append(1.0 - correct / max(total, 1))

    empirical_risk = float(np.mean(empirical_errors)) if empirical_errors else 1.0
    bound, mc_upper = pac_bayes_bound(
        empirical_risk,
        kl,
        len(split.pool.y),
        config.pac_bayes_delta,
        config.pac_bayes_samples,
        config.pac_bayes_delta_test,
    )
    return {
        "pac_bayes_bound": bound,
        "pac_bayes_empirical_risk": empirical_risk,
        "pac_bayes_mc_upper": mc_upper,
        "pac_bayes_kl": kl,
    }


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
    model = make_run_model(
        stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), split, config, device
    )

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
            optimizer_name=config.optimizer,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    prior_vector = parameter_vector(model, pac_bayes_parameter_names(model, config.pac_bayes_scope))

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    support_set = set(support)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y))
    stop_reached = False
    remaining_bad = len(split.pool.y)
    train_calls = 0

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, *split.pool.x.shape[1:]), dtype=np.float32)
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
        stable_seed(seed, f"{method}-pac-bayes-final", int(pretrain_fraction * 10_000)),
    )
    diagnostics = selected_set_diagnostics(model, split.pool, support, device, config.inference_batch_size, config.score)
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
        "test_inappropriate_risk": test_inappropriate_risk,
        **pac_stats,
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
    model = make_run_model(
        stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), split, config, device
    )

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
            optimizer_name=config.optimizer,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    prior_vector = parameter_vector(model, pac_bayes_parameter_names(model, config.pac_bayes_scope))

    support = deterministic_initial_support(split.pool, config.initial_per_class, seed)
    initial_support_size = len(support)
    support_set = set(support)
    limit = min(max(config.max_total_support, len(support)), len(split.pool.y))
    train_calls = 0
    rows: list[dict[str, Any]] = []

    if method == "GREATS":
        probe_x, probe_y = deterministic_probe(split.pool, split, config.greats_probe_size, seed)
    else:
        probe_x = np.empty((0, *split.pool.x.shape[1:]), dtype=np.float32)
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
        should_record = step == 0 or step % record_every == 0 or stop_reached or hit_limit
        if should_record:
            effective_size = len(support) if stop_reached else len(support) + remaining_bad
            bound = p2l_bound(effective_size, len(split.pool.y), config.delta) if method in CERTIFIED_METHODS else None
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
                stable_seed(seed, f"{method}-pac-bayes-trace", step + int(pretrain_fraction * 10_000)),
            )
            diagnostics = selected_set_diagnostics(
                model, split.pool, support, device, config.inference_batch_size, config.score
            )
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
                    "test_inappropriate_risk": test_inappropriate_risk,
                    **pac_stats,
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
    model = make_run_model(
        stable_seed(seed, f"{method}-model", int(pretrain_fraction * 10_000)), split, config, device
    )

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
            optimizer_name=config.optimizer,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    prior_vector = parameter_vector(model, pac_bayes_parameter_names(model, config.pac_bayes_scope))

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
        probe_x = np.empty((0, *split.pool.x.shape[1:]), dtype=np.float32)
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
            stable_seed(seed, f"{method}-pac-bayes-budget", budget + int(pretrain_fraction * 10_000)),
        )
        diagnostics = selected_set_diagnostics(model, split.pool, support, device, config.inference_batch_size, config.score)
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
                "test_inappropriate_risk": test_inappropriate_risk,
                **pac_stats,
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
        chosen = choose_next(method, model, split.pool, support, bad_candidates, bad_losses, config, device, probe_x, probe_y)
        support.append(chosen)
        support_set.add(chosen)

    return rows


def selected_set_diagnostics(
    model: nn.Module,
    pool: CertPool,
    selected: list[int],
    device: torch.device,
    batch_size: int,
    score_config: ScoreConfig,
) -> dict[str, float]:
    if not selected:
        return {
            "noise_hit_rate": 0.0,
            "duplicate_hit_rate": 0.0,
            "pairwise_feature_cosine": 0.0,
            "mean_support_redundancy": 0.0,
            "max_support_redundancy": 0.0,
            "mean_selected_residual_novelty": 0.0,
            "local_redundancy_hit_rate": 0.0,
            "residual_redundancy_hit_rate": 0.0,
            "strong_redundancy_hit_rate": 0.0,
            "mode_entropy": 0.0,
            "minority_mode_fraction": 0.0,
            "spectral_entropy": 0.0,
            "dynamic_mu": score_config.mu * (1.0 + max(float(score_config.alpha), 0.0)),
        }
    selected_arr = np.asarray(selected, dtype=np.int64)
    stats = model_stats(model, pool.x[selected_arr], pool.y[selected_arr], device, batch_size)
    if len(selected_arr) < 2:
        pairwise_feature_cosine = 0.0
        mean_support_redundancy = 0.0
        max_support_redundancy = 0.0
    else:
        sim = cosine_matrix(stats.embeddings, stats.embeddings)
        upper = sim[np.triu_indices_from(sim, k=1)]
        positive_upper = np.maximum(upper, 0.0)
        pairwise_feature_cosine = float(np.mean(upper))
        mean_support_redundancy = float(np.mean(positive_upper))
        max_support_redundancy = float(np.max(positive_upper))

    selected_unit = row_normed(stats.embeddings)
    local_hits: list[float] = []
    residual_hits: list[float] = []
    strong_hits: list[float] = []
    residual_values: list[float] = []
    for pos in range(len(selected_arr)):
        if pos == 0:
            continue
        prior = selected_unit[:pos]
        current = selected_unit[pos : pos + 1]
        local_redundancy = float(np.max(np.maximum(current @ prior.T, 0.0)))
        basis = support_span_basis(stats.embeddings[:pos], score_config.residual_rank, score_config.residual_tol)
        if basis.shape[1] == 0:
            residual_novelty = 1.0
        else:
            projection_sq = float(np.sum((current @ basis) ** 2))
            residual_novelty = float(np.clip(1.0 - projection_sq, 0.0, 1.0))
        residual_values.append(residual_novelty)
        local_hit = float(local_redundancy >= 0.90)
        residual_hit = float(residual_novelty <= 0.10)
        local_hits.append(local_hit)
        residual_hits.append(residual_hit)
        strong_hits.append(float(local_hit and residual_hit))

    group_ids = pool.group_id[selected_arr]
    valid_groups = group_ids[group_ids >= 0]
    if len(valid_groups):
        _, counts = np.unique(valid_groups, return_counts=True)
        probs = counts.astype(np.float64) / np.sum(counts)
        mode_entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / max(np.log(len(probs)), 1e-12))
        minority_mode_fraction = float(np.min(probs))
    else:
        mode_entropy = 0.0
        minority_mode_fraction = 0.0

    spectral_entropy = normalized_spectral_entropy(stats.embeddings, score_config.residual_tol)
    dynamic_mu = score_config.mu * (1.0 + max(float(score_config.alpha), 0.0) * (1.0 - spectral_entropy))
    return {
        "noise_hit_rate": float(np.mean(pool.is_noisy[selected_arr])),
        "duplicate_hit_rate": float(np.mean(pool.is_duplicate[selected_arr])),
        "pairwise_feature_cosine": pairwise_feature_cosine,
        "mean_support_redundancy": mean_support_redundancy,
        "max_support_redundancy": max_support_redundancy,
        "mean_selected_residual_novelty": float(np.mean(residual_values)) if residual_values else 0.0,
        "local_redundancy_hit_rate": float(np.mean(local_hits)) if local_hits else 0.0,
        "residual_redundancy_hit_rate": float(np.mean(residual_hits)) if residual_hits else 0.0,
        "strong_redundancy_hit_rate": float(np.mean(strong_hits)) if strong_hits else 0.0,
        "mode_entropy": mode_entropy,
        "minority_mode_fraction": minority_mode_fraction,
        "spectral_entropy": spectral_entropy,
        "dynamic_mu": dynamic_mu,
    }


def choose_next(
    method: str,
    model: nn.Module,
    pool: CertPool,
    support: list[int],
    candidate: np.ndarray,
    candidate_losses: np.ndarray,
    config: RunConfig,
    device: torch.device,
    probe_x: np.ndarray,
    probe_y: np.ndarray,
) -> int:
    if method == "MaxLoss":
        return tie_break_argmax(candidate, candidate_losses, pool.sample_id)
    if method != "Marginal" and not support:
        return tie_break_argmax(candidate, candidate_losses, pool.sample_id)

    cand_stats = model_stats(model, pool.x[candidate], pool.y[candidate], device, config.inference_batch_size)

    if method == "Marginal":
        scores = score_marginal(cand_stats)
        return tie_break_argmax(candidate, scores, pool.sample_id)

    support_arr = np.asarray(support, dtype=np.int64)
    support_stats = model_stats(model, pool.x[support_arr], pool.y[support_arr], device, config.inference_batch_size)

    if method == "PU-R":
        scores = score_pu_r(candidate_losses, support_stats, cand_stats, config.score)
    elif method == "PU-R-Vol":
        scores = score_pu_r_vol(candidate_losses, support_stats, cand_stats, config.score)
    elif method == "PU-R-Manifold":
        scores = score_pu_r_manifold(candidate_losses, support_stats, cand_stats, config.score)
    elif method in ABLATION_METHODS:
        scores = score_ablation(method, candidate_losses, support_stats, cand_stats, config.score)
    elif method == "GREATS":
        probe_stats = model_stats(model, probe_x, probe_y, device, config.inference_batch_size)
        scores = score_greats_reference(cand_stats, probe_stats, support_stats, config.score.lambda_redundancy)
    else:
        raise ValueError(f"Unknown method: {method}")
    return tie_break_argmax(candidate, scores, pool.sample_id)
