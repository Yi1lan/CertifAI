from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ModelStats


@dataclass(frozen=True)
class ScoreConfig:
    gamma: float
    c_loss: float
    alpha: float
    mu: float
    lambda_redundancy: float
    global_redundancy_weight: float
    residual_rank: int
    residual_tol: float
    manifold_k: int
    manifold_tau: float
    manifold_eigenvectors: int


def row_normed(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    return row_normed(left) @ row_normed(right).T


def last_layer_gradient_cosine(
    cand_emb: np.ndarray,
    cand_err: np.ndarray,
    pivot_emb: np.ndarray,
    pivot_err: np.ndarray,
) -> np.ndarray:
    return cosine_matrix(cand_emb, pivot_emb) * cosine_matrix(cand_err, pivot_err)


def tie_break_argmax(candidate: np.ndarray, scores: np.ndarray, sample_id: np.ndarray) -> int:
    local_ids = sample_id[candidate]
    order = np.lexsort((local_ids, -scores))
    return int(candidate[int(order[0])])


def support_span_basis(support_embeddings: np.ndarray, rank: int, tol: float) -> np.ndarray:
    if len(support_embeddings) == 0:
        return np.empty((support_embeddings.shape[1], 0), dtype=np.float64)
    support_unit = row_normed(support_embeddings)
    _, singular_values, vh = np.linalg.svd(support_unit, full_matrices=False)
    keep = singular_values > max(float(tol), 0.0)
    if rank > 0:
        rank_count = min(int(rank), int(np.sum(keep)))
    else:
        rank_count = int(np.sum(keep))
    if rank_count <= 0:
        return np.empty((support_embeddings.shape[1], 0), dtype=np.float64)
    return vh[:rank_count].T


def residual_score_terms(
    candidate_losses: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    config: ScoreConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_loss = np.clip(candidate_losses / config.gamma, 0.0, config.c_loss)
    cand_unit = row_normed(cand_stats.embeddings)
    support_unit = row_normed(support_stats.embeddings)
    if len(support_unit) == 0:
        local_redundancy = np.zeros(len(candidate_losses), dtype=np.float64)
    else:
        local_redundancy = np.max(np.maximum(cand_unit @ support_unit.T, 0.0), axis=1)

    basis = support_span_basis(support_stats.embeddings, config.residual_rank, config.residual_tol)
    if basis.shape[1] == 0:
        residual_novelty = np.ones(len(candidate_losses), dtype=np.float64)
    else:
        projection_sq = np.sum((cand_unit @ basis) ** 2, axis=1)
        residual_novelty = np.clip(1.0 - projection_sq, 0.0, 1.0)

    return clipped_loss, residual_novelty, local_redundancy


def score_pu_r(
    candidate_losses: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    config: ScoreConfig,
) -> np.ndarray:
    clipped_loss, residual_novelty, local_redundancy = residual_score_terms(
        candidate_losses, support_stats, cand_stats, config
    )
    return clipped_loss + config.mu * residual_novelty - config.global_redundancy_weight * local_redundancy


def normalized_spectral_entropy(support_embeddings: np.ndarray, tol: float) -> float:
    if len(support_embeddings) <= 1:
        return 0.0
    singular_values = np.linalg.svd(row_normed(support_embeddings), compute_uv=False)
    eigenvalues = singular_values**2
    eigenvalues = eigenvalues[eigenvalues > max(float(tol), 0.0)]
    if len(eigenvalues) <= 1:
        return 0.0
    probs = eigenvalues / np.sum(eigenvalues)
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(np.clip(entropy / np.log(len(eigenvalues)), 0.0, 1.0))


def score_pu_r_vol(
    candidate_losses: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    config: ScoreConfig,
) -> np.ndarray:
    clipped_loss, residual_novelty, local_redundancy = residual_score_terms(
        candidate_losses, support_stats, cand_stats, config
    )
    spectral_entropy = normalized_spectral_entropy(support_stats.embeddings, config.residual_tol)
    concentration = max(float(config.alpha), 0.0) * (1.0 - spectral_entropy)
    dynamic_mu = config.mu * (1.0 + concentration)
    dynamic_redundancy = config.global_redundancy_weight * (1.0 + concentration)
    return clipped_loss + dynamic_mu * residual_novelty - dynamic_redundancy * local_redundancy


def cosine_distance(left_unit: np.ndarray, right_unit: np.ndarray) -> np.ndarray:
    cosine = np.clip(left_unit @ right_unit.T, -1.0, 1.0)
    return np.sqrt(np.maximum(2.0 - 2.0 * cosine, 0.0))


def knn_affinity_from_distances(distances: np.ndarray, k: int, tau: float) -> np.ndarray:
    if distances.size == 0:
        return np.zeros_like(distances, dtype=np.float64)
    k = min(max(int(k), 1), distances.shape[1])
    tau = max(float(tau), 1e-12)
    affinity = np.zeros_like(distances, dtype=np.float64)
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(distances.shape[0])[:, None]
    affinity[rows, nearest] = np.exp(-distances[rows, nearest] / tau)
    return affinity


def support_graph_laplacian(
    support_unit: np.ndarray,
    k: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    support_count = len(support_unit)
    if support_count <= 1:
        return np.zeros((support_count,), dtype=np.float64), np.eye(support_count, dtype=np.float64)

    distances = cosine_distance(support_unit, support_unit)
    np.fill_diagonal(distances, np.inf)
    weights = knn_affinity_from_distances(distances, min(k, support_count - 1), tau)
    weights = np.maximum(weights, weights.T)
    degrees = np.sum(weights, axis=1)
    inv_sqrt_degree = np.zeros_like(degrees)
    positive = degrees > 1e-12
    inv_sqrt_degree[positive] = 1.0 / np.sqrt(degrees[positive])
    normalized_adjacency = inv_sqrt_degree[:, None] * weights * inv_sqrt_degree[None, :]
    laplacian = np.eye(support_count, dtype=np.float64) - normalized_adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def score_pu_r_manifold(
    candidate_losses: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    config: ScoreConfig,
) -> np.ndarray:
    clipped_loss, residual_novelty, _ = residual_score_terms(candidate_losses, support_stats, cand_stats, config)
    if len(support_stats.embeddings) <= 2:
        return score_pu_r(candidate_losses, support_stats, cand_stats, config)

    support_unit = row_normed(support_stats.embeddings)
    cand_unit = row_normed(cand_stats.embeddings)
    support_count = len(support_unit)
    k = min(max(int(config.manifold_k), 1), support_count)
    tau = max(float(config.manifold_tau), 1e-12)

    candidate_support_dist = cosine_distance(cand_unit, support_unit)
    direct_affinity = knn_affinity_from_distances(candidate_support_dist, k, tau)
    affinity_sum = np.sum(direct_affinity, axis=1, keepdims=True)
    normalized_affinity = direct_affinity / np.maximum(affinity_sum, 1e-12)

    eigenvalues, eigenvectors = support_graph_laplacian(support_unit, k, tau)
    diffusion = eigenvectors @ np.diag(np.exp(-eigenvalues / tau)) @ eigenvectors.T
    diffusion = np.maximum(diffusion, 0.0)
    diffusion = diffusion / np.maximum(np.max(diffusion, axis=1, keepdims=True), 1e-12)
    geodesic_similarity = normalized_affinity @ diffusion
    geodesic_redundancy = np.max(geodesic_similarity, axis=1)

    nontrivial = np.flatnonzero(eigenvalues > max(config.residual_tol, 0.0))
    eig_count = min(max(int(config.manifold_eigenvectors), 1), len(nontrivial))
    if eig_count == 0:
        geodesic_residual_novelty = residual_novelty
    else:
        smooth_basis = eigenvectors[:, nontrivial[:eig_count]]
        graph_coords = normalized_affinity @ smooth_basis
        smooth_coverage = np.sum(graph_coords**2, axis=1) * support_count / eig_count
        smooth_coverage = np.clip(smooth_coverage, 0.0, 1.0)
        geodesic_residual_novelty = np.clip(residual_novelty * (1.0 - smooth_coverage), 0.0, 1.0)

    return (
        clipped_loss
        + config.mu * geodesic_residual_novelty
        - config.global_redundancy_weight * geodesic_redundancy
    )


def score_ablation(
    method: str,
    candidate_losses: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    config: ScoreConfig,
) -> np.ndarray:
    clipped_loss, residual_novelty, local_redundancy = residual_score_terms(
        candidate_losses, support_stats, cand_stats, config
    )
    margin_score = -cand_stats.margins
    if method == "ClippedLoss":
        return clipped_loss
    if method == "ResidualOnly":
        return residual_novelty
    if method == "RedundancyOnly":
        return -local_redundancy
    if method == "Loss+Residual":
        return clipped_loss + config.mu * residual_novelty
    if method == "Loss-Redundancy":
        return clipped_loss - config.global_redundancy_weight * local_redundancy
    if method == "PU-C-style":
        local_novelty = 1.0 - local_redundancy
        return clipped_loss + config.mu * local_novelty - config.global_redundancy_weight * local_redundancy
    if method == "Marginal+Residual":
        return margin_score + config.mu * residual_novelty
    if method == "Marginal-Redundancy":
        return margin_score - config.global_redundancy_weight * local_redundancy
    if method == "Marginal+Residual-Redundancy":
        return margin_score + config.mu * residual_novelty - config.global_redundancy_weight * local_redundancy
    raise ValueError(f"Unknown ablation method: {method}")


def score_marginal(cand_stats: ModelStats) -> np.ndarray:
    return -cand_stats.margins


def score_el2n(cand_stats: ModelStats) -> np.ndarray:
    return np.linalg.norm(cand_stats.errors, axis=1)


def score_grand_last(cand_stats: ModelStats) -> np.ndarray:
    embedding_norm = np.linalg.norm(cand_stats.embeddings, axis=1)
    error_norm = np.linalg.norm(cand_stats.errors, axis=1)
    return embedding_norm * error_norm


def score_rho_pretrain_ref(candidate_losses: np.ndarray, reference_losses: np.ndarray) -> np.ndarray:
    return np.maximum(candidate_losses - reference_losses, 0.0)


def score_greats_reference(
    cand_stats: ModelStats,
    probe_stats: ModelStats,
    support_stats: ModelStats | None,
    lambda_redundancy: float,
) -> np.ndarray:
    utility = np.mean(
        last_layer_gradient_cosine(
            cand_stats.embeddings,
            cand_stats.errors,
            probe_stats.embeddings,
            probe_stats.errors,
        ),
        axis=1,
    )
    if support_stats is None or len(support_stats.losses) == 0:
        return utility
    redundancy = np.max(
        np.maximum(
            last_layer_gradient_cosine(
                cand_stats.embeddings,
                cand_stats.errors,
                support_stats.embeddings,
                support_stats.errors,
            ),
            0.0,
        ),
        axis=1,
    )
    return utility - lambda_redundancy * redundancy
