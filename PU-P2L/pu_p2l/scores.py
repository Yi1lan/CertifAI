from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CertPool
from .model import ModelStats


@dataclass(frozen=True)
class ScoreConfig:
    gamma: float
    c_loss: float
    r_h: int
    r_consensus: int
    alpha: float
    mu: float
    lambda_redundancy: float
    global_redundancy_weight: float
    consensus_weight: float
    noise_penalty: float


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


def label_support_scores(
    candidate: np.ndarray,
    support_arr: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    pool: CertPool,
    config: ScoreConfig,
) -> tuple[np.ndarray, np.ndarray]:
    easy_count = min(max(config.r_consensus, 1), len(support_arr))
    easy_local = np.lexsort((pool.sample_id[support_arr], support_stats.losses))[:easy_count]
    easy_emb = support_stats.embeddings[easy_local]
    easy_labels = pool.y[support_arr[easy_local]]
    sim = np.maximum(cosine_matrix(cand_stats.embeddings, easy_emb), 0.0)
    same_scores = np.zeros(len(candidate), dtype=np.float64)
    opposite_scores = np.zeros(len(candidate), dtype=np.float64)
    for idx, label in enumerate(pool.y[candidate]):
        same = easy_labels == label
        opposite = ~same
        same_scores[idx] = float(np.max(sim[idx, same])) if np.any(same) else 0.0
        opposite_scores[idx] = float(np.max(sim[idx, opposite])) if np.any(opposite) else 0.0
    return same_scores, opposite_scores


def score_pu_c(
    candidate: np.ndarray,
    candidate_losses: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    config: ScoreConfig,
) -> np.ndarray:
    clipped_loss = np.clip(candidate_losses / config.gamma, 0.0, config.c_loss)
    all_feature_sim = cosine_matrix(cand_stats.embeddings, support_stats.embeddings)
    max_all_feature_sim = np.max(np.maximum(all_feature_sim, 0.0), axis=1)
    novelty = 1.0 - max_all_feature_sim
    return clipped_loss + config.mu * novelty - config.global_redundancy_weight * max_all_feature_sim


def score_marginal(cand_stats: ModelStats) -> np.ndarray:
    return -cand_stats.margins


def score_pu_f_or_g(
    candidate: np.ndarray,
    candidate_losses: np.ndarray,
    support_arr: np.ndarray,
    support_stats: ModelStats,
    cand_stats: ModelStats,
    pool: CertPool,
    config: ScoreConfig,
    use_noise_penalty: bool,
) -> np.ndarray:
    clipped_loss = np.clip(candidate_losses / config.gamma, 0.0, config.c_loss)
    all_feature_sim = cosine_matrix(cand_stats.embeddings, support_stats.embeddings)
    max_all_feature_sim = np.max(np.maximum(all_feature_sim, 0.0), axis=1)
    novelty = 1.0 - max_all_feature_sim

    hard_count = min(config.r_h, len(support_arr))
    hard_local = np.lexsort((pool.sample_id[support_arr], -support_stats.losses))[:hard_count]
    kll_hard = last_layer_gradient_cosine(
        cand_stats.embeddings,
        cand_stats.errors,
        support_stats.embeddings[hard_local],
        support_stats.errors[hard_local],
    )
    positive_hard = np.maximum(kll_hard, 0.0)
    hard_redundancy = np.max(positive_hard, axis=1) if positive_hard.shape[1] else 0.0

    same_score, opposite_score = label_support_scores(
        candidate, support_arr, support_stats, cand_stats, pool, config
    )
    consensus = same_score - opposite_score
    scores = (
        clipped_loss
        + config.consensus_weight * consensus
        + config.mu * novelty
        - config.global_redundancy_weight * max_all_feature_sim
        - config.lambda_redundancy * hard_redundancy
    )
    if use_noise_penalty:
        contradiction = np.maximum(opposite_score - same_score, 0.0)
        scores -= config.noise_penalty * clipped_loss * contradiction
    return scores


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
