from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    true_y_train: np.ndarray
    group_id_train: np.ndarray
    is_duplicate_train: np.ndarray
    is_noisy_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class CertPool:
    x: np.ndarray
    y: np.ndarray
    true_y: np.ndarray
    group_id: np.ndarray
    is_duplicate: np.ndarray
    is_noisy: np.ndarray
    sample_id: np.ndarray


@dataclass(frozen=True)
class SplitBundle:
    pool: CertPool
    x_pretrain: np.ndarray
    y_pretrain: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def stable_seed(base_seed: int, name: str, offset: int = 0) -> int:
    value = sum((idx + 1) * ord(char) for idx, char in enumerate(name))
    return int(base_seed * 100_000 + value + offset * 997)


def generate_clean_points(
    rng: np.random.Generator,
    n: int,
    ambiguous_fraction: float,
    cluster_std: float,
    band_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_band = int(round(n * ambiguous_fraction))
    n_cluster = n - n_band
    centers = {
        0: np.array([[-2.0, -1.1], [-2.0, 1.1]], dtype=np.float32),
        1: np.array([[2.0, -1.1], [2.0, 1.1]], dtype=np.float32),
    }

    labels = rng.integers(0, 2, size=n_cluster)
    x_cluster = np.zeros((n_cluster, 2), dtype=np.float32)
    for idx, label in enumerate(labels):
        center = centers[int(label)][rng.integers(0, 2)]
        x_cluster[idx] = center + rng.normal(0.0, cluster_std, size=2)

    x_band = np.zeros((n_band, 2), dtype=np.float32)
    if n_band:
        x_band[:, 0] = rng.normal(0.0, band_std, size=n_band)
        x_band[:, 1] = rng.uniform(-2.4, 2.4, size=n_band)
    y_band = (x_band[:, 0] > 0).astype(np.int64)

    x = np.vstack([x_cluster, x_band]).astype(np.float32)
    y = np.concatenate([labels.astype(np.int64), y_band])
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def make_redundancy_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    duplicate_groups: int,
    duplicates_per_group: int,
    noise_rate: float,
    ambiguous_fraction: float,
    cluster_std: float,
    band_std: float,
    duplicate_std: float,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_duplicate = duplicate_groups * duplicates_per_group
    if n_duplicate >= n_train:
        raise ValueError("duplicate_groups * duplicates_per_group must be smaller than n_train.")

    x_clean, y_clean = generate_clean_points(
        rng,
        n_train - n_duplicate,
        ambiguous_fraction=ambiguous_fraction,
        cluster_std=cluster_std,
        band_std=band_std,
    )

    noisy_group_count = int(round(noise_rate * duplicate_groups))
    noisy_groups = set(rng.choice(duplicate_groups, size=noisy_group_count, replace=False).tolist())

    x_dup = []
    true_dup = []
    observed_dup = []
    group_ids = []
    noisy_flags = []
    for group in range(duplicate_groups):
        true_label = int(rng.integers(0, 2))
        x_center = (-0.18 if true_label == 0 else 0.18) + rng.normal(0.0, 0.08)
        center = np.array([x_center, rng.uniform(-2.1, 2.1)], dtype=np.float32)
        observed_label = 1 - true_label if group in noisy_groups else true_label
        for _ in range(duplicates_per_group):
            x_dup.append(center + rng.normal(0.0, duplicate_std, size=2))
            true_dup.append(true_label)
            observed_dup.append(observed_label)
            group_ids.append(group)
            noisy_flags.append(group in noisy_groups)

    x_train = np.vstack([x_clean, np.asarray(x_dup, dtype=np.float32)]).astype(np.float32)
    y_train = np.concatenate([y_clean, np.asarray(observed_dup, dtype=np.int64)])
    true_y_train = np.concatenate([y_clean, np.asarray(true_dup, dtype=np.int64)])
    group_id_train = np.concatenate(
        [np.full(len(y_clean), -1, dtype=np.int64), np.asarray(group_ids, dtype=np.int64)]
    )
    is_duplicate_train = group_id_train >= 0
    is_noisy_train = np.concatenate([np.zeros(len(y_clean), dtype=bool), np.asarray(noisy_flags, dtype=bool)])

    perm = rng.permutation(n_train)
    x_train = x_train[perm]
    y_train = y_train[perm]
    true_y_train = true_y_train[perm]
    group_id_train = group_id_train[perm]
    is_duplicate_train = is_duplicate_train[perm]
    is_noisy_train = is_noisy_train[perm]

    x_test, y_test = generate_clean_points(
        rng,
        n_test,
        ambiguous_fraction=ambiguous_fraction,
        cluster_std=cluster_std,
        band_std=band_std,
    )

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=group_id_train,
        is_duplicate_train=is_duplicate_train,
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def stratified_indices(y: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    remaining = count
    for cls in [0, 1]:
        cls_idx = np.where(y == cls)[0]
        cls_count = min(len(cls_idx), count // 2)
        if cls_count:
            chosen.extend(rng.choice(cls_idx, size=cls_count, replace=False).tolist())
            remaining -= cls_count
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        chosen.extend(rng.choice(available, size=min(remaining, len(available)), replace=False).tolist())
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def make_pretrain_split(bundle: DatasetBundle, pretrain_fraction: float, seed: int) -> SplitBundle:
    n_pretrain = int(round(pretrain_fraction * len(bundle.y_train)))
    pretrain_idx = stratified_indices(bundle.y_train, n_pretrain, stable_seed(seed, "pretrain"))
    mask = np.ones(len(bundle.y_train), dtype=bool)
    mask[pretrain_idx] = False
    cert_idx = np.arange(len(bundle.y_train))[mask]
    pool = CertPool(
        x=bundle.x_train[cert_idx],
        y=bundle.y_train[cert_idx],
        true_y=bundle.true_y_train[cert_idx],
        group_id=bundle.group_id_train[cert_idx],
        is_duplicate=bundle.is_duplicate_train[cert_idx],
        is_noisy=bundle.is_noisy_train[cert_idx],
        sample_id=cert_idx.astype(np.int64),
    )
    return SplitBundle(
        pool=pool,
        x_pretrain=bundle.x_train[pretrain_idx],
        y_pretrain=bundle.y_train[pretrain_idx],
        x_test=bundle.x_test,
        y_test=bundle.y_test,
    )


def deterministic_initial_support(pool: CertPool, per_class: int, seed: int) -> list[int]:
    if per_class <= 0:
        return []
    selected: list[int] = []
    seed_term = stable_seed(seed, "initial-support") % 2_147_483_647
    keys = ((pool.sample_id.astype(np.int64) + 1) * 1_103_515_245 + seed_term) % 2_147_483_647
    for cls in sorted(set(int(label) for label in pool.y.tolist())):
        cls_idx = np.where(pool.y == cls)[0]
        order = np.lexsort((pool.sample_id[cls_idx], -keys[cls_idx]))
        selected.extend(int(idx) for idx in cls_idx[order[: min(per_class, len(order))]])
    return sorted(set(selected), key=lambda idx: int(pool.sample_id[idx]))

