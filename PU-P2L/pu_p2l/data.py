from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NO_PAC_BAYES_DATASETS = {"synthetic_redundancy_hard", "synthetic_redundacy_hard"}


def pac_bayes_enabled_for_dataset(dataset_name: str) -> bool:
    return dataset_name.strip() not in NO_PAC_BAYES_DATASETS


def canonical_dataset_name(dataset_name: str) -> str:
    name = dataset_name.strip().lower().replace("-", "_")
    if name in {"mnist", "binary_mnist", "binarymnist"}:
        return "mnist"
    if name in {"mnist10", "mnist_10", "mnist_multiclass", "mnist10_class"}:
        return "mnist10"
    if name in {"fashion", "fashion_mnist", "fashionmnist", "fmnist"}:
        return "fashion_mnist"
    if name in {"rotated_mnist", "rotatedmnist"}:
        return "rotated_mnist"
    if name in {"rotated_fashion_mnist", "rotated_fashion", "rotated_fmnist"}:
        return "rotated_fashion_mnist"
    if name in {"mode_mnist", "mode_imbalanced_mnist", "mode_mnist_3459"}:
        return "mode_mnist"
    if name in {"boundary_duplicate_mnist", "boundary_aug_mnist", "duplicate_boundary_mnist"}:
        return "boundary_duplicate_mnist"
    if name in {"two_moons", "twomoons", "synthetic_two_moons"}:
        return "two_moons"
    if name in {"cifar", "cifar10", "cifar_10", "cifar10_reduced"}:
        return "cifar10"
    if name in {"synthetic_redundancy_hard", "synthetic_redundacy_hard"}:
        return "synthetic_redundancy_hard"
    return name


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
    classes = sorted(int(cls) for cls in np.unique(y).tolist())
    if not classes:
        return np.array([], dtype=np.int64)
    base_per_class = count // len(classes)
    extra = count % len(classes)
    for class_pos, cls in enumerate(classes):
        cls_idx = np.where(y == cls)[0]
        cls_count = min(len(cls_idx), base_per_class + int(class_pos < extra))
        if cls_count:
            chosen.extend(rng.choice(cls_idx, size=cls_count, replace=False).tolist())
            remaining -= cls_count
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        chosen.extend(rng.choice(available, size=min(remaining, len(available)), replace=False).tolist())
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def apply_label_noise(
    y: np.ndarray,
    num_classes: int,
    noise_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_noisy = y.astype(np.int64).copy()
    is_noisy = rng.random(len(y_noisy)) < float(noise_rate)
    if np.any(is_noisy):
        if num_classes == 2:
            y_noisy[is_noisy] = 1 - y_noisy[is_noisy]
        else:
            offsets = rng.integers(1, num_classes, size=int(np.sum(is_noisy)))
            y_noisy[is_noisy] = (y_noisy[is_noisy] + offsets) % num_classes
    return y_noisy, is_noisy


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
MNIST_URLS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)


def _require_torchvision():
    try:
        from torchvision import datasets
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "MNIST/CIFAR experiments require torchvision. Install it in the Conda environment first."
        ) from exc
    return datasets


def _download_mnist_file(raw_dir: Path, file_name: str, download: bool) -> Path:
    path = raw_dir / file_name
    if path.exists():
        return path
    if not download:
        raise RuntimeError(
            f"MNIST file {path} is missing. Re-run with --download-data or install torchvision."
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for base_url in MNIST_URLS:
        url = base_url + file_name
        try:
            with urllib.request.urlopen(url, timeout=30) as response, path.open("wb") as handle:
                handle.write(response.read())
            return path
        except OSError as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download MNIST IDX files: " + "; ".join(errors))


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise RuntimeError(f"Unexpected MNIST image magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise RuntimeError(f"Unexpected MNIST label magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count).astype(np.int64)


def _load_mnist_arrays(
    data_dir: str,
    download: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from torchvision import datasets  # type: ignore

        train = datasets.MNIST(root=data_dir, train=True, download=download)
        test = datasets.MNIST(root=data_dir, train=False, download=download)
        return (
            train.data.numpy().astype(np.float32),
            np.asarray(train.targets, dtype=np.int64),
            test.data.numpy().astype(np.float32),
            np.asarray(test.targets, dtype=np.int64),
        )
    except ImportError:
        raw_dir = Path(data_dir) / "MNIST" / "raw"
        paths = {
            key: _download_mnist_file(raw_dir, file_name, download)
            for key, file_name in MNIST_FILES.items()
        }
        return (
            _read_idx_images(paths["train_images"]).astype(np.float32),
            _read_idx_labels(paths["train_labels"]),
            _read_idx_images(paths["test_images"]).astype(np.float32),
            _read_idx_labels(paths["test_labels"]),
        )


def make_mnist_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    data_dir: str,
    download: bool,
) -> DatasetBundle:
    x_train_all, y_train_digits, x_test_all, y_test_digits = _load_mnist_arrays(data_dir, download)
    y_train_all = (y_train_digits > 4).astype(np.int64)
    y_test_all = (y_test_digits > 4).astype(np.int64)

    train_count = min(int(n_train), len(y_train_all))
    test_count = min(int(n_test), len(y_test_all))
    train_idx = stratified_indices(y_train_all, train_count, stable_seed(seed, "mnist-train"))
    test_idx = stratified_indices(y_test_all, test_count, stable_seed(seed, "mnist-test"))

    x_train = ((x_train_all[train_idx] / 255.0 - 0.1307) / 0.3081)[:, None, :, :].astype(np.float32)
    true_y_train = y_train_all[train_idx].astype(np.int64)
    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 2, noise_rate, stable_seed(seed, "mnist-label-noise", int(noise_rate * 10_000))
    )
    x_test = ((x_test_all[test_idx] / 255.0 - 0.1307) / 0.3081)[:, None, :, :].astype(np.float32)
    y_test = y_test_all[test_idx].astype(np.int64)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=np.full(len(y_train), -1, dtype=np.int64),
        is_duplicate_train=np.zeros(len(y_train), dtype=bool),
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def make_mnist10_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    data_dir: str,
    download: bool,
) -> DatasetBundle:
    x_train_all, y_train_all, x_test_all, y_test_all = _load_mnist_arrays(data_dir, download)

    train_count = min(int(n_train), len(y_train_all))
    test_count = min(int(n_test), len(y_test_all))
    train_idx = stratified_indices(y_train_all, train_count, stable_seed(seed, "mnist10-train"))
    test_idx = stratified_indices(y_test_all, test_count, stable_seed(seed, "mnist10-test"))

    x_train = ((x_train_all[train_idx] / 255.0 - 0.1307) / 0.3081)[:, None, :, :].astype(np.float32)
    true_y_train = y_train_all[train_idx].astype(np.int64)
    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 10, noise_rate, stable_seed(seed, "mnist10-label-noise", int(noise_rate * 10_000))
    )
    x_test = ((x_test_all[test_idx] / 255.0 - 0.1307) / 0.3081)[:, None, :, :].astype(np.float32)
    y_test = y_test_all[test_idx].astype(np.int64)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=np.full(len(y_train), -1, dtype=np.int64),
        is_duplicate_train=np.zeros(len(y_train), dtype=bool),
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def _load_fashion_mnist_arrays(
    data_dir: str,
    download: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    datasets = _require_torchvision()
    train = datasets.FashionMNIST(root=data_dir, train=True, download=download)
    test = datasets.FashionMNIST(root=data_dir, train=False, download=download)
    return (
        train.data.numpy().astype(np.float32),
        np.asarray(train.targets, dtype=np.int64),
        test.data.numpy().astype(np.float32),
        np.asarray(test.targets, dtype=np.int64),
    )


def normalize_gray_images(images: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((images / 255.0 - mean) / std)[:, None, :, :].astype(np.float32)


def make_fashion_mnist_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    data_dir: str,
    download: bool,
) -> DatasetBundle:
    x_train_all, y_train_all, x_test_all, y_test_all = _load_fashion_mnist_arrays(data_dir, download)

    train_count = min(int(n_train), len(y_train_all))
    test_count = min(int(n_test), len(y_test_all))
    train_idx = stratified_indices(y_train_all, train_count, stable_seed(seed, "fashion-train"))
    test_idx = stratified_indices(y_test_all, test_count, stable_seed(seed, "fashion-test"))

    x_train = normalize_gray_images(x_train_all[train_idx], 0.2860, 0.3530)
    true_y_train = y_train_all[train_idx].astype(np.int64)
    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 10, noise_rate, stable_seed(seed, "fashion-label-noise", int(noise_rate * 10_000))
    )
    x_test = normalize_gray_images(x_test_all[test_idx], 0.2860, 0.3530)
    y_test = y_test_all[test_idx].astype(np.int64)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=np.full(len(y_train), -1, dtype=np.int64),
        is_duplicate_train=np.zeros(len(y_train), dtype=bool),
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def rotate_images_uint8(images: np.ndarray, angles: np.ndarray) -> np.ndarray:
    try:
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as tvf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Rotated image datasets require pillow and torchvision.") from exc

    out = np.empty_like(images)
    for idx, angle in enumerate(angles):
        image = Image.fromarray(images[idx].astype(np.uint8), mode="L")
        rotated = tvf.rotate(
            image,
            float(angle),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        out[idx] = np.asarray(rotated, dtype=np.uint8)
    return out


def fixed_rotation_angles(count: int, angles: list[float], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not angles:
        angles = [-60.0, -30.0, 0.0, 30.0, 60.0]
    rng = np.random.default_rng(seed)
    group_ids = np.arange(count, dtype=np.int64) % len(angles)
    group_ids = rng.permutation(group_ids)
    angle_arr = np.asarray([angles[int(group)] for group in group_ids], dtype=np.float32)
    return angle_arr, group_ids.astype(np.int64)


def make_rotated_gray_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    data_dir: str,
    download: bool,
    fashion: bool,
    rotation_angles: list[float],
) -> DatasetBundle:
    if fashion:
        x_train_all, y_train_all, x_test_all, y_test_all = _load_fashion_mnist_arrays(data_dir, download)
        mean, std = 0.2860, 0.3530
        train_name, test_name, noise_name = "rotated-fashion-train", "rotated-fashion-test", "rotated-fashion-noise"
    else:
        x_train_all, y_train_all, x_test_all, y_test_all = _load_mnist_arrays(data_dir, download)
        mean, std = 0.1307, 0.3081
        train_name, test_name, noise_name = "rotated-mnist-train", "rotated-mnist-test", "rotated-mnist-noise"

    train_count = min(int(n_train), len(y_train_all))
    test_count = min(int(n_test), len(y_test_all))
    train_idx = stratified_indices(y_train_all, train_count, stable_seed(seed, train_name))
    test_idx = stratified_indices(y_test_all, test_count, stable_seed(seed, test_name))
    train_angles, train_groups = fixed_rotation_angles(
        len(train_idx), rotation_angles, stable_seed(seed, train_name + "-angles")
    )
    test_angles, _ = fixed_rotation_angles(
        len(test_idx), rotation_angles, stable_seed(seed, test_name + "-angles")
    )

    x_train = normalize_gray_images(rotate_images_uint8(x_train_all[train_idx], train_angles), mean, std)
    true_y_train = y_train_all[train_idx].astype(np.int64)
    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 10, noise_rate, stable_seed(seed, noise_name, int(noise_rate * 10_000))
    )
    x_test = normalize_gray_images(rotate_images_uint8(x_test_all[test_idx], test_angles), mean, std)
    y_test = y_test_all[test_idx].astype(np.int64)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=train_groups,
        is_duplicate_train=np.zeros(len(y_train), dtype=bool),
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def sample_mode_mnist_indices(
    y_digits: np.ndarray,
    count: int,
    mode_imbalance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    modes = {
        0: (3, 5),
        1: (4, 9),
    }
    mode_imbalance = float(np.clip(mode_imbalance, 0.0, 1.0))
    mode_counts = {
        0: int(round(count * mode_imbalance)),
        1: count - int(round(count * mode_imbalance)),
    }
    chosen: list[int] = []
    group_ids: list[int] = []
    for mode, digits in modes.items():
        per_digit = mode_counts[mode] // 2
        extra = mode_counts[mode] % 2
        for pos, digit in enumerate(digits):
            digit_idx = np.where(y_digits == digit)[0]
            take = min(len(digit_idx), per_digit + int(pos < extra))
            if take:
                selected = rng.choice(digit_idx, size=take, replace=False).tolist()
                chosen.extend(selected)
                group_ids.extend([mode] * len(selected))
    order = rng.permutation(len(chosen))
    return np.asarray(chosen, dtype=np.int64)[order], np.asarray(group_ids, dtype=np.int64)[order]


def make_mode_mnist_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    data_dir: str,
    download: bool,
    mode_imbalance: float,
    boundary_augmentation: int = 1,
) -> DatasetBundle:
    x_train_all, y_train_digits, x_test_all, y_test_digits = _load_mnist_arrays(data_dir, download)
    train_idx, group_id_train = sample_mode_mnist_indices(
        y_train_digits, min(int(n_train), len(y_train_digits)), mode_imbalance, stable_seed(seed, "mode-mnist-train")
    )
    test_idx, _ = sample_mode_mnist_indices(
        y_test_digits, min(int(n_test), len(y_test_digits)), 0.5, stable_seed(seed, "mode-mnist-test")
    )
    train_digits = y_train_digits[train_idx]
    test_digits = y_test_digits[test_idx]
    true_y_train = np.isin(train_digits, [5, 9]).astype(np.int64)
    y_test = np.isin(test_digits, [5, 9]).astype(np.int64)
    is_duplicate_train = np.zeros(len(train_idx), dtype=bool)

    images = x_train_all[train_idx].copy()
    boundary_augmentation = max(1, int(boundary_augmentation))
    if boundary_augmentation > 1:
        mode_a = group_id_train == 0
        rng = np.random.default_rng(stable_seed(seed, "boundary-augmentation", boundary_augmentation))
        mode_a_positions = np.flatnonzero(mode_a)
        source_count = max(1, int(np.ceil(len(mode_a_positions) / boundary_augmentation)))
        source_positions = rng.choice(mode_a_positions, size=source_count, replace=False)
        repeated_sources = rng.choice(source_positions, size=len(mode_a_positions), replace=True)
        angles = rng.uniform(-15.0, 15.0, size=len(mode_a_positions)).astype(np.float32)
        images[mode_a_positions] = rotate_images_uint8(x_train_all[train_idx[repeated_sources]], angles)
        is_duplicate_train[mode_a_positions] = True

    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 2, noise_rate, stable_seed(seed, "mode-mnist-label-noise", int(noise_rate * 10_000))
    )

    return DatasetBundle(
        x_train=normalize_gray_images(images, 0.1307, 0.3081),
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=group_id_train,
        is_duplicate_train=is_duplicate_train,
        is_noisy_train=is_noisy_train,
        x_test=normalize_gray_images(x_test_all[test_idx], 0.1307, 0.3081),
        y_test=y_test,
    )


def make_two_moons_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    cluster_std: float,
) -> DatasetBundle:
    def generate(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        half = count // 2
        rest = count - half
        theta0 = rng.uniform(0.0, np.pi, size=half)
        theta1 = rng.uniform(0.0, np.pi, size=rest)
        x0 = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
        x1 = np.stack([1.0 - np.cos(theta1), 0.5 - np.sin(theta1)], axis=1)
        x = np.vstack([x0, x1]).astype(np.float32)
        y = np.concatenate([np.zeros(half, dtype=np.int64), np.ones(rest, dtype=np.int64)])
        group = np.concatenate([(theta0 > np.pi / 2).astype(np.int64), 2 + (theta1 > np.pi / 2).astype(np.int64)])
        x += rng.normal(0.0, cluster_std, size=x.shape).astype(np.float32)
        perm = rng.permutation(count)
        return x[perm], y[perm], group[perm].astype(np.int64)

    rng = np.random.default_rng(seed)
    x_train, true_y_train, group_id_train = generate(int(n_train), rng)
    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 2, noise_rate, stable_seed(seed, "two-moons-label-noise", int(noise_rate * 10_000))
    )
    x_test, y_test, _ = generate(int(n_test), rng)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=group_id_train,
        is_duplicate_train=np.zeros(len(y_train), dtype=bool),
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def make_cifar10_dataset(
    seed: int,
    n_train: int,
    n_test: int,
    noise_rate: float,
    data_dir: str,
    download: bool,
) -> DatasetBundle:
    datasets = _require_torchvision()
    train = datasets.CIFAR10(root=data_dir, train=True, download=download)
    test = datasets.CIFAR10(root=data_dir, train=False, download=download)

    x_train_all = train.data.astype(np.float32)
    y_train_all = np.asarray(train.targets, dtype=np.int64)
    x_test_all = test.data.astype(np.float32)
    y_test_all = np.asarray(test.targets, dtype=np.int64)

    train_count = min(int(n_train), len(y_train_all))
    test_count = min(int(n_test), len(y_test_all))
    train_idx = stratified_indices(y_train_all, train_count, stable_seed(seed, "cifar10-train"))
    test_idx = stratified_indices(y_test_all, test_count, stable_seed(seed, "cifar10-test"))

    mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
    std = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32)[:, None, None]
    x_train = np.transpose(x_train_all[train_idx] / 255.0, (0, 3, 1, 2))
    x_train = ((x_train - mean) / std).astype(np.float32)
    true_y_train = y_train_all[train_idx].astype(np.int64)
    y_train, is_noisy_train = apply_label_noise(
        true_y_train, 10, noise_rate, stable_seed(seed, "cifar10-label-noise", int(noise_rate * 10_000))
    )
    x_test = np.transpose(x_test_all[test_idx] / 255.0, (0, 3, 1, 2))
    x_test = ((x_test - mean) / std).astype(np.float32)
    y_test = y_test_all[test_idx].astype(np.int64)

    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        true_y_train=true_y_train,
        group_id_train=np.full(len(y_train), -1, dtype=np.int64),
        is_duplicate_train=np.zeros(len(y_train), dtype=bool),
        is_noisy_train=is_noisy_train,
        x_test=x_test,
        y_test=y_test,
    )


def make_experiment_dataset(
    dataset_name: str,
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
    data_dir: str,
    download: bool,
    mode_imbalance: float = 0.85,
    boundary_augmentation: int = 1,
    rotation_angles: list[float] | None = None,
) -> DatasetBundle:
    name = canonical_dataset_name(dataset_name)
    if name == "synthetic_redundancy_hard":
        return make_redundancy_dataset(
            seed=seed,
            n_train=n_train,
            n_test=n_test,
            duplicate_groups=duplicate_groups,
            duplicates_per_group=duplicates_per_group,
            noise_rate=noise_rate,
            ambiguous_fraction=ambiguous_fraction,
            cluster_std=cluster_std,
            band_std=band_std,
            duplicate_std=duplicate_std,
        )
    if name == "mnist":
        return make_mnist_dataset(seed, n_train, n_test, noise_rate, data_dir, download)
    if name == "mnist10":
        return make_mnist10_dataset(seed, n_train, n_test, noise_rate, data_dir, download)
    if name == "fashion_mnist":
        return make_fashion_mnist_dataset(seed, n_train, n_test, noise_rate, data_dir, download)
    if name == "rotated_mnist":
        return make_rotated_gray_dataset(
            seed, n_train, n_test, noise_rate, data_dir, download, False, rotation_angles or []
        )
    if name == "rotated_fashion_mnist":
        return make_rotated_gray_dataset(
            seed, n_train, n_test, noise_rate, data_dir, download, True, rotation_angles or []
        )
    if name == "mode_mnist":
        return make_mode_mnist_dataset(
            seed, n_train, n_test, noise_rate, data_dir, download, mode_imbalance, boundary_augmentation=1
        )
    if name == "boundary_duplicate_mnist":
        return make_mode_mnist_dataset(
            seed,
            n_train,
            n_test,
            noise_rate,
            data_dir,
            download,
            mode_imbalance,
            boundary_augmentation=boundary_augmentation,
        )
    if name == "two_moons":
        return make_two_moons_dataset(seed, n_train, n_test, noise_rate, cluster_std)
    if name == "cifar10":
        return make_cifar10_dataset(seed, n_train, n_test, noise_rate, data_dir, download)
    raise ValueError(
        "Unknown dataset "
        f"'{dataset_name}'. Valid datasets: synthetic_redundancy_hard, mnist, mnist10, fashion_mnist, "
        "mode_mnist, boundary_duplicate_mnist, rotated_mnist, rotated_fashion_mnist, two_moons, cifar10."
    )


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
