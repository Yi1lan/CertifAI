# Paired Statistical Evidence

Target method: `PU-R-Manifold`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `GREATS`: mean diff -0.009367, 95% CI [-0.0158, -0.002934], win rate 0.61, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `MaxLoss`: mean diff -0.0109, 95% CI [-0.01715, -0.00466], win rate 0.62, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `PU-R`: mean diff 0.007443, 95% CI [0.0006236, 0.01426], win rate 0.39, n=135 (favors baseline).

## certified_bound

- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `MaxLoss`: mean diff -0.00983, 95% CI [-0.01662, -0.003042], win rate 0.57, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `PU-R`: mean diff 0.006779, 95% CI [-0.0005824, 0.01414], win rate 0.44, n=135 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `GREATS`: mean diff -8.867, 95% CI [-29.8, 12.06], win rate 0.48, n=135 (inconclusive).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `MaxLoss`: mean diff -35.1, 95% CI [-54.31, -15.88], win rate 0.57, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `PU-R`: mean diff 16.73, 95% CI [-6.61, 40.08], win rate 0.44, n=135 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `GREATS`: mean diff -0.03349, 95% CI [-0.04068, -0.02631], win rate 0.75, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `MaxLoss`: mean diff -0.03013, 95% CI [-0.03731, -0.02295], win rate 0.74, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5` vs `PU-R`: mean diff 0.01657, 95% CI [0.00859, 0.02456], win rate 0.29, n=135 (favors baseline).
