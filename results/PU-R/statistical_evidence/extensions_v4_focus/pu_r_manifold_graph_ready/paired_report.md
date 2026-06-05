# Paired Statistical Evidence

Target method: `PU-R-Manifold`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.03465, 95% CI [-0.0464, -0.0229], win rate 1.00, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.05157, 95% CI [-0.06954, -0.03361], win rate 1.00, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.01405, 95% CI [-0.02943, 0.00133], win rate 0.70, n=10 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.05802, 95% CI [-0.102, -0.014], win rate 0.90, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.0137, 95% CI [-0.04711, 0.01971], win rate 0.60, n=10 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -39.3, 95% CI [-228.5, 149.9], win rate 0.50, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -283.5, 95% CI [-499.1, -67.92], win rate 0.90, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -66.3, 95% CI [-228.7, 96.13], win rate 0.60, n=10 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.04021, 95% CI [-0.05065, -0.02978], win rate 1.00, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.02624, 95% CI [-0.03975, -0.01274], win rate 0.90, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.0198, 95% CI [-0.03877, -0.0008205], win rate 0.50, n=10 (favors target).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.03199, 95% CI [-0.07273, 0.008744], win rate 0.80, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff 0.002828, 95% CI [-0.02343, 0.02909], win rate 0.50, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.01118, 95% CI [-0.03409, 0.01173], win rate 0.60, n=10 (inconclusive).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.03199, 95% CI [-0.008744, 0.07273], win rate 0.80, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.002828, 95% CI [-0.02909, 0.02343], win rate 0.50, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff 0.01118, 95% CI [-0.01173, 0.03409], win rate 0.60, n=10 (inconclusive).
