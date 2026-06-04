# Paired Statistical Evidence

Target method: `PU-R-Manifold`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `GREATS`: mean diff 0.01281, 95% CI [0.007053, 0.01857], win rate 0.18, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `MaxLoss`: mean diff 0.001728, 95% CI [-0.003488, 0.006943], win rate 0.51, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `PU-R`: mean diff 0.003611, 95% CI [-0.001966, 0.009188], win rate 0.33, n=45 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `MaxLoss`: mean diff 0.004413, 95% CI [-0.005823, 0.01465], win rate 0.58, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `PU-R`: mean diff 0.005066, 95% CI [-0.005475, 0.01561], win rate 0.49, n=45 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `GREATS`: mean diff 63.53, 95% CI [-0.7547, 127.8], win rate 0.64, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `MaxLoss`: mean diff 23.42, 95% CI [-27.01, 73.85], win rate 0.58, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `PU-R`: mean diff 24.89, 95% CI [-27.37, 77.15], win rate 0.49, n=45 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `GREATS`: mean diff 0.3284, 95% CI [0.2512, 0.4055], win rate 0.16, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `MaxLoss`: mean diff -0.0124, 95% CI [-0.02275, -0.002059], win rate 0.64, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `PU-R`: mean diff 0.0134, 95% CI [0.004128, 0.02266], win rate 0.31, n=45 (favors baseline).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `GREATS`: mean diff 0.14, 95% CI [0.09505, 0.1849], win rate 0.24, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `MaxLoss`: mean diff -0.03749, 95% CI [-0.05023, -0.02475], win rate 0.80, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `PU-R`: mean diff 0.02365, 95% CI [0.009433, 0.03786], win rate 0.27, n=45 (favors baseline).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `GREATS`: mean diff -0.14, 95% CI [-0.1849, -0.09505], win rate 0.24, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `MaxLoss`: mean diff 0.03749, 95% CI [0.02475, 0.05023], win rate 0.80, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_manifold/manifold_orbit_noise_0p1_aug16_support` vs `PU-R`: mean diff -0.02365, 95% CI [-0.03786, -0.009433], win rate 0.27, n=45 (favors baseline).
