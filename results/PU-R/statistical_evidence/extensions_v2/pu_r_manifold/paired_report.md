# Paired Statistical Evidence

Target method: `PU-R-Manifold`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `GREATS`: mean diff 0.01249, 95% CI [0.002071, 0.02292], win rate 0.29, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `MaxLoss`: mean diff 0.005028, 95% CI [-0.004715, 0.01477], win rate 0.49, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `PU-R`: mean diff -0.0004889, 95% CI [-0.009332, 0.008354], win rate 0.44, n=45 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `MaxLoss`: mean diff -0.006418, 95% CI [-0.01884, 0.006], win rate 0.49, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `PU-R`: mean diff -0.005179, 95% CI [-0.01921, 0.008851], win rate 0.62, n=45 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `GREATS`: mean diff 47.44, 95% CI [1.997, 92.89], win rate 0.33, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `MaxLoss`: mean diff -36.31, 95% CI [-86.54, 13.92], win rate 0.49, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `PU-R`: mean diff -33.07, 95% CI [-87.64, 21.5], win rate 0.62, n=45 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `GREATS`: mean diff -0.005861, 95% CI [-0.02059, 0.008867], win rate 0.47, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `MaxLoss`: mean diff -0.01833, 95% CI [-0.03316, -0.003503], win rate 0.56, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `PU-R`: mean diff -0.003472, 95% CI [-0.01793, 0.01099], win rate 0.49, n=45 (inconclusive).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `GREATS`: mean diff -0.05827, 95% CI [-0.0789, -0.03764], win rate 0.82, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `MaxLoss`: mean diff -0.05928, 95% CI [-0.07959, -0.03897], win rate 0.71, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `PU-R`: mean diff -0.022, 95% CI [-0.04247, -0.001523], win rate 0.58, n=45 (favors target).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `GREATS`: mean diff 0.05827, 95% CI [0.03764, 0.0789], win rate 0.82, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `MaxLoss`: mean diff 0.05928, 95% CI [0.03897, 0.07959], win rate 0.71, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10` vs `PU-R`: mean diff 0.022, 95% CI [0.001523, 0.04247], win rate 0.58, n=45 (favors target).
