# Paired Statistical Evidence

Target method: `PU-R-Manifold`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `GREATS`: mean diff -0.02597, 95% CI [-0.03587, -0.01607], win rate 0.87, n=30 (favors target).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `MaxLoss`: mean diff -0.03079, 95% CI [-0.03861, -0.02297], win rate 0.90, n=30 (favors target).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `PU-R`: mean diff -0.008508, 95% CI [-0.02016, 0.003146], win rate 0.57, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.03465, 95% CI [-0.0464, -0.0229], win rate 1.00, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.05157, 95% CI [-0.06954, -0.03361], win rate 1.00, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.01405, 95% CI [-0.02943, 0.00133], win rate 0.70, n=10 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `MaxLoss`: mean diff -0.01218, 95% CI [-0.03223, 0.007862], win rate 0.53, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `PU-R`: mean diff 1.413e-05, 95% CI [-0.01845, 0.01848], win rate 0.53, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.05802, 95% CI [-0.102, -0.014], win rate 0.90, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.0137, 95% CI [-0.04711, 0.01971], win rate 0.60, n=10 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `GREATS`: mean diff 61.43, 95% CI [-18.17, 141], win rate 0.47, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `MaxLoss`: mean diff -60.17, 95% CI [-157.5, 37.21], win rate 0.53, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `PU-R`: mean diff -0.6333, 95% CI [-90.28, 89.01], win rate 0.53, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -39.3, 95% CI [-228.5, 149.9], win rate 0.50, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -283.5, 95% CI [-499.1, -67.92], win rate 0.90, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -66.3, 95% CI [-228.7, 96.13], win rate 0.60, n=10 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `GREATS`: mean diff -0.02056, 95% CI [-0.0266, -0.01451], win rate 0.83, n=30 (favors target).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `MaxLoss`: mean diff -0.02315, 95% CI [-0.0331, -0.01321], win rate 0.80, n=30 (favors target).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `PU-R`: mean diff -0.001464, 95% CI [-0.007174, 0.004245], win rate 0.43, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.04021, 95% CI [-0.05065, -0.02978], win rate 1.00, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.02624, 95% CI [-0.03975, -0.01274], win rate 0.90, n=10 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.0198, 95% CI [-0.03877, -0.0008205], win rate 0.50, n=10 (favors target).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `GREATS`: mean diff -0.01963, 95% CI [-0.03763, -0.001639], win rate 0.60, n=30 (favors target).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `MaxLoss`: mean diff -0.005505, 95% CI [-0.02688, 0.01587], win rate 0.40, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `PU-R`: mean diff -0.008855, 95% CI [-0.02446, 0.006751], win rate 0.57, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.03199, 95% CI [-0.07273, 0.008744], win rate 0.80, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff 0.002828, 95% CI [-0.02343, 0.02909], win rate 0.50, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.01118, 95% CI [-0.03409, 0.01173], win rate 0.60, n=10 (inconclusive).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `GREATS`: mean diff 0.01963, 95% CI [0.001639, 0.03763], win rate 0.60, n=30 (favors target).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `MaxLoss`: mean diff 0.005505, 95% CI [-0.01587, 0.02688], win rate 0.40, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready` vs `PU-R`: mean diff 0.008855, 95% CI [-0.006751, 0.02446], win rate 0.57, n=30 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.03199, 95% CI [-0.008744, 0.07273], win rate 0.80, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.002828, 95% CI [-0.02909, 0.02343], win rate 0.50, n=10 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff 0.01118, 95% CI [-0.01173, 0.03409], win rate 0.60, n=10 (inconclusive).
