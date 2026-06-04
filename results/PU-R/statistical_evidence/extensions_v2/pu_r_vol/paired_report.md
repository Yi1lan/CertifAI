# Paired Statistical Evidence

Target method: `PU-R-Vol`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `GREATS`: mean diff 0.04163, 95% CI [0.02865, 0.05462], win rate 0.18, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `MaxLoss`: mean diff 0.0007222, 95% CI [-0.01232, 0.01376], win rate 0.47, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `PU-R`: mean diff -0.001461, 95% CI [-0.01637, 0.01345], win rate 0.56, n=45 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `MaxLoss`: mean diff 0.0068, 95% CI [-0.02058, 0.03418], win rate 0.47, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `PU-R`: mean diff -0.01531, 95% CI [-0.03602, 0.005408], win rate 0.58, n=45 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `GREATS`: mean diff 259.2, 95% CI [180.3, 338.1], win rate 0.13, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `MaxLoss`: mean diff 46.73, 95% CI [-59.13, 152.6], win rate 0.47, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `PU-R`: mean diff -45.64, 95% CI [-115.3, 23.96], win rate 0.58, n=45 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `GREATS`: mean diff 0.01591, 95% CI [0.005307, 0.02651], win rate 0.27, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `MaxLoss`: mean diff -0.0303, 95% CI [-0.04278, -0.01782], win rate 0.69, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `PU-R`: mean diff -0.002093, 95% CI [-0.01066, 0.006477], win rate 0.42, n=45 (inconclusive).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `GREATS`: mean diff 0.01095, 95% CI [-0.009931, 0.03182], win rate 0.33, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `MaxLoss`: mean diff -0.124, 95% CI [-0.1566, -0.09133], win rate 0.96, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `PU-R`: mean diff 0.0005498, 95% CI [-0.01759, 0.01869], win rate 0.49, n=45 (inconclusive).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `GREATS`: mean diff -0.01095, 95% CI [-0.03182, 0.009931], win rate 0.33, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `MaxLoss`: mean diff 0.124, 95% CI [0.09133, 0.1566], win rate 0.96, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16` vs `PU-R`: mean diff -0.0005498, 95% CI [-0.01869, 0.01759], win rate 0.49, n=45 (inconclusive).
