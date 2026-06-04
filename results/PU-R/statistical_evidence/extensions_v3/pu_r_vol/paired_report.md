# Paired Statistical Evidence

Target method: `PU-R-Vol`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `GREATS`: mean diff 0.02809, 95% CI [0.01677, 0.0394], win rate 0.22, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `MaxLoss`: mean diff -0.005817, 95% CI [-0.01348, 0.001847], win rate 0.60, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `PU-R`: mean diff 0.001006, 95% CI [-0.006631, 0.008643], win rate 0.56, n=45 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `MaxLoss`: mean diff -0.006681, 95% CI [-0.02668, 0.01332], win rate 0.40, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `PU-R`: mean diff -0.01224, 95% CI [-0.02725, 0.002767], win rate 0.44, n=45 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `GREATS`: mean diff 173.9, 95% CI [65.63, 282.2], win rate 0.69, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `MaxLoss`: mean diff -34.38, 95% CI [-134, 65.2], win rate 0.40, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `PU-R`: mean diff -62.11, 95% CI [-137.1, 12.9], win rate 0.44, n=45 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `GREATS`: mean diff 0.2272, 95% CI [0.177, 0.2774], win rate 0.07, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `MaxLoss`: mean diff -0.07013, 95% CI [-0.08635, -0.05392], win rate 0.93, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `PU-R`: mean diff -0.01787, 95% CI [-0.02657, -0.009167], win rate 0.78, n=45 (favors target).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `GREATS`: mean diff 0.09601, 95% CI [0.06066, 0.1314], win rate 0.20, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `MaxLoss`: mean diff -0.1944, 95% CI [-0.2396, -0.1492], win rate 1.00, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `PU-R`: mean diff -0.02751, 95% CI [-0.03565, -0.01936], win rate 0.91, n=45 (favors target).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `GREATS`: mean diff -0.09601, 95% CI [-0.1314, -0.06066], win rate 0.20, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `MaxLoss`: mean diff 0.1944, 95% CI [0.1492, 0.2396], win rate 1.00, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v3/pu_r_vol/volume_gap_noise_0p1_aug32_support` vs `PU-R`: mean diff 0.02751, 95% CI [0.01936, 0.03565], win rate 0.91, n=45 (favors target).
