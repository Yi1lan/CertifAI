# Paired Statistical Evidence

Target method: `PU-R-Vol`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `GREATS`: mean diff 0.02174, 95% CI [0.01199, 0.0315], win rate 0.36, n=135 (favors baseline).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `MaxLoss`: mean diff -0.006446, 95% CI [-0.01687, 0.003981], win rate 0.60, n=135 (inconclusive).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `PU-R`: mean diff -0.001581, 95% CI [-0.01194, 0.008778], win rate 0.51, n=135 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `MaxLoss`: mean diff -0.001538, 95% CI [-0.01339, 0.01032], win rate 0.51, n=135 (inconclusive).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `PU-R`: mean diff 0.00216, 95% CI [-0.009696, 0.01402], win rate 0.51, n=135 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `GREATS`: mean diff 185.5, 95% CI [141.8, 229.1], win rate 0.19, n=135 (favors baseline).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `MaxLoss`: mean diff 3.63, 95% CI [-35.78, 43.04], win rate 0.51, n=135 (inconclusive).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `PU-R`: mean diff 21.79, 95% CI [-15.32, 58.89], win rate 0.51, n=135 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `GREATS`: mean diff -0.01988, 95% CI [-0.02748, -0.01229], win rate 0.64, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `MaxLoss`: mean diff -0.04005, 95% CI [-0.04772, -0.03238], win rate 0.79, n=135 (favors target).
- `results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8` vs `PU-R`: mean diff -0.01083, 95% CI [-0.01774, -0.003911], win rate 0.55, n=135 (favors target).
