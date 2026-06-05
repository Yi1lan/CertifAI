# Paired Statistical Evidence

Target method: `PU-R-Vol`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.002583, 95% CI [-0.009548, 0.01471], win rate 0.33, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.02118, 95% CI [-0.03701, -0.005359], win rate 0.73, n=15 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.01348, 95% CI [-0.03097, 0.004007], win rate 0.60, n=15 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.01413, 95% CI [-0.03244, 0.004175], win rate 0.67, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.006581, 95% CI [-0.02652, 0.01336], win rate 0.73, n=15 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff -1.067, 95% CI [-37.56, 35.43], win rate 0.87, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -63.73, 95% CI [-146, 18.52], win rate 0.67, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -30.13, 95% CI [-120.9, 60.62], win rate 0.73, n=15 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.003544, 95% CI [-0.009875, 0.002787], win rate 0.60, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.003821, 95% CI [-0.008574, 0.0009329], win rate 0.40, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.006964, 95% CI [-0.01211, -0.001817], win rate 0.60, n=15 (favors target).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.01933, 95% CI [-0.03176, -0.006904], win rate 0.73, n=15 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.006361, 95% CI [-0.01574, 0.003014], win rate 0.40, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.006894, 95% CI [-0.02012, 0.006328], win rate 0.33, n=15 (inconclusive).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.01933, 95% CI [0.006904, 0.03176], win rate 0.73, n=15 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff 0.006361, 95% CI [-0.003014, 0.01574], win rate 0.40, n=15 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff 0.006894, 95% CI [-0.006328, 0.02012], win rate 0.33, n=15 (inconclusive).
