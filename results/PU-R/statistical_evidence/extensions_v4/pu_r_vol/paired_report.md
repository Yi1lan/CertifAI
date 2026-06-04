# Paired Statistical Evidence

Target method: `PU-R-Vol`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.005817, 95% CI [0.0009872, 0.01065], win rate 0.22, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.004144, 95% CI [-0.01038, 0.002096], win rate 0.44, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.003094, 95% CI [-0.009256, 0.003067], win rate 0.47, n=45 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.004765, 95% CI [-0.01068, 0.001145], win rate 0.49, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.002178, 95% CI [-0.008309, 0.003953], win rate 0.44, n=45 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff -15.24, 95% CI [-26.81, -3.681], win rate 0.96, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -21.44, 95% CI [-48.01, 5.121], win rate 0.49, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -9.978, 95% CI [-37.88, 17.92], win rate 0.44, n=45 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.01574, 95% CI [-0.02146, -0.01002], win rate 0.73, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff 0.003418, 95% CI [3.018e-05, 0.006806], win rate 0.13, n=45 (favors baseline).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.0006569, 95% CI [-0.003337, 0.002024], win rate 0.27, n=45 (inconclusive).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.01333, 95% CI [-0.02097, -0.0057], win rate 0.44, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.01605, 95% CI [-0.0238, -0.008289], win rate 0.60, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.005538, 95% CI [-0.01079, -0.0002817], win rate 0.31, n=45 (favors target).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.01333, 95% CI [0.0057, 0.02097], win rate 0.44, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `MaxLoss`: mean diff 0.01605, 95% CI [0.008289, 0.0238], win rate 0.60, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support` vs `PU-R`: mean diff 0.005538, 95% CI [0.0002817, 0.01079], win rate 0.31, n=45 (favors target).
