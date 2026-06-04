# Paired Statistical Evidence

Target method: `PU-R-Manifold`.

Differences are computed as `target - baseline` over paired seeds within matching experimental conditions. For lower-is-better metrics, negative values favor the target method.

## test_inappropriate_risk

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.006889, 95% CI [-0.01408, 0.0002985], win rate 0.60, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.008906, 95% CI [-0.01747, -0.0003441], win rate 0.53, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.006617, 95% CI [-0.0146, 0.001369], win rate 0.56, n=45 (inconclusive).

## certified_bound

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.008157, 95% CI [-0.0212, 0.004883], win rate 0.53, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff 0.0002356, 95% CI [-0.0136, 0.01407], win rate 0.44, n=45 (inconclusive).

## effective_compression_size

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -31.11, 95% CI [-78.97, 16.75], win rate 0.62, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -41.4, 95% CI [-104.3, 21.46], win rate 0.53, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff 1.6, 95% CI [-63.63, 66.83], win rate 0.44, n=45 (inconclusive).

## noise_hit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.01137, 95% CI [-0.01006, 0.0328], win rate 0.51, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.02486, 95% CI [-0.03268, -0.01704], win rate 0.82, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.009534, 95% CI [-0.01695, -0.002115], win rate 0.47, n=45 (favors target).

## group_revisit_rate

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff -0.01457, 95% CI [-0.03796, 0.008829], win rate 0.60, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff -0.0189, 95% CI [-0.03186, -0.005939], win rate 0.69, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff -0.005179, 95% CI [-0.01565, 0.005289], win rate 0.51, n=45 (inconclusive).

## unique_group_fraction

- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `GREATS`: mean diff 0.01457, 95% CI [-0.008829, 0.03796], win rate 0.60, n=45 (inconclusive).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `MaxLoss`: mean diff 0.0189, 95% CI [0.005939, 0.03186], win rate 0.69, n=45 (favors target).
- `results/PU-R/FashionMNIST/extensions_v4/pu_r_manifold/manifold_group_noise_0p1_low_support` vs `PU-R`: mean diff 0.005179, 95% CI [-0.005289, 0.01565], win rate 0.51, n=45 (inconclusive).
