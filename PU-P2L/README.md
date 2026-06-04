# PU-P2L Experiments

This folder is the cleaned implementation for the PU-P2L experiments. The
older `certifai_experiments/` folder is kept as a lab/reference area.

The main runbook is [`experiment_commands.md`](experiment_commands.md). It uses
one training run per experiment. If `Marginal` is included in `--methods`, the
runner still trains only once and then emits both reporting views:

```text
results/experiments/pu_r/<experiment>/
  results.csv
  summary.csv
  summary_with_marginal.csv
  summary_without_marginal.csv
  tables/
    with_marginal.csv
    without_marginal.csv
  plots/
    with_marginal/
    without_marginal/
```

This avoids rerunning the same setting only to remove `Marginal` from a plot or
table.

## Conda Setup

Run setup from the repository root, not from inside `PU-P2L/`.

```bash
conda env create -f environment.yml
conda activate certifai-experiments
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate certifai-experiments
```

On an A40 server, check that PyTorch sees CUDA before running GPU experiments:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

All commands assume:

```bash
export PYTHONPATH=PU-P2L
```

## Experiment Scale

The current runbook uses larger research-sized pools rather than debug-sized
1000-sample settings:

- Binary MNIST, MNIST10, Fashion-MNIST, rotated image datasets: `n_train=5000`
- CIFAR-10 reduced: `n_train=5000`
- Two-moons: `n_train=3000`
- Large sensitivity grids: `n_train=3000`
- Image test sets: `n_test=10000`

The larger pool is important for redundancy, noisy-point, and mode-coverage
diagnostics; with too few certification samples, the selector pathologies are
under-populated and the comparison can become uninformative.

## Implemented Methods

Certified selectors, using the P2L/P2L-ES certificate:

- `MaxLoss`: original P2L max-loss selector.
- `Marginal`: smallest softmax top-2 margin selector.
- `EL2N`: deterministic EL2N pruning score inside the P2L loop.
- `GraNdLast`: deterministic last-layer GraNd approximation inside the P2L loop.
- `RHO-PretrainRef`: deterministic reducible-loss score using a frozen model
  trained only on the pretraining split.
- `PU-R`: clipped loss plus residual novelty minus local redundancy.
- `PU-R-Vol`: PU-R with deterministic spectral-entropy volume adaptation of novelty and redundancy weights.
- `PU-R-Manifold`: PU-R with conservative deterministic support-graph manifold adaptation.

Non-certified reference:

- `GREATS`: GREATS-style probe-gradient selector. It is recorded as a practical
  reference; it is not treated as having a P2L compression certificate.

Ablation selectors:

- `ClippedLoss`
- `ResidualOnly`
- `RedundancyOnly`
- `Loss+Residual`
- `Loss-Redundancy`
- `PU-C-style`
- `Marginal+Residual`
- `Marginal-Redundancy`
- `Marginal+Residual-Redundancy`

## Supported Datasets

Use `--dataset-name` with one of:

- `synthetic_redundancy_hard`
- `mnist`: binary MNIST, digits `0` to `4` vs `5` to `9`
- `mnist10`: ten-class MNIST
- `fashion_mnist`: ten-class Fashion-MNIST
- `mode_mnist`: binary `{3,4}` vs `{5,9}` with controllable mode imbalance
- `boundary_duplicate_mnist`: `mode_mnist` plus redundant boundary-like samples
- `boundary_duplicate_fashion_mnist`: Fashion-MNIST analogue with redundant boundary-like samples
- `volume_group_noise_fashion_mnist`: low-volume Fashion-MNIST source groups with group-correlated label noise
- `manifold_group_noise_fashion_mnist`: rotated Fashion-MNIST source orbits with group-correlated label noise
- `rotated_mnist`: deterministic fixed-angle rotated MNIST
- `rotated_fashion_mnist`: deterministic fixed-angle rotated Fashion-MNIST
- `two_moons`: synthetic two-moons manifold diagnostic
- `cifar10`: reduced CIFAR-10 subset

Dataset-specific knobs:

- `--mode-imbalance`: Mode-A probability for `mode_mnist` and
  `boundary_duplicate_mnist`.
- `--boundary-augmentation`: redundant boundary augmentation multiplier for
  `boundary_duplicate_mnist`.
- `--rotation-angles`: fixed rotation domains for rotated datasets.

## Experiment Entry Points

- `python -m pu_p2l.run_boundary`: risk, P2L/PAC-Bayes bound, compression size,
  and runtime vs pretrain fraction.
- `python -m pu_p2l.run_noise`: risk, compression size, bounds, and selection
  diagnostics vs label-noise rate.
- `python -m pu_p2l.run_es_trace`: P2L-ES trajectories vs selection step.
- `python -m pu_p2l.run_es_budget_boundary`: fixed-ES-budget bounds/risk vs
  pretrain fraction.
- `python -m pu_p2l.run_es_budget_noise`: fixed-ES-budget bounds/risk vs noise.
- `python -m pu_p2l.run_generalization_bounds`: P2L, PAC-Bayes, and external
  generalization-bound comparison.
- `python -m pu_p2l.replot`: regenerate plots from existing `results.csv`.

## Recorded Metrics

The CSV outputs record both empirical and certificate-relevant quantities:

- `compression_size`
- `remaining_bad`
- `effective_compression_size`
- `certified_bound`
- `test_inappropriate_risk`
- `test_error`
- `runtime_sec`
- `train_calls`
- `noise_hit_rate`
- `duplicate_hit_rate`
- `pairwise_feature_cosine`
- `mean_support_redundancy`
- `max_support_redundancy`
- `mean_selected_residual_novelty`
- `local_redundancy_hit_rate`
- `residual_redundancy_hit_rate`
- `strong_redundancy_hit_rate`
- `mode_entropy`
- `minority_mode_fraction`
- `spectral_entropy`
- `dynamic_mu`

`test_inappropriate_risk` is the main empirical counterpart of the P2L
certificate. `test_error` is still recorded for standard ML interpretation.

## PAC-Bayes

PAC-Bayes is enabled for image-style datasets when `--pac-bayes-samples > 0`.
The default experiment commands use a Gaussian posterior over the classifier
head:

```bash
--pac-bayes-samples 50 --pac-bayes-train-epochs 1 --pac-bayes-scope head
```

PAC-Bayes is intentionally disabled for `synthetic_redundancy_hard`, where the
current stochastic-posterior baseline was not meaningful.

## Plotting

All plots use connected standard-error bands rather than independent error bars.
For ES traces, the full-step plot shows the ES P2L bound only to avoid clutter;
the first-100-step plot includes risk and bound. The trace runner also plots:

- `remaining_bad` vs step
- `effective_compression_size` vs step
- `spectral_entropy` vs step
- `dynamic_mu` vs step

To regenerate plots after changing style:

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/experiments/pu_r/binary_mnist/core_boundary \
  --kind boundary
```

Valid `--kind` values are `boundary`, `noise`, `es_trace`,
`es_budget_boundary`, `es_budget_noise`, and `generalization_bounds`.

## Runbook

Run commands one by one from:

```text
PU-P2L/experiment_commands.md
```

The raw CSV and `summary.csv` files contain the runtime and compression fields
needed for result tables.
