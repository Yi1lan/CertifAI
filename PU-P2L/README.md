# PU-P2L Thesis Experiments

This folder is the cleaned implementation for the PU-P2L thesis experiments.
The older `certifai_experiments/` folder is kept as a lab/reference area.

The authoritative runbook for the thesis package is
[`thesis_experiment_commands.md`](thesis_experiment_commands.md). It contains
all experiment commands under the hierarchy:

```text
results/thesis_v2/
  with_marginal/
  without_marginal/
```

Every thesis experiment should be run in both versions. The `with_marginal`
branch includes `Marginal` and marginal-derived ablations; the
`without_marginal` branch excludes them so the thesis can present a published
core comparison if Marginal is not used explicitly.

## Conda Setup

Run setup from the repository root, not from inside `PU-P2L/`.

```bash
cd /Users/yi1lan/Desktop/CertifAI
conda env create -f environment.yml
conda activate certifai-experiments
```

If the environment already exists:

```bash
cd /Users/yi1lan/Desktop/CertifAI
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

## Implemented Methods

Certified selectors, using the P2L/P2L-ES certificate:

- `MaxLoss`: original P2L max-loss selector.
- `Marginal`: smallest softmax top-2 margin selector.
- `PU-R`: clipped loss plus residual novelty minus local redundancy.
- `PU-R-Vol`: PU-R with deterministic spectral-entropy volume adaptation.
- `PU-R-Manifold`: PU-R with deterministic support-graph manifold adaptation.

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
The default thesis commands use a Gaussian posterior over the classifier head:

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
  --results-dir results/thesis_v2/with_marginal/binary_mnist/core_boundary \
  --kind boundary
```

Valid `--kind` values are `boundary`, `noise`, `es_trace`,
`es_budget_boundary`, `es_budget_noise`, and `generalization_bounds`.

## Thesis Runbook

Run commands one by one from:

```text
PU-P2L/thesis_experiment_commands.md
```

The command file follows the refined thesis plan:

1. Binary MNIST core certificate comparison.
2. Literature/generalization-bound comparison.
3. Mode-imbalanced MNIST for PU-R vs Marginal mechanism.
4. Boundary-duplicate augmentation.
5. PU-R ablations.
6. PU-R hyperparameter sensitivity.
7. Noisy binary MNIST.
8. MNIST 10-class.
9. Fashion-MNIST and PU-R-Vol.
10. Rotated-MNIST and PU-R-Manifold.
11. Two-moons manifold diagnostic.
12. Optional CIFAR-10 reduced.
13. Optional rotated Fashion-MNIST.

The raw CSV and `summary.csv` files contain the runtime and compression fields
needed for thesis tables.
