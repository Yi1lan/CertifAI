# PU-P2L Clean Experiments

This folder is the cleaned implementation for the PU-P2L experiments. The older
`certifai_experiments/` folder is kept as a lab/reference area.

Implemented methods:

- `MaxLoss`: standard P2L max-loss selector.
- `PU-C`: all-support feature novelty/redundancy selector.
- `PU-F`: label-consensus PU selector.
- `PU-G`: label-consensus PU selector with explicit noisy-label contradiction penalty.
- `GREATS`: non-certified GREATS-style probe-gradient reference selector.

All P2L/PU methods run without an early-stopping budget. A large
`--max-total-support` cap is still available as a safety guard. If the cap is
hit before Stop, the output records `stop_reached=0` and uses
`effective_compression_size = |T| + remaining_bad` as a diagnostic.

## Conda Setup

Run the setup commands from the repository root, not from inside `PU-P2L/`.

```bash
cd /Users/yi1lan/Desktop/CertifAI
conda env create -f environment.yml
conda activate certifai-experiments
```

If the environment already exists, update it instead:

```bash
cd /Users/yi1lan/Desktop/CertifAI
conda env update -f environment.yml --prune
conda activate certifai-experiments
```

On an NVIDIA A40 server, install the CUDA-enabled PyTorch build for that server
before running `--device cuda` if the default Conda solve gives a CPU build.

Check that the core packages are visible:

```bash
python -c "import torch, numpy, matplotlib; print(torch.__version__)"
```

All experiment commands below assume this Conda environment is active and use
`PYTHONPATH=PU-P2L` so Python can import the clean `pu_p2l` package.

The current dataset is recorded as `synthetic_redundancy_hard` in each CSV row,
and the recommended output directories below are namespaced under
`results/synthetic_redundancy_hard/`. Use a different `--dataset-name` and
output subdirectory when adding new datasets.

PAC-Bayes is disabled for `synthetic_redundancy_hard`. The current synthetic
experiments plot clean test risk and P2L compression certificates only. A
separate PAC-Bayes implementation can be enabled later for MNIST/CIFAR-style
datasets where the stochastic-posterior baseline is meaningful.

MNIST/CIFAR experiments are enabled through the same five entry points by
setting `--dataset-name mnist` or `--dataset-name cifar10`. They require
`torchvision`; `environment.yml` includes it. For image datasets, PAC-Bayes is
computed with a trained Gaussian posterior over the classifier head by default
(`--pac-bayes-scope head`) using the pretraining checkpoint as the prior.

## Boundary Experiment

This runs the hard synthetic setting at `noise=0.0` and `noise=0.4`, plotting
effective compression size, runtime, P2L bounds, and clean test risk
versus pretrain fraction.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_boundary \
  --output-dir results/synthetic_redundancy_hard/boundary \
  --dataset-name synthetic_redundancy_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.4 \
  --pretrain-fractions 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --methods MaxLoss PU-C PU-F PU-G GREATS \
  --n-train 3000 \
  --n-test 10000 \
  --duplicate-groups 40 \
  --duplicates-per-group 10 \
  --ambiguous-fraction 0.35 \
  --duplicate-std 0.015 \
  --initial-per-class 2 \
  --p2l-epochs-per-iter 1 \
  --max-total-support 600 \
  --lambda-redundancy 1.0 \
  --global-redundancy-weight 1.5 \
  --consensus-weight 1.25 \
  --noise-penalty 2.5
```

Outputs:

- `results.csv`: one row per seed, noise, pretrain fraction, and method.
- `summary.csv`: grouped means and standard errors.
- `plots/certified_bound_and_risk_vs_pretrain_noise_0.png`
- `plots/certified_bound_and_risk_vs_pretrain_noise_0p4.png`
- `plots/effective_compression_vs_pretrain_noise_0.png`
- `plots/effective_compression_vs_pretrain_noise_0p4.png`
- `plots/runtime_vs_pretrain_noise_0.png`
- `plots/runtime_vs_pretrain_noise_0p4.png`

## Noise Robustness Experiment

This runs the same hard synthetic setting while sweeping label-noise rate. All
methods run without early stopping.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_noise \
  --output-dir results/synthetic_redundancy_hard/noise \
  --dataset-name synthetic_redundancy_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.1 0.2 0.3 0.4 \
  --pretrain-fraction 0.0 \
  --methods MaxLoss PU-C PU-F PU-G GREATS \
  --n-train 3000 \
  --n-test 10000 \
  --duplicate-groups 40 \
  --duplicates-per-group 10 \
  --ambiguous-fraction 0.35 \
  --duplicate-std 0.015 \
  --initial-per-class 2 \
  --p2l-epochs-per-iter 1 \
  --max-total-support 800 \
  --lambda-redundancy 1.0 \
  --global-redundancy-weight 1.5 \
  --consensus-weight 1.25 \
  --noise-penalty 2.5
```

Outputs:

- `plots/test_error_vs_noise.png`
- `plots/compression_size_vs_noise.png`
- `plots/bounds_vs_noise.png`
- `plots/redundancy_diagnostics_vs_noise.png`

The redundancy plot contains noise-hit rate, duplicate-hit rate, and pairwise
feature cosine.

## Early-Stop Trace Experiment

This runs the hard synthetic setting and records the clean test risk and ES
certificate during the selection process. At each recorded step, the certificate
uses `|T_step| + remaining_bad_step` until Stop is reached, so intermediate
points can be interpreted as early-stopped P2L certificates. The comparison is
shown for `MaxLoss`, `PU-C`, `PU-F`, `PU-G`, and `GREATS`; `GREATS` appears
as clean test risk only because it has no P2L compression certificate.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_es_trace \
  --output-dir results/synthetic_redundancy_hard/es_trace \
  --dataset-name synthetic_redundancy_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.4 \
  --pretrain-fractions 0.0 \
  --methods MaxLoss PU-C PU-F PU-G GREATS \
  --record-every 5 \
  --n-train 3000 \
  --n-test 10000 \
  --duplicate-groups 40 \
  --duplicates-per-group 10 \
  --ambiguous-fraction 0.35 \
  --duplicate-std 0.015 \
  --initial-per-class 2 \
  --p2l-epochs-per-iter 1 \
  --max-total-support 600 \
  --lambda-redundancy 1.0 \
  --global-redundancy-weight 1.5 \
  --consensus-weight 1.25 \
  --noise-penalty 2.5
```

Outputs:

- `plots/es_bound_and_risk_vs_step_noise_0_pretrain_0.png`
- `plots/es_bound_and_risk_vs_step_noise_0p4_pretrain_0.png`
- `plots/es_bound_and_risk_vs_step_first_100_noise_0_pretrain_0.png`
- `plots/es_bound_and_risk_vs_step_first_100_noise_0p4_pretrain_0.png`

## Fixed-ES Budget Boundary Experiment

This evaluates early-stopped certificates at fixed ES budgets. For each budget,
the certificate uses `effective_compression_size = |T_ES| + remaining_bad_ES`.
Risk is plotted as a solid line and the ES P2L certificate as a dashed line
with the same method color.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/synthetic_redundancy_hard/es_budget_boundary \
  --dataset-name synthetic_redundancy_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.4 \
  --pretrain-fractions 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-C PU-F PU-G GREATS \
  --n-train 3000 \
  --n-test 10000 \
  --duplicate-groups 40 \
  --duplicates-per-group 10 \
  --ambiguous-fraction 0.35 \
  --duplicate-std 0.015 \
  --initial-per-class 2 \
  --p2l-epochs-per-iter 1 \
  --max-total-support 600 \
  --lambda-redundancy 1.0 \
  --global-redundancy-weight 1.5 \
  --consensus-weight 1.25 \
  --noise-penalty 2.5
```

Outputs include:

- `plots/es_budget_bound_and_risk_vs_pretrain_noise_0_budget_50.png`
- `plots/es_budget_bound_and_risk_vs_pretrain_noise_0_budget_100.png`
- `plots/es_budget_bound_and_risk_vs_pretrain_noise_0_budget_200.png`
- `plots/es_budget_bound_and_risk_vs_pretrain_noise_0p4_budget_50.png`
- `plots/es_budget_bound_and_risk_vs_pretrain_noise_0p4_budget_100.png`
- `plots/es_budget_bound_and_risk_vs_pretrain_noise_0p4_budget_200.png`

## Fixed-ES Budget Noise Experiment

This sweeps label-noise rate at fixed ES budgets and plots ES effective
compression size, clean test risk, and ES P2L bounds versus noise.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_es_budget_noise \
  --output-dir results/synthetic_redundancy_hard/es_budget_noise \
  --dataset-name synthetic_redundancy_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.1 0.2 0.3 0.4 \
  --pretrain-fraction 0.0 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-C PU-F PU-G GREATS \
  --n-train 3000 \
  --n-test 10000 \
  --duplicate-groups 40 \
  --duplicates-per-group 10 \
  --ambiguous-fraction 0.35 \
  --duplicate-std 0.015 \
  --initial-per-class 2 \
  --p2l-epochs-per-iter 1 \
  --max-total-support 600 \
  --lambda-redundancy 1.0 \
  --global-redundancy-weight 1.5 \
  --consensus-weight 1.25 \
  --noise-penalty 2.5
```

Outputs include:

- `plots/es_budget_effective_compression_vs_noise_budget_50_pretrain_0.png`
- `plots/es_budget_effective_compression_vs_noise_budget_100_pretrain_0.png`
- `plots/es_budget_effective_compression_vs_noise_budget_200_pretrain_0.png`
- `plots/es_budget_test_risk_vs_noise_budget_50_pretrain_0.png`
- `plots/es_budget_test_risk_vs_noise_budget_100_pretrain_0.png`
- `plots/es_budget_test_risk_vs_noise_budget_200_pretrain_0.png`
- `plots/es_budget_bounds_vs_noise_budget_50_pretrain_0.png`
- `plots/es_budget_bounds_vs_noise_budget_100_pretrain_0.png`
- `plots/es_budget_bounds_vs_noise_budget_200_pretrain_0.png`

## Regenerate Plots

Use this after changing plot style without rerunning experiments:

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/synthetic_redundancy_hard/boundary \
  --kind boundary
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/synthetic_redundancy_hard/noise \
  --kind noise
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/synthetic_redundancy_hard/es_trace \
  --kind es_trace
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/synthetic_redundancy_hard/es_budget_boundary \
  --kind es_budget_boundary
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/synthetic_redundancy_hard/es_budget_noise \
  --kind es_budget_noise
```
