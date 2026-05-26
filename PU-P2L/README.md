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

Check that the core packages are visible:

```bash
python -c "import torch, numpy, matplotlib; print(torch.__version__)"
```

All experiment commands below assume this Conda environment is active and use
`PYTHONPATH=PU-P2L` so Python can import the clean `pu_p2l` package.

## Boundary Experiment

This runs the hard synthetic setting at `noise=0.0` and `noise=0.4`, plotting
effective compression size, runtime, certified bound, and clean test risk versus
pretrain fraction. The certified-bound plot includes only certifiable methods
(`MaxLoss`, `PU-C`, `PU-F`, and `PU-G`); `GREATS` remains a non-certified
reference selector.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_boundary \
  --output-dir results/pu_p2l_clean_boundary_hard \
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
  --output-dir results/pu_p2l_clean_noise_hard \
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
- `plots/redundancy_diagnostics_vs_noise.png`

The redundancy plot contains noise-hit rate, duplicate-hit rate, and pairwise
feature cosine.

## Early-Stop Trace Experiment

This runs the hard synthetic setting and records the clean test risk and ES
certificate during the selection process. At each recorded step, the certificate
uses `|T_step| + remaining_bad_step` until Stop is reached, so intermediate
points can be interpreted as early-stopped P2L certificates. The comparison is
restricted to certifiable methods: `MaxLoss`, `PU-C`, `PU-F`, and `PU-G`.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_es_trace \
  --output-dir results/pu_p2l_es_trace_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.4 \
  --pretrain-fractions 0.0 \
  --methods MaxLoss PU-C PU-F PU-G \
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
Risk is plotted as a solid line and the ES certificate as a dashed line with the
same method color.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/pu_p2l_es_budget_boundary_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.4 \
  --pretrain-fractions 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-C PU-F PU-G \
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
compression size and clean test risk versus noise.

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.run_es_budget_noise \
  --output-dir results/pu_p2l_es_budget_noise_hard \
  --device cpu \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.0 0.1 0.2 0.3 0.4 \
  --pretrain-fraction 0.0 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-C PU-F PU-G \
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

## Regenerate Plots

Use this after changing plot style without rerunning experiments:

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/pu_p2l_clean_boundary_hard \
  --kind boundary
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/pu_p2l_clean_noise_hard \
  --kind noise
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/pu_p2l_es_trace_hard \
  --kind es_trace
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/pu_p2l_es_budget_boundary_hard \
  --kind es_budget_boundary
```

```bash
PYTHONPATH=PU-P2L python -m pu_p2l.replot \
  --results-dir results/pu_p2l_es_budget_noise_hard \
  --kind es_budget_noise
```
