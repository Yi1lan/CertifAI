# PU-P2L Package

`PU-P2L` is the cleaned experiment package for the CertifAI PU-R study. It implements deterministic P2L/P2L-ES acquisition loops, PU-R scoring, comparison selectors, dataset perturbations, diagnostics, plotting, and statistical evidence utilities.

Run commands from the repository root with:

```bash
export PYTHONPATH=PU-P2L
```

## Package Structure

```text
PU-P2L/
├── pu_p2l/
│   ├── data.py                         # datasets, label noise, redundancy
│   ├── model.py                        # MLP and image-model helpers
│   ├── scores.py                       # selector score functions
│   ├── runner.py                       # shared P2L/P2L-ES training loop
│   ├── bounds.py                       # P2L and PAC-Bayes bound utilities
│   ├── adaptive_generalization_bounds.py
│   ├── plotting.py                     # plots and report-ready styles
│   ├── replot.py                       # regenerate plots from CSV outputs
│   ├── run_boundary.py                 # no-ES bound/risk vs pretrain
│   ├── run_es_budget_boundary.py       # fixed-ES bound/risk vs pretrain
│   ├── run_generalization_bounds.py    # P2L/PAC-Bayes/literature bounds
│   ├── run_time_matched_noise.py       # time-matched pruning comparison
│   ├── run_pu_r_hyperparameter_ablation.py
│   ├── run_selection_visualization.py
│   ├── run_selection_report.py
│   └── run_statistical_evidence.py
├── pu_p2l/experiment_setting.md        # final runbook
└── pu_p2l_algorithm_details.md         # score and algorithm specification
```

The top-level `run_*.py` files are thin compatibility wrappers around the module entry points. The recommended invocation is the module form, for example:

```bash
python -m pu_p2l.run_boundary --help
```

## Core Selectors

Reportable selectors:

- `MaxLoss`: original P2L max-loss acquisition.
- `PU-R`: clipped loss plus residual novelty minus redundancy.
- `PU-R-Vol`: PU-R with deterministic spectral-volume adaptation.
- `PU-R-Manifold`: PU-R with deterministic support-graph manifold refinement.
- `GREATS`: empirical probe-gradient selector. It is reported as a strong reference, but it is not assigned a P2L compression certificate here.
- `EL2N`, `GraNdLast`, `RHO-PretrainRef`: data-pruning scores adapted into deterministic P2L acquisition rules for Fashion-MNIST comparisons.

## Supported Datasets

The final evaluation uses:

- `mnist`: binary MNIST, digits `0`--`4` vs `5`--`9`.
- `boundary_duplicate_mnist`: binary MNIST with controlled redundant boundary samples.
- `fashion_mnist`: ten-class Fashion-MNIST for pruning comparisons.
- `boundary_duplicate_fashion_mnist`: redundant Fashion-MNIST stress setting.
- `volume_group_noise_fashion_mnist`: low-volume source-group redundancy.
- `manifold_group_noise_fashion_mnist`: rotated source-orbit redundancy.

Additional dataset builders remain available in `data.py` for diagnostics, but the report and audit bundle are based on the datasets above.

## Main Metrics

The CSV outputs include:

- `test_error`: ordinary clean classification error.
- `test_inappropriate_risk`: clean P2L inappropriate-risk event.
- `compression_size`: selected support size.
- `remaining_bad`: number of unresolved inappropriate pool examples.
- `effective_compression_size`: `compression_size + remaining_bad` for ES reporting.
- `certified_bound`: P2L or P2L-ES compression certificate when the method is admissible.
- `noise_hit_rate`, `duplicate_hit_rate`, `group_revisit_rate`, and feature redundancy diagnostics.
- `runtime_sec` and `train_calls`.

`test_inappropriate_risk` is the empirical risk notion aligned with the P2L certificate. `test_error` is reported separately for standard ML interpretation.

## Default Experimental Settings

The final report uses:

- `n_train=5000`, `n_test=10000`.
- `model_name=mnist_fcn`: fully connected `784-600-600-600-C` network.
- SGD with momentum `0.95`, learning rate `0.01`, dropout `0.2`.
- `pretrain_training_mode=support`.
- P2L threshold `gamma=-log(0.5)`.
- P2L certificate failure probability `delta=0.035` (`96.5%` confidence).
- MNIST support cap `800`; Fashion-MNIST support cap `1000`.

The exact settings for each reported result are recorded in `../experiment-results/configs/` and summarized in `../experiment-results/README.md`.

## Reproducibility

Use:

```text
PU-P2L/pu_p2l/experiment_setting.md
```

for the final command list. Use:

```text
experiment-results/README.md
```

to map each report figure/table to the copied audit artifact and the command that regenerates it.

Example:

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/MNIST/es100/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods MaxLoss PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

## Plotting

Plotting uses connected standard-error bands. To regenerate plots from existing CSV outputs:

```bash
python -m pu_p2l.replot \
  --results-dir <result-folder> \
  --kind <boundary|es_budget_boundary|generalization_bounds|noise|es_trace>
```

The report-ready plot and table files are copied into `../experiment-results/`.
