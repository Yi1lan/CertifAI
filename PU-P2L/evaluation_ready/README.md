# PU-R Evaluation-Ready Results

This folder contains the reportable artifacts selected from the full experiment
workspace. It intentionally excludes `Marginal` plots/tables and old exploratory
extension versions unless they are needed as negative or partial evidence.

## Main Interpretation

The current result set is positive enough for the evaluation chapter. No further
experiment is needed before writing unless the thesis needs a larger-scale
external-dataset validation.

Reportable claims:

- `PU-R` is strongest under label noise and time-matched selection. In the
  Fashion-MNIST time-matched setting at noise `0.3`, PU-R has clean risk
  `0.6063`, compared with MaxLoss `0.8888`, EL2N `0.8585`, GraNd `0.8512`,
  RHO `0.9505`, and GREATS `0.9724`.
- On MNIST with noise `0.3` and no ES at pretrain `0.3`, PU-R has clean risk
  `0.2685`, compared with MaxLoss `0.4041` and GREATS `0.4220`.
- `PU-R-Manifold` is the reportable extension. In the confirmation run, it
  improves risk over GREATS by `-0.0260` with 95% CI `[-0.0359, -0.0161]`, and
  over MaxLoss by `-0.0308` with 95% CI `[-0.0386, -0.0230]`. It also reduces
  noise-hit rate versus GREATS and MaxLoss.
- Against plain `PU-R`, `PU-R-Manifold` is directionally better in risk but not
  statistically decisive in the confirmation run: mean difference `-0.0085`,
  95% CI `[-0.0202, 0.0031]`. Report it as a targeted extension, not as a
  universal replacement for PU-R.

Important limitations:

- Clean/no-redundancy settings are not the main win condition. PU-R is designed
  for noisy or redundant support-set construction, so small or neutral gains on
  clean MNIST are expected.
- `PU-R-Vol` is only partial evidence. It improves some diversity/noise-hit
  diagnostics but is not a strong risk-improvement method. Do not make it a
  headline result.
- `PU-R-Manifold` costs runtime. The confirmation report shows higher runtime
  than PU-R, GREATS, and MaxLoss, so the evaluation should present it as an
  accuracy/robustness tradeoff.

## Folder Contents

- `reports/`
  - `core_reportable_paired_report.md`: paired statistical evidence for the
    core PU-R claims.
  - `pu_r_manifold_confirmation_paired_report.md`: paired evidence for the
    confirmed PU-R-Manifold extension.
  - `pu_r_vol_partial_paired_report.md`: partial/limitation evidence for
    PU-R-Vol.
- `tables/core_mnist/`
  - MNIST no-ES and ES=100 summary tables for clean, noisy, and redundant
    settings.
- `tables/fashion_mnist/`
  - Time-matched and data-pruning comparison tables.
- `tables/extensions/`
  - PU-R-Manifold confirmation table.
- `plots/core_mnist/`
  - Core MNIST bound/risk and selection-diagnostic figures without Marginal.
- `plots/fashion_mnist_time_matched/`
  - Time-matched comparison figures.
- `plots/fashion_mnist_pruning/`
  - Data-pruning literature comparison figures.
- `plots/extensions/`
  - PU-R-Manifold confirmation figures.

The original full result folders remain under `results/PU-R/`.
