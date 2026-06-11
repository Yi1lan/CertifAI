# CertifAI PU-R

This repository contains the final implementation and result audit bundle for the CertifAI PU-R experiments. The project studies deterministic compression-compatible data selection for Pick-to-Learn (P2L), with the main implemented selector `PU-R` and two mechanism-specific extensions: `PU-R-Vol` and `PU-R-Manifold`.

The final report uses the cleaned implementation in `PU-P2L/`. The compact files required to inspect reported figures and tables are kept in `experiment-results/`.

## Repository Layout

```text
.
├── PU-P2L/
│   ├── README.md                         # package-level implementation notes
│   ├── pu_p2l/                           # executable experiment package
│   │   ├── data.py                       # dataset and perturbation builders
│   │   ├── scores.py                     # MaxLoss, PU-R, PU-R extensions
│   │   ├── runner.py                     # common P2L/P2L-ES execution logic
│   │   ├── plotting.py                   # plotting utilities
│   │   └── run_*.py                      # experiment entry points
│   ├── pu_p2l/experiment_setting.md      # final reproducibility runbook
│   └── pu_p2l_algorithm_details.md       # algorithm-level implementation spec
├── experiment-results/
│   ├── README.md                         # figure/table audit and rerun guide
│   ├── configs/                          # JSON configs for reported runs
│   ├── figures/                          # PNGs used in the evaluation
│   └── tables/                           # CSV/TeX sources for reported tables
├── P2L-Boundary-Graph/                   # optional P2L-bound plotting utility
├── environment.yml                       # conda environment specification
└── README.md
```

## Implemented Methods

The report focuses on:

- `MaxLoss`: the original P2L max-loss selector.
- `PU-R`: the proposed residual-novelty and redundancy-aware P2L selector.
- `PU-R-Vol`: a volume-adaptive PU-R extension, reported mainly as negative or partial evidence.
- `PU-R-Manifold`: a support-graph extension for source-orbit redundancy.
- `GREATS`: a strong empirical reference; it is not assigned a P2L compression certificate in this project.
- Data-pruning baselines: `EL2N`, `GraNdLast`, and `RHO-PretrainRef`, embedded as deterministic P2L acquisition scores for comparison.

## Environment

Create and activate the conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate certifai-experiments
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate certifai-experiments
```

On a CUDA machine, check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

All experiment commands assume:

```bash
export PYTHONPATH=PU-P2L
```

## Results

The final evaluation artifacts are in `experiment-results/`:

- `experiment-results/figures/`: plots used by the report.
- `experiment-results/tables/`: CSV/TeX table sources and statistical evidence.
- `experiment-results/configs/`: exact JSON run configurations.
- `experiment-results/README.md`: mapping from report figure/table labels to audit files and regeneration commands.

This folder is intentionally compact. It is not a full raw-output archive; it is the audit subset needed to verify the report figures and tables.

## Re-running Experiments

Use the final command runbook:

```text
PU-P2L/pu_p2l/experiment_setting.md
```

For example, after setting the common environment variables shown in that file, the core noisy MNIST no-ES experiment is:

```bash
python -m pu_p2l.run_boundary \
  --output-dir results/PU-R/MNIST/no_es/mnist_noise_0p3 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods MaxLoss PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

The full set of commands for the reported evaluation is listed in `experiment-results/README.md` and mirrored in the package runbook.
