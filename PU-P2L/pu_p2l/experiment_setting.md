# Core PU-R MNIST Experiments

These commands generate the core MNIST results showing whether PU-R improves
P2L-style compression under label noise and redundant data. Run from the
repository root.

The commands train all core methods once, including `Marginal`. The plotting
code then writes both views automatically:

- `plots/with_marginal/`
- `plots/without_marginal/`

For redundant MNIST, we use `boundary_duplicate_mnist`, which is the implemented
MNIST-style redundant boundary dataset. The setting `--boundary-augmentation 5`
injects repeated boundary-like samples so the selector can be tested against
redundant high-loss candidates.

## Pretrain Semantics

The original P2L code treats the pretrain fraction as the initial support set:
those points remain in the training set during every P2L iteration, but the P2L
certificate charges only the newly picked certification points. To make this
choice explicit, all experiment entry points support:

- `--pretrain-training-mode warm_start`: current PU-P2L behavior used by the
  existing result folders. The model is trained once on the pretrain split, then
  each P2L iteration trains only on selected certification support.
- `--pretrain-training-mode support`: original P2L-aligned behavior. The
  pretrain split is included in every iterative P2L training update and is not
  counted in the compression size. There is no separate one-shot warm start in
  this mode, so `--p2l-epochs-per-iter` controls the training applied to
  pretrain-plus-support during the P2L loop.
- `--pretrain-training-mode warm_start_and_support`: diagnostic ablation. The
  model is first warm-started on pretrain data and then the same pretrain data is
  also included in every P2L update.

For the MNIST runs below, use `support` when comparing the bound-vs-pretrain
shape against the original P2L experiments. Use `warm_start` only to reproduce
the previous PU-P2L result directories.

## Setup

```bash
conda activate certifai-experiments

export PYTHONPATH=PU-P2L
export DEVICE=cuda
export DATA_DIR=data
export SEEDS5="0 1 2 3 4"

export N_MNIST=5000
export N_IMAGE=5000
export N_TEST=10000
export SUPPORT_MNIST=800
export SUPPORT_IMAGE=1000
export PRETRAIN_TRAINING_MODE=support

export MNIST_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name mnist_fcn --optimizer sgd --momentum 0.95 --batch-size 60000 --inference-batch-size 1024 --pretrain-training-mode $PRETRAIN_TRAINING_MODE --pretrain-epochs 20 --p2l-epochs-per-iter 5 --pretrain-lr 0.01 --p2l-lr 0.01 --dropout-prob 0.2 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"
export IMAGE_COMMON="$MNIST_COMMON"

export CORE_METHODS="MaxLoss Marginal PU-R GREATS"
export TRACE_METHODS="MaxLoss Marginal PU-R"
export LITERATURE_BOUND_METHODS="PU-R GREATS"
export PRETRAIN_GRID="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
export TRACE_PRETRAIN="0.0"
```

## No ES

### 1. Clean MNIST, No Redundancy

Noise is `0.0`, redundancy is off, and the plot is risk/certificate bound vs
pretrain fraction.

```bash
python -m pu_p2l.run_boundary \
  --output-dir results/PU-R/MNIST/no_es/mnist_noise_0 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods $CORE_METHODS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 2. Noisy MNIST, No Redundancy

Noise is `0.3`, redundancy is off, and the plot is risk/certificate bound vs
pretrain fraction.

```bash
python -m pu_p2l.run_boundary \
  --output-dir results/PU-R/MNIST/no_es/mnist_noise_0p3 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods $CORE_METHODS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 3. Noisy MNIST, Redundant Boundary Samples

Noise is `0.3`, redundancy is on via `boundary_duplicate_mnist`, and the plot is
risk/certificate bound vs pretrain fraction.

```bash
python -m pu_p2l.run_boundary \
  --output-dir results/PU-R/MNIST/no_es/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods $CORE_METHODS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

## ES = 100

These commands use the P2L-ES certificate with fixed early-stopping budget
`ES=100`.

### 4. Clean MNIST, No Redundancy, ES=100

Noise is `0.0`, redundancy is off, and the plot is ES certificate/risk vs
pretrain fraction.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/MNIST/es100/mnist_noise_0 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods $CORE_METHODS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 5. Noisy MNIST, No Redundancy, ES=100

Noise is `0.3`, redundancy is off, and the plot is ES certificate/risk vs
pretrain fraction.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/MNIST/es100/mnist_noise_0p3 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods $CORE_METHODS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 6. Noisy MNIST, Redundant Boundary Samples, ES=100

Noise is `0.3`, redundancy is on via `boundary_duplicate_mnist`, and the plot is
ES certificate/risk vs pretrain fraction.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/MNIST/es100/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods $CORE_METHODS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

## Iteration-Wise ES Bound Diagnostics

These experiments track the ES P2L generalization bound over the selection
trajectory. The plot marks the minimum bound point and its corresponding
iteration. We record every 2 selection steps and train/update the traced model
every 2 selected points, which gives the intended block trace for these
boundary-only plots.

Each command below produces two requested boundary-only plots:

- first 100 iterations:
  `plots/*/es_bound_vs_step_first_100_noise_*_pretrain_0.png`
- full recorded trajectory:
  `plots/*/es_bound_vs_step_noise_*_pretrain_0.png`

Therefore the three commands below generate the six requested plots.

### 1 and 4. Clean MNIST, No Redundancy, ES Bound vs Iteration

Noise is `0.0`, redundancy is off, and the plots are ES bound vs iteration for
the first 100 steps and for the full recorded trajectory.

```bash
python -m pu_p2l.run_es_trace \
  --output-dir results/PU-R/MNIST/es_trace/mnist_noise_0 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $TRACE_PRETRAIN \
  --methods $TRACE_METHODS \
  --record-every 2 \
  --train-every 2 \
  --bound-only \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 2 and 5. Noisy MNIST, No Redundancy, ES Bound vs Iteration

Noise is `0.3`, redundancy is off, and the plots are ES bound vs iteration for
the first 100 steps and for the full recorded trajectory.

```bash
python -m pu_p2l.run_es_trace \
  --output-dir results/PU-R/MNIST/es_trace/mnist_noise_0p3 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $TRACE_PRETRAIN \
  --methods $TRACE_METHODS \
  --record-every 2 \
  --train-every 2 \
  --bound-only \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 3 and 6. Noisy MNIST, Redundant Boundary Samples, ES Bound vs Iteration

Noise is `0.3`, redundancy is on via `boundary_duplicate_mnist`, and the plots
are ES bound vs iteration for the first 100 steps and for the full recorded
trajectory.

```bash
python -m pu_p2l.run_es_trace \
  --output-dir results/PU-R/MNIST/es_trace/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $TRACE_PRETRAIN \
  --methods $TRACE_METHODS \
  --record-every 2 \
  --train-every 2 \
  --bound-only \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

To run the whole ES-trace group without typing each command manually:

```bash
bash PU-P2L/run_core_mnist_es_trace.sh
```

## Literature Generalization-Bound Comparison

These experiments compare PU-R against GREATS under the implemented
generalization-bound baselines:

- PU-R P2L compression certificate
- PAC-Bayes bound
- self-selected-data bound
- ADA growing-data bound

GREATS is included as a practical data-selection reference. The code does not
assign GREATS a P2L compression certificate, so its P2L curve is intentionally
absent; its risk and non-P2L literature-bound diagnostics are still reported.

### 1. Clean MNIST, No Redundancy

Noise is `0.0`, redundancy is off, and the plot is generalization bound vs
pretrain fraction.

```bash
python -m pu_p2l.run_generalization_bounds \
  --output-dir results/PU-R/MNIST/literature_bounds/mnist_noise_0 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --pac-bayes-samples 50 \
  --pac-bayes-train-epochs 1 \
  --pac-bayes-scope head
```

### 2. Noisy MNIST, No Redundancy

Noise is `0.3`, redundancy is off, and the plot is generalization bound vs
pretrain fraction.

```bash
python -m pu_p2l.run_generalization_bounds \
  --output-dir results/PU-R/MNIST/literature_bounds/mnist_noise_0p3 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --pac-bayes-samples 50 \
  --pac-bayes-train-epochs 1 \
  --pac-bayes-scope head
```

### 3. Noisy MNIST, Redundant Boundary Samples

Noise is `0.3`, redundancy is on via `boundary_duplicate_mnist`, and the plot is
generalization bound vs pretrain fraction.

```bash
python -m pu_p2l.run_generalization_bounds \
  --output-dir results/PU-R/MNIST/literature_bounds/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5 \
  --pac-bayes-samples 50 \
  --pac-bayes-train-epochs 1 \
  --pac-bayes-scope head
```

## Fashion-MNIST Data-Pruning Literature Comparison

These experiments compare PU-R against data-pruning and selection baselines on
Fashion-MNIST with fixed P2L-ES budgets:

```text
MaxLoss, Marginal, EL2N, GraNdLast, RHO-PretrainRef, PU-R, GREATS
```

For redundant Fashion-MNIST we use `boundary_duplicate_fashion_mnist`. This is a
Fashion-MNIST analogue of `boundary_duplicate_mnist`: it uses ambiguous clothing
pairs and duplicates the dominant mode through rotated repeated sources. The
binary pairs are:

```text
mode 0: T-shirt/top vs shirt
mode 1: pullover vs coat
```

### 1. Fashion-MNIST, Noise=0, No Redundancy, ES=50

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es50/noise_0 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 50 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 2. Fashion-MNIST, Noise=0, No Redundancy, ES=100

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es100/noise_0 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 3. Fashion-MNIST, Noise=0, No Redundancy, ES=200

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es200/noise_0 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 200 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 4. Fashion-MNIST, Noise=0.3, No Redundancy, ES=50

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es50/noise_0p3 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 50 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 5. Fashion-MNIST, Noise=0.3, No Redundancy, ES=100

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es100/noise_0p3 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 6. Fashion-MNIST, Noise=0.3, No Redundancy, ES=200

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es200/noise_0p3 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 200 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 7. Fashion-MNIST, Noise=0.3, Redundant Boundary Samples, ES=50

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es50/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 50 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

### 8. Fashion-MNIST, Noise=0.3, Redundant Boundary Samples, ES=100

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es100/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

### 9. Fashion-MNIST, Noise=0.3, Redundant Boundary Samples, ES=200

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es200/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 200 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

## PU-R Extension Stress Tests

These compact experiments are designed specifically for the two PU-R extension
methods. Both use noise `0.1`, redundant data, and ES budgets `50`, `100`, and
`200` in one run.

### PU-R-Vol: Low-Volume Redundant Fashion-MNIST

This dataset uses `volume_duplicate_fashion_mnist`, where the dominant
Fashion-MNIST mode is generated from a small number of repeated sources. This
creates low spectral diversity in the selected support set, which is the regime
where `PU-R-Vol` should improve on plain `PU-R`.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/extensions/pu_r_vol/volume_duplicate_noise_0p1_aug8 \
  --dataset-name volume_duplicate_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 50 100 200 \
  --methods MaxLoss Marginal PU-R PU-R-Vol GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 8 \
  --alpha 2.0
```

### PU-R-Manifold: Rotated Redundant Fashion-MNIST

This dataset uses `manifold_duplicate_fashion_mnist`, where repeated dominant
mode sources are distributed over fixed rotation angles. This creates redundant
points along a nonlinear rotation manifold, which is the regime where
`PU-R-Manifold` should improve on plain Euclidean residual novelty.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/extensions/pu_r_manifold/manifold_duplicate_noise_0p1_aug5 \
  --dataset-name manifold_duplicate_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 50 100 200 \
  --methods MaxLoss Marginal PU-R PU-R-Manifold GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5 \
  --rotation-angles -60 -30 0 30 60 \
  --manifold-k 5 \
  --manifold-tau 0.2 \
  --manifold-eigenvectors 8
```

## Source-Orbit Extension Validation

The previous extension stress tests are broad but do not isolate the extension
mechanisms strongly enough. The following experiments use datasets whose group
IDs are source-orbit IDs, so the diagnostics can directly measure whether a
selector repeatedly picks augmented copies from the same source.

These commands exclude `Marginal` because it is not a reportable method. They
also use a compact pretrain grid to keep the runs practical while still checking
low, medium, and high pretraining regimes.

### PU-R-Vol: Low-Volume Gap Dataset

This dataset uses `volume_gap_fashion_mnist`. The dominant mode is generated
from a very small number of repeated source images with tiny perturbations,
while the remaining mode is more diverse. This creates support-volume collapse,
where plain `PU-R` can remain close to the repeated dominant source groups.
`PU-R-Vol` is tested with a stronger entropy-driven novelty boost.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/extensions_v2/pu_r_vol/volume_gap_noise_0p1_aug16 \
  --dataset-name volume_gap_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.0 0.3 0.6 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Vol GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.95 \
  --boundary-augmentation 16 \
  --mu 0.35 \
  --alpha 4.0 \
  --global-redundancy-weight 1.0
```

### PU-R-Manifold: Source-Orbit Rotation Dataset

This dataset uses `manifold_orbit_fashion_mnist`. Repeated dominant-mode samples
are generated as rotation orbits of the same source images, and all copies from
the same source share one group ID. This is the intended setting for
`PU-R-Manifold`: Euclidean residual novelty may treat rotated copies as new,
while the graph-diffusion penalty should reduce repeated orbit selection.

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold/manifold_orbit_noise_0p1_aug10 \
  --dataset-name manifold_orbit_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.0 0.3 0.6 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Manifold GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 10 \
  --rotation-angles -45 -30 -15 0 15 30 45 \
  --mu 0.5 \
  --global-redundancy-weight 0.75 \
  --manifold-k 7 \
  --manifold-tau 0.8 \
  --manifold-eigenvectors 8
```

After running these two commands, generate paired extension evidence:

```bash
python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/FashionMNIST/extensions_v2/pu_r_vol \
  --output-dir results/PU-R/statistical_evidence/extensions_v2/pu_r_vol \
  --target-method PU-R-Vol \
  --baselines PU-R MaxLoss GREATS \
  --metrics test_inappropriate_risk certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate group_revisit_rate unique_group_fraction mode_entropy spectral_entropy runtime_sec

python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/FashionMNIST/extensions_v2/pu_r_manifold \
  --output-dir results/PU-R/statistical_evidence/extensions_v2/pu_r_manifold \
  --target-method PU-R-Manifold \
  --baselines PU-R MaxLoss GREATS \
  --metrics test_inappropriate_risk certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate group_revisit_rate unique_group_fraction mode_entropy spectral_entropy runtime_sec
```

## Extension Hyperparameter Sensitivity

Run these only after the source-orbit validation runs. They are designed to show
whether the extension gains are stable around the selected setting rather than
being a single-point artifact.

### PU-R-Vol Alpha Sensitivity

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/hyperparameter_sensitivity/pu_r_vol/alpha_2 \
  --dataset-name volume_gap_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Vol GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.95 \
  --boundary-augmentation 16 \
  --mu 0.35 \
  --alpha 2.0 \
  --global-redundancy-weight 1.0

python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/hyperparameter_sensitivity/pu_r_vol/alpha_4 \
  --dataset-name volume_gap_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Vol GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.95 \
  --boundary-augmentation 16 \
  --mu 0.35 \
  --alpha 4.0 \
  --global-redundancy-weight 1.0

python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/hyperparameter_sensitivity/pu_r_vol/alpha_6 \
  --dataset-name volume_gap_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Vol GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.95 \
  --boundary-augmentation 16 \
  --mu 0.35 \
  --alpha 6.0 \
  --global-redundancy-weight 1.0
```

### PU-R-Manifold Tau Sensitivity

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/hyperparameter_sensitivity/pu_r_manifold/tau_0p5 \
  --dataset-name manifold_orbit_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Manifold GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 10 \
  --rotation-angles -45 -30 -15 0 15 30 45 \
  --mu 0.5 \
  --global-redundancy-weight 0.75 \
  --manifold-k 7 \
  --manifold-tau 0.5 \
  --manifold-eigenvectors 8

python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/hyperparameter_sensitivity/pu_r_manifold/tau_0p8 \
  --dataset-name manifold_orbit_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Manifold GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 10 \
  --rotation-angles -45 -30 -15 0 15 30 45 \
  --mu 0.5 \
  --global-redundancy-weight 0.75 \
  --manifold-k 7 \
  --manifold-tau 0.8 \
  --manifold-eigenvectors 8

python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/hyperparameter_sensitivity/pu_r_manifold/tau_1p2 \
  --dataset-name manifold_orbit_fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Manifold GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 10 \
  --rotation-angles -45 -30 -15 0 15 30 45 \
  --mu 0.5 \
  --global-redundancy-weight 0.75 \
  --manifold-k 7 \
  --manifold-tau 1.2 \
  --manifold-eigenvectors 8
```

## Time-Matched Literature Ablation

This ablation tests the runtime concern directly. For each seed and noise level,
the code first runs `PU-R` to `ES=50` on Fashion-MNIST and records the
selection-loop wall-clock time `t`. It then runs each comparison method for the
same time `t`; faster methods may therefore reach more than 50 selected steps.

The timer covers the P2L selection/update loop only. Shared pretraining, final
test evaluation, certificate computation, diagnostics, CSV writing, and plotting
are outside the time budget. This makes the result a selector-efficiency
comparison, not a replacement for the fixed-ES comparisons above.

The plots include:

- risk/P2L bound vs label-noise rate
- clean test error vs label-noise rate
- effective compression size vs label-noise rate
- selected steps reached vs label-noise rate
- measured selection-loop runtime vs label-noise rate

### Fashion-MNIST, No Redundancy, Time-Matched to PU-R ES=50

```bash
python -m pu_p2l.run_time_matched_noise \
  --output-dir results/PU-R/FashionMNIST/time_matched_literature/pur_es50/no_redundancy \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.0 0.1 0.2 0.3 \
  --pretrain-fractions 0.3 \
  --reference-method PU-R \
  --reference-es-budget 50 \
  --methods MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

## Statistical Evidence Tables

These commands turn finished `results.csv` files into paired statistical tables.
They exclude `Marginal` by default and compare `PU-R` against the reportable
literature baselines. Differences are computed as `PU-R - baseline` over paired
seeds within the same dataset/noise/pretrain/ES condition. For lower-is-better
metrics such as risk, bound, compression size, noise-hit rate, and runtime,
negative differences favor `PU-R`.

```bash
python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/MNIST \
  --results-dir results/PU-R/FashionMNIST/literature_pruning \
  --results-dir results/PU-R/FashionMNIST/time_matched_literature \
  --output-dir results/PU-R/statistical_evidence/core_reportable \
  --target-method PU-R \
  --baselines MaxLoss GREATS EL2N GraNdLast RHO-PretrainRef \
  --metrics test_inappropriate_risk test_error certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate runtime_sec selection_runtime_sec
```

For extension-only evidence, run the same paired procedure with the extension
method as the target and plain `PU-R` as the baseline:

```bash
python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/FashionMNIST/extensions/pu_r_vol \
  --output-dir results/PU-R/statistical_evidence/extensions/pu_r_vol \
  --target-method PU-R-Vol \
  --baselines PU-R MaxLoss GREATS \
  --metrics test_inappropriate_risk certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate mode_entropy spectral_entropy runtime_sec

python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/FashionMNIST/extensions/pu_r_manifold \
  --output-dir results/PU-R/statistical_evidence/extensions/pu_r_manifold \
  --target-method PU-R-Manifold \
  --baselines PU-R MaxLoss GREATS \
  --metrics test_inappropriate_risk certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate mode_entropy spectral_entropy runtime_sec
```
