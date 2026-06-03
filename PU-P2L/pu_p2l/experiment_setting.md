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

## Setup

```bash
conda activate certifai-experiments

export PYTHONPATH=PU-P2L
export DEVICE=cuda
export DATA_DIR=data
export SEEDS5="0 1 2 3 4"

export N_MNIST=5000
export N_TEST=10000
export SUPPORT_MNIST=800

export MNIST_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name mnist_fcn --optimizer sgd --momentum 0.95 --batch-size 60000 --inference-batch-size 1024 --pretrain-epochs 20 --p2l-epochs-per-iter 5 --pretrain-lr 0.01 --p2l-lr 0.01 --dropout-prob 0.2 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"

export CORE_METHODS="MaxLoss Marginal PU-R GREATS"
export TRACE_METHODS="MaxLoss Marginal PU-R"
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
