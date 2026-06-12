# Final Experiment Runbook

This file lists the commands needed to regenerate the experiments reported in the final evaluation. It intentionally omits development-only and historical settings. The compact result audit bundle is in `experiment-results/`.

Run all commands from the repository root.

## Common Setup

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
export PRETRAIN_GRID="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
export PRETRAIN_TRAINING_MODE=warm_start

export MNIST_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name mnist_fcn --optimizer sgd --momentum 0.95 --batch-size 60000 --inference-batch-size 1024 --pretrain-training-mode $PRETRAIN_TRAINING_MODE --pretrain-epochs 20 --p2l-epochs-per-iter 5 --pretrain-lr 0.01 --p2l-lr 0.01 --dropout-prob 0.2 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"
export IMAGE_COMMON="$MNIST_COMMON"
```

The main reported result bundle uses `PRETRAIN_TRAINING_MODE=warm_start`, the historical default. In this mode, the pretrain subset is used once to warm-start the model before the P2L loop, is removed from the certification pool, and is not used again in the iterative support-training updates. Set `PRETRAIN_TRAINING_MODE=support` for original-P2L-aligned reruns where the pretrain subset is replayed during iterative support training without being charged to the reported compression size.

Unless overridden, P2L and P2L-ES certificates use `--delta 0.035`, corresponding to confidence `96.5%`.

## 1. Core MNIST: No ES

### 1.1 Noisy MNIST, No Redundancy

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

### 1.2 Noisy MNIST, Boundary Redundancy

```bash
python -m pu_p2l.run_boundary \
  --output-dir results/PU-R/MNIST/no_es/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --methods MaxLoss PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5
```

## 2. Core MNIST: ES=100

### 2.1 Noisy MNIST, No Redundancy

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/MNIST/es100/mnist_noise_0p3 \
  --dataset-name mnist \
  $MNIST_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods MaxLoss PU-R GREATS \
  --n-train $N_MNIST \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_MNIST
```

### 2.2 Noisy MNIST, Boundary Redundancy

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

## 3. MNIST Generalization-Bound Comparison

These commands compare PU-R with GREATS and additional implemented generalization-bound diagnostics. GREATS is included as an empirical reference, but it is not assigned a P2L compression certificate.

### 3.1 Noisy MNIST, No Redundancy

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

### 3.2 Noisy MNIST, Boundary Redundancy

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

## 4. Fashion-MNIST Data-Pruning Comparison

The comparison methods are:

```text
MaxLoss, EL2N, GraNdLast, RHO-PretrainRef, PU-R, GREATS
```

### 4.1 Time-Matched Noise Sweep

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
  --methods MaxLoss EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

### 4.2 Fixed ES Budgets at Noise 0.3

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es50/noise_0p3 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 50 \
  --methods MaxLoss EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE

python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es100/noise_0p3 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 100 \
  --methods MaxLoss EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE

python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/literature_pruning/es200/noise_0p3 \
  --dataset-name fashion_mnist \
  $IMAGE_COMMON \
  --seeds $SEEDS5 \
  --noise-rates 0.3 \
  --pretrain-fractions $PRETRAIN_GRID \
  --es-budgets 200 \
  --methods MaxLoss EL2N GraNdLast RHO-PretrainRef PU-R GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE
```

## 5. PU-R-Manifold Confirmation

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready \
  --dataset-name manifold_group_noise_fashion_mnist \
  $IMAGE_COMMON \
  --pretrain-training-mode support \
  --seeds 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.0 \
  --es-budgets 100 200 \
  --methods MaxLoss PU-R PU-R-Manifold GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 12 \
  --rotation-angles -45 -30 -15 0 15 30 45 \
  --mu 0.45 \
  --global-redundancy-weight 1.1 \
  --manifold-k 7 \
  --manifold-tau 0.5 \
  --manifold-eigenvectors 8

python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/FashionMNIST/extension_confirmation/pu_r_manifold_graph_ready \
  --output-dir results/PU-R/statistical_evidence/extension_confirmation/pu_r_manifold_graph_ready \
  --target-method PU-R-Manifold \
  --baselines PU-R MaxLoss GREATS \
  --metrics test_inappropriate_risk certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate group_revisit_rate unique_group_fraction runtime_sec
```

## 6. PU-R-Vol Negative Evidence

```bash
python -m pu_p2l.run_es_budget_boundary \
  --output-dir results/PU-R/FashionMNIST/extensions_v4/pu_r_vol/volume_group_noise_0p1_low_support \
  --dataset-name volume_group_noise_fashion_mnist \
  $IMAGE_COMMON \
  --pretrain-training-mode support \
  --seeds $SEEDS5 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.0 0.05 0.1 \
  --es-budgets 50 100 200 \
  --methods MaxLoss PU-R PU-R-Vol GREATS \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.98 \
  --boundary-augmentation 32 \
  --mu 0.35 \
  --alpha 2.0 \
  --global-redundancy-weight 1.0

python -m pu_p2l.run_statistical_evidence \
  --results-dir results/PU-R/FashionMNIST/extensions_v4/pu_r_vol \
  --output-dir results/PU-R/statistical_evidence/extensions_v4_focus/pu_r_vol_cold_start \
  --target-method PU-R-Vol \
  --baselines PU-R MaxLoss GREATS \
  --row-filter pretrain_fraction=0.0 \
  --metrics test_inappropriate_risk certified_bound effective_compression_size noise_hit_rate duplicate_hit_rate group_revisit_rate unique_group_fraction mode_entropy spectral_entropy runtime_sec
```

## 7. PU-R Hyperparameter Ablation

```bash
python -m pu_p2l.run_pu_r_hyperparameter_ablation \
  --output-dir results/PU-R/FashionMNIST/ablations/pu_r_hyperparameters/boundary_duplicate_noise_0p3_aug5 \
  --dataset-name boundary_duplicate_fashion_mnist \
  $IMAGE_COMMON \
  --pretrain-training-mode support \
  --seeds 0 1 2 \
  --noise-rates 0.3 \
  --pretrain-fractions 0.3 \
  --es-budgets 50 100 200 \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.85 \
  --boundary-augmentation 5 \
  --mu-values 0.0 0.25 0.5 1.0 1.5 \
  --redundancy-weight-values 0.0 0.5 1.0 1.5 2.0 \
  --residual-rank-values 0 32 64 128
```

## 8. Selection Visualisation and New-Selection Diagnostics

```bash
python -m pu_p2l.run_selection_visualization \
  --output-dir results/PU-R/FashionMNIST/ablations/selection_visualization/manifold_group_noise_0p1_aug12_embedding_pca_multiseed \
  --dataset-name manifold_group_noise_fashion_mnist \
  $IMAGE_COMMON \
  --pretrain-training-mode support \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-rates 0.1 \
  --pretrain-fractions 0.0 \
  --methods MaxLoss GREATS PU-R PU-R-Manifold \
  --budgets 50 100 200 \
  --projection pca \
  --projection-source embedding \
  --background-limit 6000 \
  --n-train $N_IMAGE \
  --n-test $N_TEST \
  --max-total-support $SUPPORT_IMAGE \
  --mode-imbalance 0.90 \
  --boundary-augmentation 12 \
  --rotation-angles -45 -30 -15 0 15 30 45 \
  --mu 0.45 \
  --global-redundancy-weight 1.1 \
  --manifold-k 7 \
  --manifold-tau 0.5 \
  --manifold-eigenvectors 8 \
  --no-plots

python -m pu_p2l.run_selection_report \
  --input-dir results/PU-R/FashionMNIST/ablations/selection_visualization/manifold_group_noise_0p1_aug12_embedding_pca_multiseed \
  --output-dir results/PU-R/FashionMNIST/ablations/selection_visualization/manifold_group_noise_0p1_aug12_embedding_pca_multiseed/report \
  --methods MaxLoss GREATS PU-R PU-R-Manifold \
  --target-methods PU-R \
  --example-budget 100 \
  --best-metric noise_hit_rate \
  --best-direction min
```

## Auditing Reported Results

The report-ready subset of figures, tables, and JSON configs is copied to:
```text
experiment-results/
```
See `experiment-results/README.md` for the mapping between report figure/table labels and copied audit files.
