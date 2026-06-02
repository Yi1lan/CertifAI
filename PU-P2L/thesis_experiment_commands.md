# PU-R Thesis Experiment Commands

Run all commands from the repository root.

```bash
cd /Users/yi1lan/Desktop/CertifAI
conda activate certifai-experiments
export PYTHONPATH=PU-P2L
export DEVICE=cuda
export DATA_DIR=data
export SEEDS5="0 1 2 3 4"
```

The result hierarchy is:

```text
results/thesis_v2/
  with_marginal/
  without_marginal/
```

Every experiment below has both versions. The `with_marginal` version includes
`Marginal` or marginal-specific ablations. The `without_marginal` version
excludes them.

## Shared Settings

```bash
export MNIST_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name mnist_fcn --optimizer sgd --momentum 0.95 --batch-size 60000 --inference-batch-size 512 --pretrain-epochs 20 --p2l-epochs-per-iter 5 --pretrain-lr 0.01 --p2l-lr 0.01 --dropout-prob 0.2 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"
export IMAGE_COMMON="$MNIST_COMMON"
export CIFAR_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name cifar_resnet18 --optimizer sgd --momentum 0.9 --batch-size 256 --inference-batch-size 256 --pretrain-epochs 10 --p2l-epochs-per-iter 2 --pretrain-lr 0.01 --p2l-lr 0.01 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 64 --residual-tol 1e-6 --pac-bayes-samples 0"
export SYNTH_COMMON="--device $DEVICE --model-name small_mlp --hidden-dim 128 --optimizer adam --batch-size 256 --inference-batch-size 1024 --pretrain-epochs 30 --p2l-epochs-per-iter 2 --pretrain-lr 0.001 --p2l-lr 0.001 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"

export CORE_WITH_MARGINAL="MaxLoss Marginal PU-R GREATS"
export CORE_NO_MARGINAL="MaxLoss PU-R GREATS"
export PURVOL_WITH_MARGINAL="MaxLoss Marginal PU-R PU-R-Vol GREATS"
export PURVOL_NO_MARGINAL="MaxLoss PU-R PU-R-Vol GREATS"
export MANIFOLD_WITH_MARGINAL="MaxLoss Marginal PU-R PU-R-Vol PU-R-Manifold"
export MANIFOLD_NO_MARGINAL="MaxLoss PU-R PU-R-Vol PU-R-Manifold"
export ABLATION_WITH_MARGINAL="MaxLoss ClippedLoss Loss+Residual Loss-Redundancy PU-R PU-C-style Marginal Marginal+Residual Marginal-Redundancy Marginal+Residual-Redundancy"
export ABLATION_NO_MARGINAL="MaxLoss ClippedLoss Loss+Residual Loss-Redundancy PU-R PU-C-style"
```

## 1. Binary MNIST Core Certificate Comparison

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/binary_mnist/core_boundary --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods $CORE_WITH_MARGINAL --n-train 1000 --n-test 10000 --max-total-support 300
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/binary_mnist/core_boundary --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods $CORE_NO_MARGINAL --n-train 1000 --n-test 10000 --max-total-support 300
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/with_marginal/binary_mnist/core_es_trace --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R --record-every 2 --n-train 1000 --n-test 10000 --max-total-support 300
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/without_marginal/binary_mnist/core_es_trace --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss PU-R --record-every 2 --n-train 1000 --n-test 10000 --max-total-support 300
```

## 2. Literature Generalization-Bound Comparison

```bash
python -m pu_p2l.run_generalization_bounds --output-dir results/thesis_v2/with_marginal/binary_mnist/generalization_bounds --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods MaxLoss Marginal PU-R PU-R-Vol GREATS --n-train 1000 --n-test 10000 --max-total-support 300 --pac-bayes-samples 50 --pac-bayes-train-epochs 1 --pac-bayes-scope head
```

```bash
python -m pu_p2l.run_generalization_bounds --output-dir results/thesis_v2/without_marginal/binary_mnist/generalization_bounds --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods MaxLoss PU-R PU-R-Vol GREATS --n-train 1000 --n-test 10000 --max-total-support 300 --pac-bayes-samples 50 --pac-bayes-train-epochs 1 --pac-bayes-scope head
```

## 3. Mode-Imbalanced MNIST: PU-R vs Marginal Mechanism

```bash
for L in 0.50 0.70 0.85 0.95; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/mode_mnist/lambda_${L}/boundary --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 --methods MaxLoss Marginal PU-R Marginal+Residual-Redundancy --n-train 1000 --n-test 10000 --max-total-support 300 --mode-imbalance $L; done
```

```bash
for L in 0.50 0.70 0.85 0.95; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/mode_mnist/lambda_${L}/boundary --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 --methods MaxLoss PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --mode-imbalance $L; done
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/with_marginal/mode_mnist/lambda_0.85/es_trace --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R Marginal+Residual-Redundancy --record-every 2 --n-train 1000 --n-test 10000 --max-total-support 300 --mode-imbalance 0.85
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/without_marginal/mode_mnist/lambda_0.85/es_trace --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R --record-every 2 --n-train 1000 --n-test 10000 --max-total-support 300 --mode-imbalance 0.85
```

## 4. Boundary-Duplicate Augmentation

```bash
for R in 1 3 5 10; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/boundary_duplicate_mnist/r_${R}/boundary --dataset-name boundary_duplicate_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R Marginal+Residual-Redundancy --n-train 1000 --n-test 10000 --max-total-support 300 --mode-imbalance 0.85 --boundary-augmentation $R; done
```

```bash
for R in 1 3 5 10; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/boundary_duplicate_mnist/r_${R}/boundary --dataset-name boundary_duplicate_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --mode-imbalance 0.85 --boundary-augmentation $R; done
```

## 5. PU-R Ablation

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/binary_mnist/ablation --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $ABLATION_WITH_MARGINAL --n-train 1000 --n-test 10000 --max-total-support 300
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/binary_mnist/ablation --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $ABLATION_NO_MARGINAL --n-train 1000 --n-test 10000 --max-total-support 300
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/fashion_mnist/ablation --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $ABLATION_WITH_MARGINAL --n-train 2000 --n-test 10000 --max-total-support 500
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/fashion_mnist/ablation --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $ABLATION_NO_MARGINAL --n-train 2000 --n-test 10000 --max-total-support 500
```

## 6. PU-R Hyperparameter Sensitivity on Binary MNIST

```bash
for MU in 0 0.25 0.5 1.0 2.0 4.0; do for BETA in 0 0.25 0.5 1.0 2.0; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/binary_mnist/sensitivity/mu_${MU}_beta_${BETA} --dataset-name mnist $MNIST_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --mu $MU --global-redundancy-weight $BETA; done; done
```

```bash
for MU in 0 0.25 0.5 1.0 2.0 4.0; do for BETA in 0 0.25 0.5 1.0 2.0; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/binary_mnist/sensitivity/mu_${MU}_beta_${BETA} --dataset-name mnist $MNIST_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --mu $MU --global-redundancy-weight $BETA; done; done
```

```bash
for C in 1.5 2.0 3.0 5.0 inf; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/binary_mnist/sensitivity/c_loss_${C} --dataset-name mnist $MNIST_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --c-loss $C; done
```

```bash
for C in 1.5 2.0 3.0 5.0 inf; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/binary_mnist/sensitivity/c_loss_${C} --dataset-name mnist $MNIST_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --c-loss $C; done
```

```bash
for RANK in 0 8 16 32 64; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/binary_mnist/sensitivity/residual_rank_${RANK} --dataset-name mnist $MNIST_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --residual-rank $RANK; done
```

```bash
for RANK in 0 8 16 32 64; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/binary_mnist/sensitivity/residual_rank_${RANK} --dataset-name mnist $MNIST_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R --n-train 1000 --n-test 10000 --max-total-support 300 --residual-rank $RANK; done
```

## 7. Noisy Binary MNIST

```bash
for RHO in 0.3 0.5; do python -m pu_p2l.run_noise --output-dir results/thesis_v2/with_marginal/binary_mnist/noise/pretrain_${RHO} --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.05 0.10 0.20 0.30 0.40 --pretrain-fraction $RHO --methods MaxLoss Marginal PU-R PU-R-Vol GREATS --n-train 1000 --n-test 10000 --max-total-support 300; done
```

```bash
for RHO in 0.3 0.5; do python -m pu_p2l.run_noise --output-dir results/thesis_v2/without_marginal/binary_mnist/noise/pretrain_${RHO} --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.05 0.10 0.20 0.30 0.40 --pretrain-fraction $RHO --methods MaxLoss PU-R PU-R-Vol GREATS --n-train 1000 --n-test 10000 --max-total-support 300; done
```

## 7b. Noisy Fashion-MNIST

```bash
for RHO in 0.3 0.5; do python -m pu_p2l.run_noise --output-dir results/thesis_v2/with_marginal/fashion_mnist/noise/pretrain_${RHO} --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.05 0.10 0.20 0.30 0.40 --pretrain-fraction $RHO --methods MaxLoss Marginal PU-R PU-R-Vol GREATS --n-train 2000 --n-test 10000 --max-total-support 500; done
```

```bash
for RHO in 0.3 0.5; do python -m pu_p2l.run_noise --output-dir results/thesis_v2/without_marginal/fashion_mnist/noise/pretrain_${RHO} --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.05 0.10 0.20 0.30 0.40 --pretrain-fraction $RHO --methods MaxLoss PU-R PU-R-Vol GREATS --n-train 2000 --n-test 10000 --max-total-support 500; done
```

## 8. MNIST 10-Class

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/mnist10/boundary --dataset-name mnist10 $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R --n-train 2000 --n-test 10000 --max-total-support 500
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/mnist10/boundary --dataset-name mnist10 $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss PU-R --n-train 2000 --n-test 10000 --max-total-support 500
```

## 9. Fashion-MNIST PU-R / PU-R-Vol

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/fashion_mnist/pur_vol_boundary --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R PU-R-Vol --n-train 2000 --n-test 10000 --max-total-support 500 --alpha 1.0
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/fashion_mnist/pur_vol_boundary --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss PU-R PU-R-Vol --n-train 2000 --n-test 10000 --max-total-support 500 --alpha 1.0
```

```bash
for A in 0 0.5 1.0 2.0 4.0; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/fashion_mnist/alpha_${A} --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R PU-R-Vol --n-train 2000 --n-test 10000 --max-total-support 500 --alpha $A; done
```

```bash
for A in 0 0.5 1.0 2.0 4.0; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/fashion_mnist/alpha_${A} --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Vol --n-train 2000 --n-test 10000 --max-total-support 500 --alpha $A; done
```

```bash
for MU in 0 0.5 1.0 2.0; do for BETA in 0 0.5 1.0 2.0; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/fashion_mnist/sensitivity/mu_${MU}_beta_${BETA} --dataset-name fashion_mnist $IMAGE_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R PU-R-Vol --n-train 2000 --n-test 10000 --max-total-support 500 --mu $MU --global-redundancy-weight $BETA; done; done
```

```bash
for MU in 0 0.5 1.0 2.0; do for BETA in 0 0.5 1.0 2.0; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/fashion_mnist/sensitivity/mu_${MU}_beta_${BETA} --dataset-name fashion_mnist $IMAGE_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Vol --n-train 2000 --n-test 10000 --max-total-support 500 --mu $MU --global-redundancy-weight $BETA; done; done
```

## 10. Rotated-MNIST PU-R-Manifold

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/rotated_mnist/manifold_boundary --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R PU-R-Vol PU-R-Manifold --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -60 -30 0 30 60 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/rotated_mnist/manifold_boundary --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss PU-R PU-R-Vol PU-R-Manifold --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -60 -30 0 30 60 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/with_marginal/rotated_mnist/manifold_es_trace --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Manifold --record-every 5 --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -60 -30 0 30 60 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/without_marginal/rotated_mnist/manifold_es_trace --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R PU-R-Manifold --record-every 5 --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -60 -30 0 30 60 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
for K in 3 5 10; do for TAU in 0.05 0.1 0.2 0.5; do for EIG in 4 8 16; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/rotated_mnist/manifold_sweep/k_${K}_tau_${TAU}_eig_${EIG} --dataset-name rotated_mnist $IMAGE_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss PU-R PU-R-Manifold --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -60 -30 0 30 60 --manifold-k $K --manifold-tau $TAU --manifold-eigenvectors $EIG; done; done; done
```

```bash
for K in 3 5 10; do for TAU in 0.05 0.1 0.2 0.5; do for EIG in 4 8 16; do python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/rotated_mnist/manifold_sweep/k_${K}_tau_${TAU}_eig_${EIG} --dataset-name rotated_mnist $IMAGE_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Manifold --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -60 -30 0 30 60 --manifold-k $K --manifold-tau $TAU --manifold-eigenvectors $EIG; done; done; done
```

## 11. Synthetic Two-Moons Manifold Diagnostic

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/with_marginal/two_moons/manifold_es_trace --dataset-name two_moons $SYNTH_COMMON --seeds $SEEDS5 --noise-rates 0.05 --pretrain-fractions 0.1 0.3 0.5 --methods MaxLoss Marginal PU-R PU-R-Manifold --record-every 2 --n-train 1000 --n-test 5000 --cluster-std 0.10 --max-total-support 300 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/thesis_v2/without_marginal/two_moons/manifold_es_trace --dataset-name two_moons $SYNTH_COMMON --seeds $SEEDS5 --noise-rates 0.05 --pretrain-fractions 0.1 0.3 0.5 --methods MaxLoss PU-R PU-R-Manifold --record-every 2 --n-train 1000 --n-test 5000 --cluster-std 0.10 --max-total-support 300 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

## 12. Optional CIFAR-10 Reduced PU-R-Vol

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/cifar10_reduced/pur_vol_boundary --dataset-name cifar10 $CIFAR_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R PU-R-Vol GREATS --n-train 2000 --n-test 10000 --max-total-support 500 --alpha 1.0
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/cifar10_reduced/pur_vol_boundary --dataset-name cifar10 $CIFAR_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss PU-R PU-R-Vol GREATS --n-train 2000 --n-test 10000 --max-total-support 500 --alpha 1.0
```

## 13. Optional Rotated Fashion-MNIST

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/with_marginal/rotated_fashion_mnist/manifold_boundary --dataset-name rotated_fashion_mnist $IMAGE_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R PU-R-Manifold --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -45 -30 -15 0 15 30 45 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_boundary --output-dir results/thesis_v2/without_marginal/rotated_fashion_mnist/manifold_boundary --dataset-name rotated_fashion_mnist $IMAGE_COMMON --seeds 0 1 2 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss PU-R PU-R-Manifold --n-train 2000 --n-test 10000 --max-total-support 500 --rotation-angles -45 -30 -15 0 15 30 45 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```
