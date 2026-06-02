# PU-R Experiment Commands

Run from the repository root. Each experiment trains once with `Marginal`
included where relevant. The runners then write both table/plot views:

- `summary_with_marginal.csv`, `tables/with_marginal.csv`, `plots/with_marginal/`
- `summary_without_marginal.csv`, `tables/without_marginal.csv`, `plots/without_marginal/`

```bash
conda activate certifai-experiments
export PYTHONPATH=PU-P2L
export DEVICE=cuda
export DATA_DIR=data
export SEEDS5="0 1 2 3 4"
export SEEDS3="0 1 2"

export N_MNIST=5000
export N_IMAGE=5000
export N_CIFAR=5000
export N_SYNTH=3000
export N_SENS=3000
export N_TEST=10000
export SUPPORT_MNIST=800
export SUPPORT_IMAGE=1000
export SUPPORT_SYNTH=800
export SUPPORT_SENS=600

export MNIST_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name mnist_fcn --optimizer sgd --momentum 0.95 --batch-size 60000 --inference-batch-size 1024 --pretrain-epochs 20 --p2l-epochs-per-iter 5 --pretrain-lr 0.01 --p2l-lr 0.01 --dropout-prob 0.2 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"
export IMAGE_COMMON="$MNIST_COMMON"
export CIFAR_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name cifar_resnet18 --optimizer sgd --momentum 0.9 --batch-size 256 --inference-batch-size 512 --pretrain-epochs 10 --p2l-epochs-per-iter 2 --pretrain-lr 0.01 --p2l-lr 0.01 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 64 --residual-tol 1e-6 --pac-bayes-samples 0"
export SYNTH_COMMON="--device $DEVICE --model-name small_mlp --hidden-dim 128 --optimizer adam --batch-size 256 --inference-batch-size 2048 --pretrain-epochs 30 --p2l-epochs-per-iter 2 --pretrain-lr 0.001 --p2l-lr 0.001 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"

export CORE_METHODS="MaxLoss Marginal PU-R GREATS"
export PURVOL_METHODS="MaxLoss Marginal PU-R PU-R-Vol GREATS"
export MANIFOLD_METHODS="MaxLoss Marginal PU-R PU-R-Vol PU-R-Manifold"
export ABLATION_METHODS="MaxLoss ClippedLoss Loss+Residual Loss-Redundancy PU-R PU-C-style Marginal Marginal+Residual Marginal-Redundancy Marginal+Residual-Redundancy"
export LIT_METHODS="MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R GREATS"
export LIT_CORE_METHODS="MaxLoss Marginal EL2N GraNdLast RHO-PretrainRef PU-R PU-R-Vol GREATS"
```

## Core PU-R Experiments

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/binary_mnist/core_boundary --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods $CORE_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/experiments/pu_r/binary_mnist/core_es_trace --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R --record-every 2 --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST
```

```bash
python -m pu_p2l.run_generalization_bounds --output-dir results/experiments/pu_r/binary_mnist/generalization_bounds --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods MaxLoss Marginal PU-R PU-R-Vol GREATS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --pac-bayes-samples 50 --pac-bayes-train-epochs 1 --pac-bayes-scope head
```

```bash
for L in 0.50 0.70 0.85 0.95; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/mode_mnist/lambda_${L}/boundary --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 --methods MaxLoss Marginal PU-R Marginal+Residual-Redundancy --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --mode-imbalance $L; done
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/experiments/pu_r/mode_mnist/lambda_0.85/es_trace --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R Marginal+Residual-Redundancy --record-every 2 --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --mode-imbalance 0.85
```

```bash
for R in 1 3 5 10; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/boundary_duplicate_mnist/r_${R}/boundary --dataset-name boundary_duplicate_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R Marginal+Residual-Redundancy --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --mode-imbalance 0.85 --boundary-augmentation $R; done
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/binary_mnist/ablation --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $ABLATION_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/fashion_mnist/ablation --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $ABLATION_METHODS --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE
```

```bash
for MU in 0 0.25 0.5 1.0 2.0 4.0; do for BETA in 0 0.25 0.5 1.0 2.0; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/binary_mnist/sensitivity/mu_${MU}_beta_${BETA} --dataset-name mnist $MNIST_COMMON --seeds $SEEDS3 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R --n-train $N_SENS --n-test $N_TEST --max-total-support $SUPPORT_SENS --mu $MU --global-redundancy-weight $BETA; done; done
```

```bash
for C in 1.5 2.0 3.0 5.0 inf; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/binary_mnist/sensitivity/c_loss_${C} --dataset-name mnist $MNIST_COMMON --seeds $SEEDS3 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R --n-train $N_SENS --n-test $N_TEST --max-total-support $SUPPORT_SENS --c-loss $C; done
```

```bash
for RANK in 0 8 16 32 64; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/binary_mnist/sensitivity/residual_rank_${RANK} --dataset-name mnist $MNIST_COMMON --seeds $SEEDS3 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R --n-train $N_SENS --n-test $N_TEST --max-total-support $SUPPORT_SENS --residual-rank $RANK; done
```

```bash
for RHO in 0.3 0.5; do python -m pu_p2l.run_noise --output-dir results/experiments/pu_r/binary_mnist/noise/pretrain_${RHO} --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.05 0.10 0.20 0.30 0.40 --pretrain-fraction $RHO --methods $PURVOL_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST; done
```

```bash
for RHO in 0.3 0.5; do python -m pu_p2l.run_noise --output-dir results/experiments/pu_r/fashion_mnist/noise/pretrain_${RHO} --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.05 0.10 0.20 0.30 0.40 --pretrain-fraction $RHO --methods $PURVOL_METHODS --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE; done
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/mnist10/boundary --dataset-name mnist10 $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/fashion_mnist/pur_vol_boundary --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R PU-R-Vol --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --alpha 1.0
```

```bash
for A in 0 0.5 1.0 2.0 4.0; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/fashion_mnist/alpha_${A} --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Vol --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --alpha $A; done
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/rotated_mnist/manifold_boundary --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods $MANIFOLD_METHODS --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --rotation-angles -60 -30 0 30 60 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/experiments/pu_r/rotated_mnist/manifold_es_trace --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Manifold --record-every 5 --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --rotation-angles -60 -30 0 30 60 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
for K in 3 5 10; do for TAU in 0.05 0.1 0.2 0.5; do for EIG in 4 8 16; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/rotated_mnist/manifold_sweep/k_${K}_tau_${TAU}_eig_${EIG} --dataset-name rotated_mnist $IMAGE_COMMON --seeds $SEEDS3 --noise-rates 0.0 --pretrain-fractions 0.3 --methods MaxLoss Marginal PU-R PU-R-Manifold --n-train $N_SENS --n-test $N_TEST --max-total-support $SUPPORT_SENS --rotation-angles -60 -30 0 30 60 --manifold-k $K --manifold-tau $TAU --manifold-eigenvectors $EIG; done; done; done
```

```bash
python -m pu_p2l.run_es_trace --output-dir results/experiments/pu_r/two_moons/manifold_es_trace --dataset-name two_moons $SYNTH_COMMON --seeds $SEEDS5 --noise-rates 0.05 --pretrain-fractions 0.1 0.3 0.5 --methods MaxLoss Marginal PU-R PU-R-Manifold --record-every 2 --n-train $N_SYNTH --n-test $N_TEST --cluster-std 0.10 --max-total-support $SUPPORT_SYNTH --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/cifar10_reduced/pur_vol_boundary --dataset-name cifar10 $CIFAR_COMMON --seeds $SEEDS3 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods $PURVOL_METHODS --n-train $N_CIFAR --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --alpha 1.0
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/rotated_fashion_mnist/manifold_boundary --dataset-name rotated_fashion_mnist $IMAGE_COMMON --seeds $SEEDS3 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods MaxLoss Marginal PU-R PU-R-Manifold --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --rotation-angles -45 -30 -15 0 15 30 45 --manifold-k 5 --manifold-tau 0.2 --manifold-eigenvectors 8
```

## Literature Pruning Baselines

```bash
for L in 0.85; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/literature_baselines/mode_mnist/lambda_${L}/boundary --dataset-name mode_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods $LIT_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --mode-imbalance $L; done
```

```bash
for R in 3 5 10; do python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/literature_baselines/boundary_duplicate_mnist/r_${R}/boundary --dataset-name boundary_duplicate_mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 --methods $LIT_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --mode-imbalance 0.85 --boundary-augmentation $R; done
```

```bash
python -m pu_p2l.run_noise --output-dir results/experiments/pu_r/literature_baselines/binary_mnist/noise/pretrain_0.3 --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.10 0.20 0.40 --pretrain-fraction 0.3 --methods $LIT_CORE_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST
```

```bash
python -m pu_p2l.run_noise --output-dir results/experiments/pu_r/literature_baselines/fashion_mnist/noise/pretrain_0.3 --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 0.10 0.20 0.40 --pretrain-fraction 0.3 --methods $LIT_CORE_METHODS --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE
```

```bash
python -m pu_p2l.run_boundary --output-dir results/experiments/pu_r/literature_baselines/fashion_mnist/core --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.3 0.5 --methods $LIT_CORE_METHODS --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --alpha 1.0
```

```bash
python -m pu_p2l.run_generalization_bounds --output-dir results/experiments/pu_r/literature_bounds/binary_mnist/generalization_bounds --dataset-name mnist $MNIST_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --methods $LIT_CORE_METHODS --n-train $N_MNIST --n-test $N_TEST --max-total-support $SUPPORT_MNIST --pac-bayes-samples 50 --pac-bayes-train-epochs 1 --pac-bayes-scope head
```

```bash
python -m pu_p2l.run_generalization_bounds --output-dir results/experiments/pu_r/literature_bounds/fashion_mnist/generalization_bounds --dataset-name fashion_mnist $IMAGE_COMMON --seeds $SEEDS5 --noise-rates 0.0 --pretrain-fractions 0.1 0.3 0.5 0.7 --methods $LIT_CORE_METHODS --n-train $N_IMAGE --n-test $N_TEST --max-total-support $SUPPORT_IMAGE --pac-bayes-samples 50 --pac-bayes-train-epochs 1 --pac-bayes-scope head
```
