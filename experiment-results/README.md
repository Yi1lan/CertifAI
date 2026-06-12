# Experiment Result Audit Bundle

This folder contains the plots and source result files used by the evaluation section of the final report. It is intended as a compact audit bundle: the PDF can be checked against the figures in `figures/`, and the numeric tables can be checked against the CSV and statistical-evidence files in `tables/`.

The full experiment command catalogue remains in `PU-P2L/pu_p2l/experiment_setting.md`. The commands below are the specific commands needed to regenerate the results reported in the evaluation chapter.

## Folder Layout

- `figures/`: copied PNG files used by the reported evaluation figures.
- `tables/`: copied CSV/TeX files used to construct the reported numeric tables.
- `configs/`: copied JSON configs from the corresponding result folders.


## Common Setup

Run commands from the repository root.

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
export PRETRAIN_TRAINING_MODE=warm_start

export MNIST_COMMON="--data-dir $DATA_DIR --download-data --device $DEVICE --model-name mnist_fcn --optimizer sgd --momentum 0.95 --batch-size 60000 --inference-batch-size 1024 --pretrain-training-mode $PRETRAIN_TRAINING_MODE --pretrain-epochs 20 --p2l-epochs-per-iter 5 --pretrain-lr 0.01 --p2l-lr 0.01 --dropout-prob 0.2 --initial-per-class 2 --mu 1.0 --global-redundancy-weight 1.0 --residual-rank 0 --residual-tol 1e-6 --pac-bayes-samples 0"
export IMAGE_COMMON="$MNIST_COMMON"
export PRETRAIN_GRID="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"
```

The main result tables were generated under the historical default pretraining semantics, `pretrain_training_mode=warm_start`: the pretrain subset warms the model before the P2L loop, is removed from the certification pool, and is not replayed in iterative support-training updates. Extension confirmation runs that explicitly record or pass `pretrain_training_mode=support` use the P2L-aligned support semantics.

Unless otherwise stated, P2L and P2L-ES certificates use `--delta 0.035`, corresponding to confidence `96.5%`. Statistical evidence tables use paired `95%` confidence intervals over matched seeds and conditions.

## Figure and Table Mapping

| Report result label | Audit files in this folder |
| --- | --- |
| Table `tab:time_matched_noise03` | `tables/table_time_matched_noise03.csv`, `configs/time_matched_fashion_mnist_config.json` |
| Figure `fig:time_matched_risk` | `figures/time_matched_clean_test_error_vs_noise.png`, `figures/time_matched_bounds_vs_noise.png` |
| Table `tab:pruning_budget_noise03` | `tables/table_pruning_noise03_es50.csv`, `tables/table_pruning_noise03_es100.csv`, `tables/table_pruning_noise03_es200.csv` |
| Figure `fig:pruning_es200_noise03` | `figures/pruning_es200_noise_0p3_bound_risk.png` |
| Table `tab:mnist_core_pretrain03` | `tables/table_mnist_core_no_es_noise03.csv`, `tables/table_mnist_core_es100_noise03.csv` |
| Figures `fig:mnist_noes_noise03_bound`, `fig:mnist_noes_noise03_diag` | `figures/mnist_no_es_noise_0p3_bound_risk.png`, `figures/mnist_no_es_noise_0p3_selection_diagnostics.png` |
| Table `tab:mnist_redundant_pretrain03` | `tables/table_mnist_redundant_no_es_noise03.csv`, `tables/table_mnist_redundant_es100_noise03.csv` |
| Figure `fig:mnist_es_redundant_bound` | `figures/mnist_es100_noise_0p3_redundant_bound_risk.png` |
| Figure `fig:mnist_noise03_combined` | `figures/mnist_noise_0p3_literature_bounds.png`, `figures/mnist_noise_0p3_redundant_literature_bounds.png` |
| Table `tab:pur_manifold_confirmation` | `tables/table_pur_manifold_confirmation.csv`, `tables/stat_pur_manifold_paired_overall.csv` |
| Figures `fig:pur_manifold_es100_bound`, `fig:pur_manifold_es100_diag` | `figures/pu_r_manifold_es100_bound_risk.png`, `figures/pu_r_manifold_es100_selection_diagnostics.png` |
| Table `tab:pur_vol_negative_evidence` | `tables/stat_pur_vol_paired_overall.csv`, `tables/stat_pur_vol_paired_by_condition.csv` |
| Figure `fig:pur_vol_negative_evidence` | `figures/pu_r_vol_negative_evidence_es100.png` |
| Figure `fig:ablation_combined_hyperparams` | `figures/global_redundancy_weight_bound_risk_noise_0p3_pretrain_0p3.png`, `figures/mu_bound_risk_noise_0p3_pretrain_0p3.png` |
| Table `tab:selection_visualisation_noisehit` | `tables/table_selection_visualisation_new_selection_mean_sd.csv` |
| Figure `fig:selection_visualisation_manifold` | `figures/selection_best_pu_r_seed_7_budget_100.png` |

## Regeneration Commands

### Time-Matched Fashion-MNIST

Generates Table `tab:time_matched_noise03` and Figure `fig:time_matched_risk`.

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

### Fixed-Budget Fashion-MNIST Pruning Comparison

Generates Table `tab:pruning_budget_noise03` and Figure `fig:pruning_es200_noise03`. Run all three ES budgets.

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

### Core MNIST Noise Comparison

Generates Table `tab:mnist_core_pretrain03` and Figures `fig:mnist_noes_noise03_bound` and `fig:mnist_noes_noise03_diag`.

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

### Redundant MNIST Boundary Comparison

Generates Table `tab:mnist_redundant_pretrain03` and Figure `fig:mnist_es_redundant_bound`.

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

### MNIST Literature-Bound Comparison

Generates Figure `fig:mnist_noise03_combined`.

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

### PU-R-Manifold Confirmation

Generates Table `tab:pur_manifold_confirmation` and Figures `fig:pur_manifold_es100_bound` and `fig:pur_manifold_es100_diag`.

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

### PU-R-Vol Negative Evidence

Generates Table `tab:pur_vol_negative_evidence` and Figure `fig:pur_vol_negative_evidence`.

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

### PU-R Hyperparameter Ablation

Generates Figure `fig:ablation_combined_hyperparams`.

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

### Selection Visualisation and New-Selection Diagnostics

Generates Table `tab:selection_visualisation_noisehit` and Figure `fig:selection_visualisation_manifold`.

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

## Copying Regenerated Artifacts Into This Bundle

After regenerating results, copy the relevant reportable plots, CSVs, and configs into this folder using the filenames listed in the mapping table above.

The copied files in this audit bundle should match the figures and table values used in the PDF.
