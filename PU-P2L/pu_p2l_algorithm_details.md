# PU-P2L Algorithm Details

This document is the implementation reference for the cleaned `PU-P2L`
package. It describes the meta-algorithm, the P2L early-stopping certificate,
and the score functions currently implemented in code.

The current valid methods are:

- `MaxLoss`
- `Marginal`
- `PU-R`
- `PU-R-Vol`
- `PU-R-Manifold`
- `GREATS`

The old `PU-C`, `PU-F`, and `PU-G` selectors have been removed from the clean
package. Their weakness is discussed below because it motivated the residual
PU-R family.

Source-code mapping:

- Score functions: `pu_p2l/scores.py`
- Method dispatch: `pu_p2l/runner.py::choose_next`
- Method registry: `pu_p2l/runner.py::METHODS`
- CLI score hyperparameters: `pu_p2l/run_boundary.py::add_score_args`
- Plot order and colors: `pu_p2l/plotting.py`

## 1. Notation

Let:

- `S = {(x_i, y_i)}_{i=1}^n` be the certification pool.
- `P` be the optional pretraining set.
- `T_t subset {1,...,n}` be the support/compression set after selection step
  `t`.
- `h_t` be the current model after training on `P` and the current support set.
- `gamma > 0` be the P2L loss threshold. In the experiments the default is
  `gamma = -log(0.5)`.
- `ell_i(h_t) = CE(h_t(x_i), y_i)` be the cross-entropy loss.
- `B_t = {i notin T_t : ell_i(h_t) > gamma}` be the current inappropriate or
  bad set.
- `phi_t(x_i)` be the penultimate-layer embedding of sample `i`.
- `v_i = phi_t(x_i) / max(||phi_t(x_i)||_2, 1e-12)` be the normalized
  embedding used by the code.
- `p_i = softmax(h_t(x_i))` be the model probability vector.
- `margin_i = top1(p_i) - top2(p_i)`.

The selector always chooses from the current bad set `B_t`, not from all
non-support samples. Points already in `T_t` are excluded from future
selection.

## 2. Original P2L Meta-Algorithm

The implementation follows the standard P2L selection loop.

```text
Input:
  certification pool S
  optional pretraining set P
  deterministic initial support rule
  loss threshold gamma
  maximum support cap max_total_support
  deterministic selector score q(i; T_t, h_t, S)

Initialize:
  Train h on P if P is nonempty.
  T_0 = deterministic class-balanced initial support from S.

For t = 0, 1, 2, ...
  Train/update h_t on T_t for p2l_epochs_per_iter epochs.

  Compute losses on all non-support samples:
    ell_i = CE(h_t(x_i), y_i), i notin T_t

  Define:
    B_t = {i notin T_t : ell_i > gamma}

  If B_t is empty:
    Stop. The compression set is T_t.

  If |T_t| >= max_total_support:
    Stop by safety cap. The true P2L Stop was not reached.

  Select:
    i_t = argmax_{i in B_t} q(i; T_t, h_t, S)

  Update:
    T_{t+1} = T_t union {i_t}
```

The exact selector changes by method, but the loop does not.

## 3. P2L and P2L-ES Certificates

If true Stop is reached, then `B_t` is empty. The compression size is:

```text
k = |T_t|
```

The P2L bound is then:

```text
epsilon = p2l_bound(k, n, delta)
```

where `n = |S|`. The code computes this in `pu_p2l/bounds.py`.

If Stop is not reached and we record an early-stopped model, the implementation
uses the P2L-ES diagnostic certificate:

```text
b_t = |B_t|
k_eff(t) = |T_t| + b_t
epsilon_ES(t) = p2l_bound(k_eff(t), n, delta)
```

Interpretation: the early-stopped compression object consists of the selected
support points plus the remaining inappropriate points. If Stop is later
reached, then `b_t = 0` and this reduces to the ordinary P2L certificate.

Important: the score function does not directly change the P2L formula. A new
score improves the P2L bound only if it reaches Stop with smaller `|T_t|`, or
if under early stopping it gives a smaller `|T_t| + |B_t|`.

## 4. Preferent-Property Requirement

The P2L certificate requires that the compression rule be preferent. In this
implementation, a selector is treated as preferent if the next selected point is
a deterministic function of the current state:

```text
i_t = argmax_{i in B_t} q(i; T_t, h_t, S, fixed_hyperparameters)
```

The score must not depend on:

- random minibatch candidate sampling;
- random tie-breaking;
- validation labels;
- future selected points;
- external state that changes across runs;
- stochastic graph sampling or stochastic matrix approximations.

The current `PU-R`, `PU-R-Vol`, and `PU-R-Manifold` implementations satisfy
this requirement because they score the full current bad set and use only
deterministic linear algebra from the current model and support set.

Every method uses the same deterministic tie-breaking rule:

```text
Sort candidates by:
  1. descending score
  2. ascending original sample_id
Select the first candidate.
```

This is implemented by `tie_break_argmax`.

## 5. Shared Score Quantities

For all PU-R family methods, define clipped normalized loss:

```text
bar_loss_i = clip(ell_i / gamma, 0, c_loss)
```

The default `c_loss` is `3.0`. Clipping prevents a single mislabeled or
extreme-loss point from dominating the geometry terms.

For support embeddings:

```text
V_T = row_normalize([phi_t(x_j) : j in T_t])
V_C = row_normalize([phi_t(x_i) : i in B_t])
```

The positive cosine similarity between candidate `i` and support point `j` is:

```text
K_+(i, j) = max(<v_i, v_j>, 0)
```

The base local redundancy is:

```text
R_loc(i) = max_{j in T_t} K_+(i, j)
```

If the support set is empty, the implementation sets `R_loc(i) = 0`. In normal
experiments the deterministic initial support is nonempty, so this edge case is
mainly for robustness.

## 6. Baseline Selectors

### MaxLoss

`MaxLoss` is the original P2L selector:

```text
q_MaxLoss(i) = ell_i
```

It is preferent because losses are deterministic functions of `h_t` and `S`.

### Marginal

`Marginal` selects the current bad point closest to the decision boundary:

```text
q_Marginal(i) = -margin_i
margin_i = top1(p_i) - top2(p_i)
```

A smaller margin gives a larger score. This selector is also preferent because
the margin is computed deterministically from `h_t`.

## 7. PU-R: Residual-Novelty Selector

PU-R is the base proposed selector. It separates two notions that were
collapsed in the old PU-C design:

- local redundancy: is the candidate close to one already selected support
  point?
- residual novelty: does the candidate contain a direction not spanned by the
  current support set?

### Support-Span Basis

The code computes a deterministic SVD of normalized support embeddings:

```text
V_T = row_normalize(support_embeddings)
_, singular_values, Vh = svd(V_T, full_matrices=False)
```

It keeps singular directions satisfying:

```text
singular_value > residual_tol
```

If `residual_rank > 0`, the number of kept directions is capped at
`residual_rank`. If `residual_rank = 0`, every direction above `residual_tol`
is kept.

The support-span basis is:

```text
U_t = Vh_kept^T
```

### Residual Novelty

For a normalized candidate embedding `v_i`, residual novelty is:

```text
N_res(i) = 1 - ||U_t^T v_i||_2^2
```

The value is clipped to `[0, 1]` in code. If no basis direction is kept, the
implementation sets `N_res(i) = 1`.

### Score

The PU-R score is:

```text
q_PU-R(i) = bar_loss_i + mu * N_res(i) - beta * R_loc(i)
```

where:

```text
beta = global_redundancy_weight
```

Interpretation:

- `bar_loss_i` prioritizes inappropriate high-loss points.
- `mu * N_res(i)` rewards candidates that expand the current support span.
- `beta * R_loc(i)` penalizes candidates that are locally redundant with an
  existing support point.

PU-R remains preferent because `U_t`, `N_res`, and `R_loc` are deterministic
functions of the current support embeddings and current model.

## 8. PU-R-Vol: Spectral-Entropy Volume Selector

PU-R uses constant novelty and redundancy weights. PU-R-Vol makes both weights
depend on the current support span volume.

The motivation is simple: when the support span has low spectral entropy, the
selected support points occupy a small or anisotropic part of representation
space. In that stage, residual novelty should be emphasized more and repeated
local directions should be penalized more strongly. Once the support span has
high spectral entropy, the method should return toward the base PU-R behavior.

### Spectral Entropy

The code computes singular values of the normalized support matrix:

```text
s_1, ..., s_r = singular_values(row_normalize(support_embeddings))
lambda_a = s_a^2
```

It removes eigenvalues below `residual_tol`. If fewer than two positive
eigenvalues remain, the entropy is defined as `0`.

Otherwise:

```text
p_a = lambda_a / sum_b lambda_b
H_spec(T_t) = -sum_a p_a log(p_a) / log(number_of_positive_eigenvalues)
```

Thus:

```text
0 <= H_spec(T_t) <= 1
```

Low entropy means support directions are concentrated. High entropy means the
support energy is more evenly spread across retained directions.

### Dynamic Volume Weights

The implemented dynamic coefficients are:

```text
c_t = max(alpha, 0) * (1 - H_spec(T_t))
mu_t = mu * (1 + c_t)
beta_t = beta * (1 + c_t)
```

This is important. A formula like `mu * H_spec(T_t)` would reduce novelty
pressure when entropy is low, which contradicts the intended volume-expansion
behavior. Increasing only `mu_t` is also too weak in highly redundant image
settings, because it can reward novel high-loss outliers without discouraging
near-duplicate directions. The current implementation therefore increases both
the residual-novelty reward and the local-redundancy penalty when support
volume is concentrated.

### Score

```text
q_PU-R-Vol(i) = bar_loss_i + mu_t * N_res(i) - beta_t * R_loc(i)
```

PU-R-Vol is preferent because `H_spec(T_t)` is computed deterministically from
the current support embeddings, and `mu_t` and `beta_t` are deterministic
functions of the current support set and fixed hyperparameters.

Mathematical status: PU-R-Vol does not introduce a new generalization theorem.
It uses the same P2L certificate as PU-R. Its value is empirical: it may reduce
the effective compression size by selecting more information-covering support
points earlier.

## 9. PU-R-Manifold: Support-Graph Geodesic Selector

PU-R-Manifold is a deterministic graph-based refinement of PU-R. It is designed
for cases where Euclidean residual directions in embedding space are not the
right notion of redundancy. A candidate may look Euclidean-novel while being
topologically close to the currently selected support graph.

The implementation is intentionally local to the current support set. It does
not build an all-pairs graph over the whole dataset at each step, because that
would be too expensive for MNIST/CIFAR experiments.

If the support size is at most `2`, the method falls back to ordinary `PU-R`.

### Candidate-to-Support Distance

The code first normalizes support and candidate embeddings. It then uses cosine
chord distance:

```text
D(i, j) = sqrt(max(2 - 2 * <v_i, v_j>, 0))
```

For each candidate, it keeps its `manifold_k` nearest support points and forms
direct affinities:

```text
A_direct(i, j) = exp(-D(i, j) / tau)
```

where:

```text
tau = max(manifold_tau, 1e-12)
```

Non-nearest support points receive affinity `0`. Each candidate affinity row is
normalized to sum to `1`.

### Support Graph and Laplacian

The support graph is built from support-support cosine chord distances. For
each support point, the code keeps the nearest `k` support neighbors and assigns
weights:

```text
W(a, b) = exp(-D(a, b) / tau)
```

The graph is symmetrized:

```text
W = max(W, W^T)
```

Then the normalized graph Laplacian is:

```text
D_degree(a, a) = sum_b W(a, b)
L = I - D_degree^{-1/2} W D_degree^{-1/2}
```

The eigensystem is computed deterministically:

```text
L z_a = lambda_a z_a
```

with eigenpairs sorted by increasing eigenvalue.

### Geodesic Redundancy

The code builds a heat-kernel-style diffusion matrix on the support graph:

```text
Diff = Z diag(exp(-lambda_a / tau)) Z^T
```

Negative numerical values are clipped to `0`, and each support row is
normalized by its maximum value. Candidate direct affinities are then diffused:

```text
S_geo(i, :) = A_direct_normalized(i, :) Diff
```

Geodesic redundancy is:

```text
R_geo(i) = max_j S_geo(i, j)
```

This replaces `R_loc(i)` in the final score.

### Manifold Residual Novelty

PU-R-Manifold starts from the Euclidean residual novelty `N_res(i)` computed in
PU-R. It then downweights it if the candidate is already covered by smooth
support-graph modes.

Let `Z_low` be the first `manifold_eigenvectors` nontrivial eigenvectors, where
nontrivial means:

```text
lambda_a > residual_tol
```

The candidate graph coordinates are:

```text
c_i = A_direct_normalized(i, :) Z_low
```

The smooth coverage proxy is:

```text
C_smooth(i) = clip(||c_i||_2^2 * |T_t| / number_of_kept_eigenvectors, 0, 1)
```

The manifold residual novelty is:

```text
N_geo_res(i) = clip(N_res(i) * (1 - C_smooth(i)), 0, 1)
```

If no nontrivial eigenvectors are available, the code uses:

```text
N_geo_res(i) = N_res(i)
```

### Score

```text
q_PU-R-Manifold(i) = bar_loss_i + mu * N_geo_res(i) - beta * R_geo(i)
```

PU-R-Manifold is preferent because every graph, affinity, eigendecomposition,
and diffusion quantity is computed deterministically from the current support
embeddings, current candidate embeddings, and fixed hyperparameters.

Mathematical status: this is a deterministic manifold-aware proxy, not a proof
that the P2L capacity term is uniformly tighter. It can improve the observed
P2L bound only by selecting support points that reduce the final effective
compression size.

## 10. GREATS Reference Selector

`GREATS` is retained as a non-certified reference method.

The implementation uses a deterministic probe set:

- if pretraining data exists, a class-balanced deterministic subset of
  pretraining data is used;
- otherwise, a class-balanced deterministic subset of the certification pool is
  used.

For the last linear layer, the per-sample gradient has a factorized form:

```text
grad_i proportional to embedding_i outer (p_i - onehot(y_i))
```

The implementation uses this factorization through:

```text
K_grad(i, j)
  = cosine(embedding_i, embedding_j)
    * cosine(error_i, error_j)
```

The GREATS-style score is:

```text
Utility(i) = mean_{p in Probe} K_grad(i, p)
Redundancy(i) = max_{j in T_t} max(K_grad(i, j), 0)
q_GREATS(i) = Utility(i) - lambda_redundancy * Redundancy(i)
```

The current code does not assign a P2L compression certificate to `GREATS`.
It is plotted as a risk reference and, for non-synthetic datasets, may be shown
with PAC-Bayes diagnostics. This is a deliberate experimental convention:
`GREATS` is included to compare selection quality, while the certified PU
family is kept separate.

## 11. Why PU-C, PU-F, and PU-G Were Removed

The old PU-C score used:

```text
N_local(i) = 1 - R_loc(i)
q_PU-C(i) = bar_loss_i + mu * N_local(i) - beta * R_loc(i)
```

This can be rewritten as:

```text
q_PU-C(i) = bar_loss_i + mu - (mu + beta) * R_loc(i)
```

Since `mu` is constant across candidates, PU-C was effectively max-loss with a
nearest-support redundancy penalty. It could not represent the case where a
point is locally close to one support point but still contains a useful new
direction outside the current support span.

PU-F and PU-G added label-consensus and contradiction terms, but they still
inherited the same local novelty weakness because novelty was tied to
`1 - R_loc`. They also depended on observed labels in nearby support points,
which can be unreliable under label noise. For a clean implementation and a
clear mathematical story, the project now keeps the residual PU-R family
instead.

## 12. Hyperparameter Mapping

The score-related CLI flags are:

```text
--gamma
    P2L loss threshold.

--c-loss
    Maximum value of clipped normalized loss ell_i / gamma.

--mu
    Base residual novelty weight.

--global-redundancy-weight
    beta, the local or geodesic redundancy penalty.

--alpha
    PU-R-Vol entropy boost. Higher values increase both residual novelty
    pressure and local-redundancy pressure when support spectral entropy is low.

--residual-rank
    Maximum number of SVD support-span directions for PU-R and PU-R-Vol.
    Use 0 to keep all directions above residual_tol.

--residual-tol
    Singular-value/eigenvalue tolerance used for support-span and graph
    computations.

--manifold-k
    Number of nearest support neighbors for PU-R-Manifold affinities.

--manifold-tau
    Temperature for PU-R-Manifold affinities and graph diffusion.

--manifold-eigenvectors
    Number of nontrivial graph Laplacian eigenvectors used for manifold
    residual coverage.

--lambda-redundancy
    GREATS gradient-redundancy penalty. This does not affect PU-R, PU-R-Vol,
    or PU-R-Manifold.
```

## 13. Computational Notes

For each selection step:

- `PU-R` computes candidate/support embeddings and an SVD of the support
  embedding matrix.
- `PU-R-Vol` adds singular-value entropy computation, using the same support
  embedding information.
- `PU-R-Manifold` additionally computes a support kNN graph and a support graph
  eigendecomposition.

Thus `PU-R-Manifold` is the heaviest selector. It is deterministic and valid as
a preferent score, but it can be slower when the support set becomes large.
For large image experiments, use `--max-total-support`, `--manifold-k`, and
`--manifold-eigenvectors` conservatively.

## 14. Summary

The P2L certificate depends on the effective compression size, not directly on
which score function is used. The role of the score function is to construct a
smaller or better support set while preserving the preferent property.

The current clean method family is:

```text
MaxLoss
  baseline P2L loss selector

Marginal
  deterministic boundary-proximity selector

PU-R
  hardness + residual novelty - local redundancy

PU-R-Vol
  PU-R with deterministic spectral-entropy novelty boost

PU-R-Manifold
  PU-R with deterministic support-graph geodesic redundancy and manifold
  residual coverage

GREATS
  non-certified gradient-probe reference selector
```

The certified methods in code are:

```text
MaxLoss, Marginal, PU-R, PU-R-Vol, PU-R-Manifold
```

`GREATS` is intentionally treated as a reference method in the current
experiments.
