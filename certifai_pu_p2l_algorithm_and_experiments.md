# Certified Pivot-Utility Pick-to-Learn (PU-P2L): Algorithm and Experiment Plan

**Project:** CertifAI — Generalization Boundary  
**Purpose:** Design a GREATS-inspired data-selection rule that keeps the Pick-to-Learn (P2L) / P2L-ES compression certificate, while reducing the failure mode of loss-based selection under noisy, duplicate, or redundant hard examples.

---

## 0. Executive summary

The current experiment shows a clear limitation of the standard loss-based P2L selection rule:

- As the label-noise rate in duplicate hard groups increases, **MaxLoss / loss-based P2L repeatedly selects noisy or redundant hard points**.
- This inflates the compression set or the early-stopping penalty, so the P2L/P2L-ES bound worsens.
- GREATS-style gradient interaction is attractive because it corrects for redundancy, but standard GREATS is **not directly certifiable** under the P2L theorem: it uses stochastic online candidate batches and validation-loss utility, and it is not a P2L-style compression function.

The proposed route is therefore:

> Keep the P2L Stop rule and compression structure, but replace the secondary ranking among currently inappropriate points with a deterministic, pivot-based utility/diversity score. The pivots are selected only from the current compression set, so the whole selection path remains reconstructible from the compression set.

We call the method:

> **Pivot-Utility Pick-to-Learn (PU-P2L)**  
> and its early-stopping version  
> **PU-P2L-ES**.

The algorithm is designed to satisfy the P2L preference property by construction. Empirically, it is expected to improve the bound in either of two ways:

1. **Faster Stop:** it reaches the Stop condition with a smaller compression set.
2. **Better early stopping:** for a fixed budget \(M\), it leaves fewer inappropriate points, reducing the P2L-ES effective compression size.

---

## 1. Background and design goal

### 1.1 P2L certification mechanism

P2L runs a black-box learner \(L\) on an iteratively constructed compression set \(T\subseteq D\). At each iteration, it selects the least appropriate remaining point under a hypothesis-dependent total order and stops when every remaining point is appropriate.

For classification, the appropriateness condition can be

\[
\ell(h,z) \le \gamma,
\]

where \(\ell\) is cross-entropy and \(\gamma=-\log(1/2)\) in binary classification. The risk is

\[
R(h)=\mathbb{P}\{\ell(h,z)>\gamma\}.
\]

The P2L bound has the form

\[
\mathbb{P}\left\{R(h)\le \bar\varepsilon(|T|,\delta)\right\}\ge 1-\delta.
\]

For P2L-ES, if the algorithm stops after a fixed budget \(M\), the valid bound uses the **effective compression size**

\[
k_{\mathrm{eff}}
=
|T|
+
\left|
\{z\in D\setminus T:\ell(h,z)>\gamma\}
\right|.
\]

Thus,

\[
\mathbb{P}\left\{R(h)\le \bar\varepsilon(k_{\mathrm{eff}},\delta)\right\}\ge 1-\delta.
\]

The aim of our selection rule is therefore **not merely to reduce test error**, but to reduce \(|T|\) or \(k_{\mathrm{eff}}\).

---

### 1.2 GREATS motivation and why we cannot use standard GREATS directly

GREATS optimizes online mini-batch selection through a validation-loss utility. In simplified form, the marginal gain of selecting a point \(z\) is approximated by

\[
U^{(t)}(z\mid \widehat B_t)
\approx
\eta_t\, g_t(z)^\top g_t(z^{\mathrm{val}})
-
\eta_t^2\, g_t(z)^\top H_t(z^{\mathrm{val}})
\sum_{u\in \widehat B_t} g_t(u),
\]

and under an identity-Hessian approximation,

\[
U^{(t)}(z\mid \widehat B_t)
\approx
\eta_t\, g_t(z)^\top g_t(z^{\mathrm{val}})
-
\eta_t^2
\sum_{u\in \widehat B_t}
g_t(z)^\top g_t(u).
\]

This is useful because it prefers points that help the pivot/validation direction but penalizes points that are redundant with already selected points.

However, standard GREATS breaks the P2L certification route for three reasons:

1. It samples online candidate mini-batches \(B_t\), so adding a new point can change which old points are considered.
2. It uses validation pivots rather than compression-set-reconstructible pivots.
3. Its utility is not tied to the P2L Stop event \(\ell(h,z)>\gamma\).

Therefore, the new algorithm should borrow the **utility-diversity scoring idea**, but it must preserve the **Stop-based total order** and **compression-set reconstructibility**.

---

## 2. Core design principles

The proposed algorithm must obey the following rules.

### Rule 1: No online candidate mini-batch sampling for certification

At each P2L iteration, the selector evaluates all remaining points

\[
D\setminus T.
\]

Mini-batching can be used for computational approximation only if it does not change the mathematical selection rule. For example, one may process candidates in chunks to compute scores, but the final selected point must be the global maximizer under the deterministic score.

### Rule 2: Pivots are selected only from the current compression set

At iteration \(m\), the pivot set is

\[
P_m = \Pi_r(T_m,h_m),
\]

where \(r\) is a hyperparameter and \(\Pi_r\) is a deterministic pivot selector. The pivot set must depend only on the current compression set \(T_m\), the current hypothesis \(h_m=L(T_m)\), and fixed hyperparameters.

Allowed pivot choices include:

1. **Hard pivots**
   \[
   P_m^{H}=\operatorname{Top}_{r_H}\{\ell(h_m,u):u\in T_m\}.
   \]

2. **Easy pivots**
   \[
   P_m^{E}=\operatorname{Bottom}_{r_E}\{\ell(h_m,u):u\in T_m\}.
   \]

3. **Mixed pivots**
   \[
   P_m=P_m^{H}\cup P_m^{E}.
   \]

4. **Gradient centroid pivot**
   \[
   a_m^{H}
   =
   \frac{1}{|P_m^H|}
   \sum_{p\in P_m^H}
   \widetilde g_m(p),
   \]
   where \(\widetilde g_m(p)\) is a normalized gradient or last-layer gradient.

5. **Feature centroid pivot**
   \[
   c_m^{H}
   =
   \frac{1}{|P_m^H|}
   \sum_{p\in P_m^H}
   \widetilde \phi_m(p),
   \]
   where \(\phi_m(p)\) is a frozen representation, such as the penultimate-layer feature.

The number of pivots \(r\), or \((r_H,r_E)\), is treated as a hyperparameter.

### Rule 3: No validation data are required

Unlike GREATS, PU-P2L uses no external validation set. Pivots are chosen from \(T_m\). This is important because P2L aims to use the training/certification data themselves to produce a self-certified model.

### Rule 4: Stop is controlled only by appropriateness

The Stop decision must be independent of the secondary utility score. Define

\[
v_m(z)=\mathbf{1}\{\ell(h_m,z)>\gamma\}.
\]

Then:

- if \(v_m(z)=0\), the point is appropriate;
- if \(v_m(z)=1\), the point is inappropriate.

The algorithm stops if and only if

\[
v_m(z)=0
\quad
\forall z\in D\setminus T_m.
\]

The utility score decides only **which inappropriate point to add next**.

---

## 3. PU-P2L algorithm

### 3.1 Inputs

- Dataset:
  \[
  D=\{z_i=(x_i,y_i)\}_{i=1}^N.
  \]

- Inner learner:
  \[
  L:T\mapsto h_T.
  \]

- Initial hypothesis:
  \[
  h_0.
  \]

- Loss:
  \[
  \ell(h,z).
  \]

- Appropriateness threshold:
  \[
  \gamma.
  \]

- Pivot selector:
  \[
  \Pi_r(T,h).
  \]

- Utility-diversity score:
  \[
  q(z;h,T,\Pi_r).
  \]

- Deterministic tie-breaker:
  \[
  \tau(z),
  \]
  for example the original index in the dataset or a deterministic hash of the sample.

- Optional early-stopping budget:
  \[
  M.
  \]

---

### 3.2 Deterministic total order

At iteration \(m\), define the extended state

\[
s_m=(h_m,T_m),
\]

where

\[
h_m=L(T_m).
\]

For any remaining data point \(z\in D\setminus T_m\), define

\[
V_m(z)=\mathbf{1}\{\ell(h_m,z)>\gamma\},
\]

and

\[
Q_m(z)=q(z;h_m,T_m,\Pi_r).
\]

Define a lexicographic key

\[
K_m(z)=
\left(
V_m(z),
Q_m(z),
\tau(z)
\right).
\]

The total order is

\[
z_1\preceq_{s_m} z_2
\quad
\Longleftrightarrow
\quad
K_m(z_1)\le_{\mathrm{lex}}K_m(z_2).
\]

The Stop element is placed between appropriate and inappropriate points:

\[
z\preceq_{s_m}\mathrm{Stop}
\quad
\Longleftrightarrow
\quad
V_m(z)=0,
\]

\[
\mathrm{Stop}\preceq_{s_m}z
\quad
\Longleftrightarrow
\quad
V_m(z)=1.
\]

Therefore, the maximum element is Stop if and only if there are no inappropriate remaining points.

---

### 3.3 PU-P2L pseudocode

```text
Algorithm: PU-P2L(D)

Input:
    dataset D
    learner L
    initial hypothesis h0
    loss threshold gamma
    pivot selector Pi_r
    score function q
    deterministic tie-breaker tau

Initialize:
    T0 = empty set
    h0 = initial hypothesis
    m = 0

Repeat:
    For every z in D \ Tm:
        compute violation V_m(z) = 1{ell(h_m,z) > gamma}

    If V_m(z)=0 for all z in D \ Tm:
        return h_m, T_m

    Choose pivots:
        P_m = Pi_r(T_m,h_m)

    For every z in D \ Tm:
        compute Q_m(z) = q(z; h_m, T_m, P_m)

    Select:
        z_m = argmax_{z in D \ Tm} (V_m(z), Q_m(z), tau(z))

    Update:
        T_{m+1} = T_m union {z_m}
        h_{m+1} = L(T_{m+1})
        m = m + 1
```

---

### 3.4 PU-P2L-ES pseudocode

```text
Algorithm: PU-P2L-ES(D, M)

Input:
    same as PU-P2L, plus early-stopping budget M

Initialize:
    T0 = empty set
    h0 = initial hypothesis
    m = 0

Repeat while |T_m| < M:
    For every z in D \ Tm:
        compute violation V_m(z) = 1{ell(h_m,z) > gamma}

    If V_m(z)=0 for all z in D \ Tm:
        return h_m, T_m, k_eff = |T_m|

    Choose pivots:
        P_m = Pi_r(T_m,h_m)

    For every z in D \ Tm:
        compute Q_m(z) = q(z; h_m, T_m, P_m)

    Select:
        z_m = argmax_{z in D \ Tm} (V_m(z), Q_m(z), tau(z))

    Update:
        T_{m+1} = T_m union {z_m}
        h_{m+1} = L(T_{m+1})
        m = m + 1

After early stopping:
    P_bad = {z in D \ T_M : ell(h_M,z) > gamma}
    k_eff = |T_M| + |P_bad|
    return h_M, T_M, k_eff
```

---

## 4. Why the algorithm is preference-preserving

### 4.1 Key theorem

**Theorem 1 — Preference of PU-P2L.**  
Assume:

1. \(L\) is deterministic given an ordered compression set, or its randomness is fixed by a recorded seed.
2. \(\Pi_r(T,h)\) is deterministic and depends only on \(T\), \(h=L(T)\), and fixed hyperparameters.
3. \(q(z;h,T,\Pi_r)\) is deterministic and depends only on \(z\), \(h\), \(T\), and \(\Pi_r(T,h)\).
4. Ties are broken deterministically.
5. Stop is defined only by \(\ell(h,z)\le \gamma\).

Then the compression function \(\kappa_{\mathrm{PU}}\) induced by PU-P2L satisfies the preference property:

\[
\kappa_{\mathrm{PU}}(D)\ne V
\quad\Longrightarrow\quad
\kappa_{\mathrm{PU}}(D\cup\{z\})\ne V,
\qquad
\forall V\subseteq D,\ \forall z\in Z.
\]

### 4.2 Proof sketch

Run PU-P2L on \(D\) and on \(D\cup\{z\}\). Suppose the two runs agree up to iteration \(m\). Then:

\[
T_m'=T_m,
\qquad
h_m'=h_m,
\qquad
P_m'=P_m.
\]

Therefore, every old point \(u\in D\setminus T_m\) receives exactly the same key in both executions:

\[
K_m'(u)=K_m(u).
\]

The only additional candidate in the augmented execution is \(z\). Hence

\[
\max_{\preceq_{s_m}}((D\cup\{z\})\setminus T_m\cup\{\mathrm{Stop}\})
\in
\left\{
\max_{\preceq_{s_m}}(D\setminus T_m\cup\{\mathrm{Stop}\}),
z
\right\}.
\]

Thus the first divergence between the two executions can only happen because the new point \(z\) is selected. Therefore, if the compression changes after adding \(z\), the new compression contains \(z\), so it cannot equal any old subset \(V\subseteq D\). This proves preference.

### 4.3 Why using pivots from the full remaining dataset is unsafe

If pivots are selected from \(D\setminus T_m\), then adding \(z\) can change the pivot set even before \(z\) is selected. This can change the scores of old points and violate preference. Therefore, percentile pivots, global hard/easy pivots from \(D\setminus T_m\), or validation-like pivots extracted from the uncompressed dataset are not safe unless they are explicitly recorded in the compression object.

### 4.4 Why compression-set pivots are safe

If pivots are selected only from \(T_m\), then before the first divergence the two executions have identical \(T_m\), identical \(h_m\), and identical pivots. Thus old points retain identical scores. This is exactly the structure needed by the P2L preference proof.

---

## 5. Candidate score functions

The first implementation should include several score functions. The theorem above holds for all of them as long as the score is deterministic and uses only \(z\), \(h_m\), \(T_m\), and pivots from \(T_m\).

Let

\[
\phi_m(z)
\]

be the penultimate-layer feature of \(z\), and let

\[
e_m(z)=p_m(z)-y
\]

be the softmax error vector. For the last-layer weight matrix, the per-example gradient has the form

\[
g_m^{\mathrm{LL}}(z)
=
\phi_m(z)\otimes e_m(z).
\]

The last-layer gradient cosine is

\[
K_m^{\mathrm{LL}}(z,u)
=
\frac{
\langle \phi_m(z),\phi_m(u)\rangle
\langle e_m(z),e_m(u)\rangle
}{
\|\phi_m(z)\|\|\phi_m(u)\|\|e_m(z)\|\|e_m(u)\|
+\epsilon
}.
\]

This is much cheaper than full per-sample gradients and approximates a last-layer NTK similarity.

---

### Score A: Loss-Diversity Score

This is the cheapest robust baseline.

\[
q_A(z)
=
\operatorname{clip}\left(\frac{\ell(h_m,z)}{\gamma},0,c_\ell\right)
-
\lambda
\max_{p\in P_m}
\operatorname{sim}_\phi(z,p),
\]

where

\[
\operatorname{sim}_\phi(z,p)
=
\frac{
\langle \phi_m(z),\phi_m(p)\rangle
}{
\|\phi_m(z)\|\|\phi_m(p)\|+\epsilon
}.
\]

**Interpretation.**

- The first term keeps the point hard enough to be useful.
- The clipping prevents very noisy labels from dominating.
- The second term avoids selecting points already represented by the compression-set pivots.

**Expected strength.**

Good for noisy duplicate groups because near-duplicates have high feature similarity and are penalized.

**Cost.**

One forward pass over remaining data plus feature similarity to \(r\) pivots.

---

### Score B: Pivot-NTK Utility Score

This is the closest safe analogue of GREATS.

Let hard pivots be

\[
P_m^H=\operatorname{Top}_{r_H}\{\ell(h_m,p):p\in T_m\},
\]

and easy pivots be

\[
P_m^E=\operatorname{Bottom}_{r_E}\{\ell(h_m,p):p\in T_m\}.
\]

Define

\[
A_H(z)
=
\frac{1}{|P_m^H|}
\sum_{p\in P_m^H}
K_m^{\mathrm{LL}}(z,p),
\]

\[
A_E(z)
=
\frac{1}{|P_m^E|}
\sum_{p\in P_m^E}
K_m^{\mathrm{LL}}(z,p),
\]

\[
R_H(z)
=
\max_{p\in P_m^H}
K_m^{\mathrm{LL}}(z,p).
\]

Then

\[
q_B(z)
=
\operatorname{clip}\left(\frac{\ell(h_m,z)}{\gamma},0,c_\ell\right)
+
\alpha A_H(z)
-
\beta A_E(z)
-
\lambda R_H(z).
\]

**Interpretation.**

- \(A_H(z)\): prefer points whose update helps currently hard compression pivots.
- \(A_E(z)\): reduce focus on already easy regions.
- \(R_H(z)\): penalize redundant near-copies of hard pivots.
- Clipped loss avoids over-selecting pure label noise.

**Expected strength.**

This should be the primary proposed score. It captures GREATS-like gradient interaction while keeping pivots inside the compression set.

**Cost.**

One forward/backward-compatible last-layer-gradient computation over remaining data, plus \(O(Nr)\) similarity operations.

---

### Score C: Novel-Hardness Score

\[
q_C(z)
=
\operatorname{clip}\left(\frac{\ell(h_m,z)}{\gamma},0,c_\ell\right)
+
\lambda
\left[
1-
\max_{p\in P_m}
\operatorname{sim}_\phi(z,p)
\right].
\]

**Interpretation.**

Among violating points, prefer hard points that are feature-novel relative to current pivots.

**Expected strength.**

Simple and fast. May outperform MaxLoss when noisy duplicates are clustered, but may sometimes pick outliers if \(\lambda\) is too large.

---

### Score D: Orthogonal Gradient Novelty Score

Let

\[
G_P=[\widetilde g_m^{\mathrm{LL}}(p_1),\dots,\widetilde g_m^{\mathrm{LL}}(p_r)]
\]

be the matrix of normalized last-layer pivot gradients. Define the projection onto the pivot-gradient span as

\[
\Pi_P
=
G_P(G_P^\top G_P+\rho I)^{-1}G_P^\top.
\]

Then

\[
q_D(z)
=
\operatorname{clip}\left(\frac{\ell(h_m,z)}{\gamma},0,c_\ell\right)
+
\lambda
\left\|
(I-\Pi_P)\widetilde g_m^{\mathrm{LL}}(z)
\right\|_2^2.
\]

**Interpretation.**

This selects hard points whose gradients add new directions not already covered by compression pivots.

**Expected strength.**

Good for reducing redundancy and improving NTK coverage. It is more expensive but still feasible for small models or last-layer gradients.

---

### Score E: Robust Pivot-Utility Score

This is the recommended initial main score.

\[
q_E(z)
=
s_\ell(z)
+
\alpha A_H(z)
+
\mu N_\phi(z)
-
\lambda R_H(z),
\]

where

\[
s_\ell(z)
=
\operatorname{clip}\left(\frac{\ell(h_m,z)}{\gamma},0,c_\ell\right),
\]

\[
A_H(z)
=
\frac{1}{|P_m^H|}
\sum_{p\in P_m^H}
\max\{K_m^{\mathrm{LL}}(z,p),0\},
\]

\[
R_H(z)
=
\max_{p\in P_m^H}
\max\{K_m^{\mathrm{LL}}(z,p),0\},
\]

and

\[
N_\phi(z)
=
1-\max_{p\in P_m}
\operatorname{sim}_\phi(z,p).
\]

**Recommended default hyperparameters.**

- \(r_H=5\), \(r_E=0\) for small compression sets.
- \(r_H=10\), \(r_E=5\) for larger compression sets.
- \(c_\ell=3\).
- \(\alpha\in\{0.25,0.5,1.0\}\).
- \(\mu\in\{0.1,0.25,0.5\}\).
- \(\lambda\in\{0.25,0.5,1.0\}\).

**Why this is promising.**

It combines:
- clipped hardness,
- usefulness to hard pivots,
- feature novelty,
- redundancy penalty.

This directly targets the observed failure mode of MaxLoss: repeatedly selecting noisy duplicate hard examples.

---

## 6. Efficiency considerations

### 6.1 Why full gradients should be avoided initially

Full per-sample gradients are expensive. GREATS avoids this through ghost inner-products, but the full-dataset P2L setting is different from online batch selection. We should start with cheaper proxies.

### 6.2 Recommended efficient implementations

#### Option 1: Feature-only score

Use Score A or C with penultimate features. This requires only forward passes.

Best for:
- quick experiments,
- 2D synthetic data,
- binary MNIST/Fashion-MNIST,
- checking whether the theorem-compatible score improves bounds.

#### Option 2: Last-layer gradient score

Use Score B or E with last-layer gradient cosine:

\[
K_m^{\mathrm{LL}}(z,p)
=
\cos(\phi_m(z)\otimes e_m(z),\phi_m(p)\otimes e_m(p)).
\]

This avoids materializing full gradients.

Best for:
- principled GREATS-style utility,
- NTK diagnostics,
- moderate-scale image experiments.

#### Option 3: Chunked candidate scoring

The mathematical selection is over all remaining points, but computation can be chunked:

```text
best_score = -infinity
best_point = None

for chunk C in partition(D \ T):
    compute Q_m(z) for all z in C
    update best_point using global deterministic tie-breaker
```

This preserves the global argmax and does not break preference.

#### Option 4: cached features with periodic refresh

For neural networks, \(\phi_m(z)\) changes as \(h_m\) changes. To reduce cost:

- refresh features every \(s\) P2L iterations;
- use stale features for the next \(s-1\) selections;
- treat this schedule as part of the deterministic algorithm.

This is theoretically safe if the refresh schedule is deterministic and reconstructible from \(T_m\) and iteration count. However, stale features may weaken empirical performance.

---

## 7. Theoretical claim we can make

### Claim 1: certification validity

If PU-P2L uses compression-set pivots and the Stop-based lexicographic order, then the induced compression function is preferent. Therefore, the standard P2L compression bound applies.

### Claim 2: early-stopping certification validity

If PU-P2L-ES uses the same preference-preserving order and returns the early-stopping penalty

\[
P_{\mathrm{bad}}
=
\{z\in D\setminus T:\ell(h,z)>\gamma\},
\]

then the P2L-ES bound applies with

\[
k_{\mathrm{eff}}
=
|T|+|P_{\mathrm{bad}}|.
\]

### Claim 3: conditions for improving over MaxLoss

PU-P2L improves the bound over MaxLoss-P2L if either:

1. **Faster Stop**
   \[
   |T_{\mathrm{PU}}|<|T_{\mathrm{MaxLoss}}|;
   \]

2. **Smaller early-stopping penalty**
   \[
   |T_{\mathrm{PU}}|+|P_{\mathrm{bad,PU}}|
   <
   |T_{\mathrm{MaxLoss}}|+|P_{\mathrm{bad,MaxLoss}}|.
   \]

Because \(\bar\varepsilon(k,\delta)\) is increasing in \(k\), either inequality yields a tighter certificate.

### What we cannot claim before experiments

We cannot claim PU-P2L always outperforms MaxLoss. There are clean, non-redundant settings where the highest-loss point is genuinely the most informative point. The intended improvement regime is:

- noisy hard groups,
- duplicate or near-duplicate examples,
- redundant high-loss samples,
- outlier-heavy datasets,
- cases where MaxLoss overfits the worst point rather than improving global appropriateness.

---

## 8. Experiment plan

The experiments should be lightweight but theory-aligned. The key is not only to show better test error, but to show better **certification-efficiency trade-off**.

---

# Experiment 1: P2L boundary guarantee under train/pretrain fraction

## Goal

Show that the proposed PU-P2L/PU-P2L-ES score preserves the P2L boundary guarantee and can produce tighter bounds than MaxLoss when it reaches Stop faster or leaves fewer inappropriate points.

## Datasets

Use two levels:

### Level 1: synthetic duplicate-hard binary classification

This is the dataset already used in the current experiment.

- Binary classification.
- Clean base clusters plus duplicate hard groups.
- Label-noise rate can be fixed at \(\rho=0.2\) for this experiment.
- Total training size: \(N\in[1000,3000]\).
- Test set: clean distribution without corrupted duplicate labels.

### Level 2: binary MNIST / Fashion-MNIST

Follow the P2L setup:

- Binary MNIST:
  \[
  y=0 \text{ for digits }0\text{--}4,\quad y=1 \text{ for digits }5\text{--}9.
  \]
- Optional Fashion-MNIST binary split:
  classes \(0\text{--}4\) vs \(5\text{--}9\).
- Use \(N=1000\) or \(N=2000\) training points per trial to keep cost low.
- Use the standard test set only for reporting actual risk, not for training or certification.

## Train/pretrain split

For each fraction

\[
\rho_{\mathrm{pre}}\in\{0.1,0.2,\dots,0.9\},
\]

split the training data into:

- pretrain set \(S_0\) of size \(\rho_{\mathrm{pre}}N\),
- compression/certification set \(D_c\) of size \((1-\rho_{\mathrm{pre}})N\).

Train \(h_0\) on \(S_0\). Then run P2L variants on \(D_c\).

## Methods

1. P2L-MaxLoss.
2. P2L-ES-MaxLoss.
3. PU-P2L Score A: Loss-Diversity.
4. PU-P2L Score B: Pivot-NTK Utility.
5. PU-P2L Score E: Robust Pivot-Utility.
6. PU-P2L-ES Score E.
7. GREATS-lite only as an empirical reference, with no P2L bound.

## Bound computation

For standard P2L:

\[
k=|T|.
\]

For P2L-ES:

\[
k_{\mathrm{eff}}
=
|T|
+
\left|
\{z\in D_c\setminus T:\ell(h,z)>\gamma\}
\right|.
\]

Plot

\[
\bar\varepsilon(k,\delta)
\quad
\text{or}
\quad
\bar\varepsilon(k_{\mathrm{eff}},\delta).
\]

Use:

\[
\delta=0.035
\]

as the main setting, and optionally

\[
\delta=0.001.
\]

## Metrics

- Test risk:
  \[
  \widehat R_{\mathrm{test}}(h).
  \]
- P2L/P2L-ES bound:
  \[
  \bar\varepsilon(k,\delta)
  \quad\text{or}\quad
  \bar\varepsilon(k_{\mathrm{eff}},\delta).
  \]
- Compression size:
  \[
  |T|.
  \]
- Early-stopping penalty:
  \[
  |P_{\mathrm{bad}}|.
  \]
- Effective compression size:
  \[
  k_{\mathrm{eff}}.
  \]
- Runtime.
- Number of calls to the inner learner.

## Plots

### Plot 1A: Bound and test risk versus train/pretrain fraction

- x-axis:
  \[
  \rho_{\mathrm{pre}}.
  \]
- y-axis:
  test risk and generalization bound.
- Solid line: test risk.
- Dashed line: P2L/P2L-ES bound.
- One curve per method.

### Plot 1B: Effective compression size versus train/pretrain fraction

- x-axis:
  \[
  \rho_{\mathrm{pre}}.
  \]
- y-axis:
  \[
  |T| \text{ or } k_{\mathrm{eff}}.
  \]

### Plot 1C: Runtime versus train/pretrain fraction

- x-axis:
  \[
  \rho_{\mathrm{pre}}.
  \]
- y-axis:
  runtime.

## Expected outcome

PU-P2L should retain valid bounds by construction. It may outperform MaxLoss-P2L if it avoids repeatedly selecting redundant/noisy hard examples. The clearest success criterion is not only lower test risk but smaller \(k_{\mathrm{eff}}\).

---

# Experiment 2: Noise-label robustness

## Goal

Show that PU-P2L avoids noisy duplicate labels better than loss-based methods.

## Dataset

Use the current synthetic duplicate-hard dataset.

Vary label-noise rate:

\[
\rho_{\mathrm{noise}}\in\{0,0.05,0.1,0.2,0.3,0.4\}.
\]

For each noise rate, generate several random seeds.

## Methods

1. Random.
2. MaxLoss.
3. GradNorm.
4. GraNd.
5. EL2N.
6. KCenter.
7. GREATS-lite.
8. P2L-MaxLoss.
9. P2L-ES-MaxLoss.
10. PU-P2L Score A.
11. PU-P2L Score B.
12. PU-P2L Score E.
13. PU-P2L-ES Score E.

## Metrics

### Main metrics

- Clean test error.
- Bound:
  \[
  \bar\varepsilon(k,\delta)
  \quad\text{or}\quad
  \bar\varepsilon(k_{\mathrm{eff}},\delta).
  \]
- Effective compression size:
  \[
  k_{\mathrm{eff}}.
  \]

### Noise/redundancy diagnostics

- Noise-hit rate:
  \[
  \frac{|T\cap D_{\mathrm{noisy}}|}{|T|}.
  \]

- Duplicate-hit rate:
  \[
  \frac{|T\cap D_{\mathrm{duplicate}}|}{|T|}.
  \]

- Pairwise feature cosine:
  \[
  \frac{2}{|T|(|T|-1)}
  \sum_{i<j}
  \cos(\phi(z_i),\phi(z_j)).
  \]

- Pairwise last-layer gradient cosine:
  \[
  \frac{2}{|T|(|T|-1)}
  \sum_{i<j}
  K^{\mathrm{LL}}(z_i,z_j).
  \]

- Cluster coverage:
  \[
  \frac{\#\{\text{selected clusters}\}}{\#\{\text{all clusters}\}}.
  \]

## Plots

### Plot 2A: Clean test error versus label-noise rate

- x-axis:
  \[
  \rho_{\mathrm{noise}}.
  \]
- y-axis:
  clean test error.

### Plot 2B: P2L/P2L-ES bound versus label-noise rate

- x-axis:
  \[
  \rho_{\mathrm{noise}}.
  \]
- y-axis:
  generalization bound.

### Plot 2C: Effective compression size versus label-noise rate

- x-axis:
  \[
  \rho_{\mathrm{noise}}.
  \]
- y-axis:
  \[
  k_{\mathrm{eff}}.
  \]

### Plot 2D: Noise-hit and duplicate-hit diagnostics

Use bar plots at high noise, e.g.

\[
\rho_{\mathrm{noise}}=0.4.
\]

## Expected outcome

MaxLoss and loss-like rules should increasingly select noisy duplicate points as \(\rho_{\mathrm{noise}}\) increases. PU-P2L should reduce noise-hit and duplicate-hit rates because the redundancy penalty and clipping prevent repeated selection of very similar high-loss points.

---

# Experiment 3: Training-dynamics efficiency using NTK-style analysis

## Goal

Show that PU-P2L selects compression sets whose gradients cover the learning dynamics more efficiently than MaxLoss.

## Motivation

The Data Diet paper uses gradient/NTK-style analysis to study how high-scoring examples drive training dynamics. We use a similar diagnostic to ask whether the PU-P2L compression set better covers useful gradient directions.

## Practical choice

Full NTK is expensive. Use one of the following:

### Option A: last-layer empirical NTK

\[
K^{\mathrm{LL}}(z,u)
=
\langle g^{\mathrm{LL}}(z),g^{\mathrm{LL}}(u)\rangle.
\]

This is efficient and directly related to our score.

### Option B: feature kernel

\[
K^\phi(z,u)
=
\langle \phi(z),\phi(u)\rangle.
\]

This is cheaper and useful for larger experiments.

### Option C: small-model full NTK

For 2D synthetic data or a small MLP, compute full per-example gradients and exact empirical NTK.

---

## Metrics

Let \(U\) be a fixed probe set sampled from the training distribution. It is used only for diagnostics, not for selection.

### Metric 1: NTK coverage ratio

Let \(T_m\) be the compression set at iteration \(m\). Define

\[
\mathrm{Coverage}(T_m;U)
=
\frac{
\operatorname{Tr}
\left[
K_{U,T_m}
(K_{T_m,T_m}+\rho I)^{-1}
K_{T_m,U}
\right]
}{
\operatorname{Tr}(K_{U,U})
}.
\]

This measures how much of the probe-set gradient kernel is covered by the selected compression set.

### Metric 2: effective rank of selected gradients

\[
\mathrm{erank}(K_{T_m,T_m})
=
\exp
\left(
-\sum_i \bar\lambda_i\log \bar\lambda_i
\right),
\]

where

\[
\bar\lambda_i
=
\frac{\lambda_i}{\sum_j\lambda_j}.
\]

Higher effective rank indicates less redundancy.

### Metric 3: predicted one-step loss reduction on probe set

For a selected point \(z\),

\[
\Delta_U(z)
\approx
\eta
\sum_{u\in U}
g(u)^\top g(z).
\]

For a compression set \(T_m\),

\[
\Delta_U(T_m)
\approx
\eta
\sum_{z\in T_m}
\sum_{u\in U}
g(u)^\top g(z).
\]

### Metric 4: inappropriate-point decay

Track

\[
N_{\mathrm{bad}}(m)
=
\left|
\{z\in D\setminus T_m:\ell(h_m,z)>\gamma\}
\right|.
\]

This is directly tied to the P2L-ES effective compression size.

## Plots

### Plot 3A: NTK coverage versus compression size

- x-axis:
  \[
  |T_m|.
  \]
- y-axis:
  \[
  \mathrm{Coverage}(T_m;U).
  \]

### Plot 3B: Effective rank versus compression size

- x-axis:
  \[
  |T_m|.
  \]
- y-axis:
  \[
  \mathrm{erank}(K_{T_m,T_m}).
  \]

### Plot 3C: Inappropriate-point decay versus iteration

- x-axis:
  iteration \(m\).
- y-axis:
  \[
  N_{\mathrm{bad}}(m).
  \]

### Plot 3D: predicted one-step loss reduction versus iteration

- x-axis:
  iteration \(m\).
- y-axis:
  \[
  \Delta_U(T_m).
  \]

## Expected outcome

PU-P2L should have higher NTK coverage and effective rank at the same compression size, while MaxLoss may have low effective rank due to repeatedly selecting near-duplicate high-loss points.

---

# Experiment 4: Compression-set comparison

## Goal

Understand what the new score selects differently from MaxLoss.

## Methods

Compare the compression sets produced by:

1. P2L-MaxLoss.
2. P2L-ES-MaxLoss.
3. PU-P2L Score A.
4. PU-P2L Score B.
5. PU-P2L Score E.
6. GREATS-lite empirical reference.

## Metrics

### Set overlap

\[
\mathrm{Jaccard}(T_a,T_b)
=
\frac{|T_a\cap T_b|}{|T_a\cup T_b|}.
\]

### Label distribution

Class proportions in selected compression set:

\[
\hat p_T(y)
=
\frac{|\{z\in T:y_z=y\}|}{|T|}.
\]

### Hardness distribution

Histogram of selected points by original loss or current loss.

### Noise/duplicate selection

Same as Experiment 2.

### Feature-space visualization

Use PCA or UMAP on \(\phi(z)\), plotting:

- all clean training points,
- noisy points,
- MaxLoss selected points,
- PU-P2L selected points.

### Compression-set diversity

\[
\mathrm{Diversity}(T)
=
1-
\frac{2}{|T|(|T|-1)}
\sum_{i<j}
\operatorname{sim}_\phi(z_i,z_j).
\]

## Plots

### Plot 4A: selected-set visualization

Feature-space scatter plot with selected points highlighted.

### Plot 4B: Jaccard overlap heatmap

Rows and columns are methods; entries are Jaccard overlaps.

### Plot 4C: selected-set similarity histogram

Distribution of pairwise feature/gradient similarities inside \(T\).

### Plot 4D: selected-set loss histogram

Shows whether PU-P2L avoids extreme noisy loss outliers.

## Expected outcome

PU-P2L should select fewer noisy duplicates, higher diversity, and more gradient/feature coverage than MaxLoss.

---

## 9. Initial hyperparameter plan

Use a small grid to avoid expensive tuning.

### Pivot counts

\[
r_H\in\{3,5,10\},
\qquad
r_E\in\{0,3,5\}.
\]

### Loss clipping

\[
c_\ell\in\{2,3,5\}.
\]

### Utility/diversity weights

\[
\alpha\in\{0.25,0.5,1.0\},
\]

\[
\lambda\in\{0.25,0.5,1.0\},
\]

\[
\mu\in\{0.1,0.25,0.5\}.
\]

### Recommended first configuration

Start with Score E:

\[
r_H=5,
\quad
r_E=0,
\quad
c_\ell=3,
\quad
\alpha=0.5,
\quad
\lambda=0.5,
\quad
\mu=0.25.
\]

Then adjust:

- If the method selects too many outliers/noise: increase \(\lambda\), decrease \(\mu\), decrease \(c_\ell\).
- If the method becomes too conservative and ignores hard regions: increase \(\alpha\), increase \(c_\ell\).
- If the method selects too many feature-novel but unhelpful samples: decrease \(\mu\).
- If the method is too close to MaxLoss: increase \(\lambda\) and/or add \(r_E>0\).

---

## 10. Implementation checklist

### Determinism

To preserve theoretical validity:

- deterministic tie-breaking;
- fixed random seeds if the inner learner is stochastic;
- deterministic pivot selector;
- deterministic chunk order;
- deterministic feature-refresh schedule if caching is used.

### Certification split

If pretraining is used:

- train \(h_0\) on \(S_0\);
- run PU-P2L only on \(D_c\);
- compute the bound using \(|D_c|\), not the full \(N\).

### Early stopping

For PU-P2L-ES, always record:

\[
|T|,
\quad
|P_{\mathrm{bad}}|,
\quad
k_{\mathrm{eff}}=|T|+|P_{\mathrm{bad}}|.
\]

### Do not use test data in selection

The test set is only for reporting realized risk.

### Do not use validation data for the certified selector

GREATS-lite can use validation as an empirical baseline, but PU-P2L should not, unless the validation set is fixed externally and the certification statement is carefully conditioned on it.

---

## 11. Main paper narrative supported by these experiments

The final story should be:

1. **P2L gives strong compression-based generalization certificates**, but standard loss-based selection is vulnerable to noisy duplicate hard examples.
2. **GREATS solves part of the selection problem** by using gradient-interaction corrections, but standard GREATS is not a P2L compression scheme and does not satisfy the P2L preference route.
3. **PU-P2L preserves the P2L certificate** by using a Stop-based lexicographic order and compression-set pivots.
4. **PU-P2L improves the certification-efficiency trade-off** when it reaches Stop earlier or leaves fewer inappropriate points under early stopping.
5. **The NTK/gradient diagnostics explain why:** the selected compression set has better coverage and lower redundancy.

---

## 12. Minimal first experiment to run

If time is limited, run only this:

1. Dataset: synthetic duplicate-hard dataset.
2. Noise rates:
   \[
   \rho_{\mathrm{noise}}\in\{0,0.1,0.2,0.4\}.
   \]
3. Methods:
   - P2L-ES-MaxLoss,
   - PU-P2L-ES Score A,
   - PU-P2L-ES Score E,
   - GREATS-lite,
   - Random.
4. Budget:
   \[
   M\in\{50,100,200\}.
   \]
5. Metrics:
   - clean test error,
   - P2L-ES bound,
   - \(k_{\mathrm{eff}}\),
   - noise-hit rate,
   - duplicate-hit rate,
   - selected-set pairwise similarity.
6. Key plot:
   - x-axis: label-noise rate;
   - y-axis: bound, \(k_{\mathrm{eff}}\), and clean test error.

Success criterion:

\[
k_{\mathrm{eff}}^{\mathrm{PU}}
<
k_{\mathrm{eff}}^{\mathrm{MaxLoss}}
\]

for moderate/high noise, with no worse test error.

---

## 13. Risks and fallback options

### Risk 1: PU-P2L does not beat MaxLoss in clean data

This is acceptable. MaxLoss can be optimal in clean non-redundant settings. The target regime is noisy/redundant hard groups.

### Risk 2: Feature-diversity score selects outliers

Use clipped loss and reduce novelty weight:

\[
c_\ell\downarrow,
\quad
\mu\downarrow,
\quad
\lambda\uparrow.
\]

### Risk 3: Pivot-NTK score is too expensive

Use Score A first. Then use Score E only for small datasets or with last-layer gradient approximations.

### Risk 4: Score behaves like MaxLoss

Increase redundancy penalty:

\[
\lambda\uparrow.
\]

Use more pivots:

\[
r_H\uparrow.
\]

Add feature novelty:

\[
\mu>0.
\]

### Risk 5: Pivots from \(T_m\) are uninformative early on

Use fallback rule:

- for \(|T_m|<r_{\min}\), use clipped MaxLoss with feature diversity over \(T_m\);
- once \(|T_m|\ge r_{\min}\), activate Pivot-NTK score.

This is still deterministic and safe.

---

## 14. Final recommended algorithm for first implementation

Use PU-P2L-ES with Score E.

At iteration \(m\):

1. Train/update model:
   \[
   h_m=L(T_m).
   \]

2. Compute violations:
   \[
   V_m(z)=\mathbf{1}\{\ell(h_m,z)>\gamma\}.
   \]

3. If no violations, Stop.

4. Choose hard pivots:
   \[
   P_m^H=\operatorname{Top}_{r_H}\{\ell(h_m,p):p\in T_m\}.
   \]

5. For each candidate \(z\in D\setminus T_m\), compute:
   \[
   s_\ell(z)
   =
   \operatorname{clip}\left(\frac{\ell(h_m,z)}{\gamma},0,c_\ell\right),
   \]

   \[
   A_H(z)
   =
   \frac{1}{|P_m^H|}
   \sum_{p\in P_m^H}
   \max\{K_m^{\mathrm{LL}}(z,p),0\},
   \]

   \[
   R_H(z)
   =
   \max_{p\in P_m^H}
   \max\{K_m^{\mathrm{LL}}(z,p),0\},
   \]

   \[
   N_\phi(z)
   =
   1-\max_{p\in P_m^H}
   \operatorname{sim}_\phi(z,p).
   \]

6. Score:
   \[
   q_E(z)
   =
   s_\ell(z)
   +
   \alpha A_H(z)
   +
   \mu N_\phi(z)
   -
   \lambda R_H(z).
   \]

7. Select:
   \[
   z_m
   =
   \arg\max_{z\in D\setminus T_m}
   \left(
   V_m(z),
   q_E(z),
   \tau(z)
   \right).
   \]

8. Update:
   \[
   T_{m+1}=T_m\cup\{z_m\}.
   \]

Default:

\[
r_H=5,
\quad
c_\ell=3,
\quad
\alpha=0.5,
\quad
\mu=0.25,
\quad
\lambda=0.5.
\]

For \(|T_m|=0\), set \(q_E(z)=s_\ell(z)\), so the first selected point is the clipped MaxLoss point.

---

## 15. Deliverables after running experiments

The expected deliverables are:

1. **Certification plot**
   - test risk and bound versus train/pretrain fraction.

2. **Noise robustness plot**
   - test error, bound, \(k_{\mathrm{eff}}\), noise-hit rate versus noise.

3. **Training dynamics plot**
   - NTK coverage, effective rank, inappropriate-point decay versus compression size.

4. **Compression set analysis**
   - Jaccard overlap, diversity, selected-set visualization.

5. **Ablation table**
   - score function, pivot number, clipping, redundancy weight.

Suggested ablation table:

| Method | Score | \(r_H\) | \(c_\ell\) | \(\alpha\) | \(\mu\) | \(\lambda\) | Test error | Bound | \(k_{\mathrm{eff}}\) | Noise-hit | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MaxLoss-P2L-ES | loss | - | - | - | - | - | | | | | |
| PU-P2L-ES-A | feature diversity | 5 | 3 | - | 0.25 | 0.5 | | | | | |
| PU-P2L-ES-B | pivot NTK | 5 | 3 | 0.5 | - | 0.5 | | | | | |
| PU-P2L-ES-E | robust pivot utility | 5 | 3 | 0.5 | 0.25 | 0.5 | | | | | |

---

## 16. Reference anchors for writing

- **P2L:** The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance, NeurIPS 2023.
- **P2L-ES / GP:** Pick-to-Learn and Self-Certified Gaussian Process Approximations, AISTATS 2025.
- **GREATS:** GREATS: Online Selection of High-Quality Data for LLM Training in Every Iteration, NeurIPS 2024.
- **GraNd / EL2N / NTK diagnostics:** Deep Learning on a Data Diet: Finding Important Examples Early in Training, NeurIPS 2021 / arXiv v2.
