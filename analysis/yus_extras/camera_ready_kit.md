# Camera-ready / rebuttal kit — v28 reviewer responses

Generated 2026-07-23. Every number below is computed from committed results
JSONs (script fragments inline in the session log; sources named per table).
Addresses the convergent asks of the three simulated reviews
(`simulated_reviews_v28.md`).

## 1. Bar and τ sensitivity (R1-W7, R2-W2, R2-Q2)

Sources: `honest_reaudit.json` (existing 21, suite = max(XGB, MLP)),
`expansion_reaudit_paperframing.json` (new 46, suite = tier1_max). The
bar=0.55 row reproduces the paper's 59/67 and 1/36 exactly.

**Audit verdict vs protection bar (approvals fixed at τ=0.05):**

| bar | fail | survive |
|---|---|---|
| 0.52 | 64/67 | 3 |
| **0.55** | **59/67** | **8** |
| 0.60 | 51/67 | 16 |

**Published-method combos passing vs bar:**

| bar | passes / 36 | who |
|---|---|---|
| 0.52 | 0 | — |
| **0.55** | **1** | VFAE easy T1 |
| 0.60 | 4 | VFAE easy/middle/hard T1 + VFAE easy T2 — all via the incidental noise channel |

**Approvals vs τ** (new-cells partition, instrument-exact: stored at-rest R²):

| τ | approved | fail at 0.55 bar |
|---|---|---|
| 0.01 | 5 | 2 (40%) |
| **0.05** | **46** | **41 (89%)** |
| 0.10 | 63 | 58 (92%) |

(Existing-partition τ curve via its stored attacked-R² instrument: τ=0.01
approves 3, all σ≥2 noise; τ=0.10 approves 20 of 21. Note the instrument
caveat if quoting combined counts.)

**Suggested rebuttal sentence:** "The verdicts are threshold-robust: moving
the bar to 0.52/0.60 gives 64/51 of 67 failing (published methods: 0/4 of 36
passing, every additional pass at 0.60 being VFAE's incidental noise channel),
and tightening τ to 0.01 shrinks approvals to the high-noise configurations the
suite already favors while the failure fraction stays 40–92%."

## 2. Correlation uncertainty (R3-W2, R3-Q2)

Source: `continuous_cost.json` + `expansion_cost.json`, 27 natural cells,
6 datasets. Seed 0, 10k resamples.

- r = 0.799: Fisher 95% CI **[0.602, 0.904]**; cell-level bootstrap
  **[0.654, 0.904]**; dataset-cluster bootstrap (6 clusters)
  **[0.696, 0.932]**.
- Aggregating to one point per dataset (n=6): r = **0.864** — the correlation
  strengthens, not weakens, under clustering.
- External-only (n=7): r = 0.872, Fisher 95% CI **[0.348, 0.981]** — wide, as
  the reviewer suspected; quote with the CI or drop the third digit.

**Suggested text:** "Across 27 natural cells r = 0.80 (95% CI 0.60–0.90;
dataset-cluster bootstrap 0.70–0.93; aggregating to one point per dataset,
r = 0.86), and the seven pre-registered external cells alone give r = 0.87
(n = 7, CI 0.35–0.98)."

## 3. DP comparison appendix table (R1-W4/Q4, R2-W3/Q3)

Source: `analysis/yus_extras/dp_fullrank.json` (registered bet e87332b,
results 538c840). Clip-in-the-loop full-rank Gaussian channel: release =
clip_C(h) + σ·N(0, I₆₄), C = ε·σ/(2√(2 ln(1.25/δ))), δ = 10⁻⁵, classical
calibration (conservative above ε=1), 5 training seeds, same architecture /
optimizer / utility convention as the full-rank arm.

```latex
\begin{table}[t]
\centering\small
\caption{A formally private variant of the full-rank channel: utility kept
(\% of clean lift, 5 seeds) for the per-release Gaussian mechanism with
clip-in-the-loop, $\delta=10^{-5}$. At $\varepsilon\le 3$ no cell retains
useful utility; the middle cell reaches its Tier-2 utility (21.7\%) only
between $\varepsilon=3$ and $\varepsilon=6$. Tier-2 anchors: 55.5 / 21.7 /
$-7.0$.}
\label{tab:dp}
\begin{tabular}{llrrrr}
\toprule
cell & $\sigma$ & $\varepsilon{=}0.5$ & $\varepsilon{=}1$ & $\varepsilon{=}3$ & ($\varepsilon{=}6$) \\
\midrule
easy   & 8  & $+0.0$ & $+0.0$ & $+0.0$ & $-0.1$ \\
easy   & 32 & $-0.0$ & $-0.0$ & $-0.0$ & $-0.1$ \\
middle & 8  & $-1.3$ & $+0.3$ & $+8.8$ & $+22.6$ \\
middle & 32 & $-2.0$ & $-1.0$ & $+4.7$ & $+11.2$ \\
hard   & 8  & $+0.0$ & $+0.0$ & $-0.4$ & $+4.4$ \\
hard   & 32 & $-0.4$ & $-0.4$ & $-2.0$ & $-2.8$ \\
\bottomrule
\end{tabular}
\end{table}
```

Supporting facts for the prose: 76–100% of rows are clipped at ε≤1; median
pre-clip ‖h‖ grows with the clip radius (4.2 → 11.9), so norms are equilibrium
properties of the clip, not constants; σ=8 dominates σ=32 at fixed ε (tighter
clip + less absolute noise). Answer to "at what ε does DP match Tier 1":
nowhere on the grid — the best ε=6 reading (22.6% middle) is below every
Tier-1 point (94.6/56.6/−0.5 with the hard Tier-1 point ≈ 0 anyway).

**Replaces the two Future-work sentences with:** "A formally private variant
exists: clip the representation and calibrate full-rank noise to an
$(\varepsilon,\delta)$ budget (Table~\ref{tab:dp}). At conventional budgets
($\varepsilon\le 3$, $\delta=10^{-5}$) it retains essentially no task utility
on our cells; it reaches our Tier-2 utility only on the middle cell at
$\varepsilon\approx 5$–$6$ per release, and no point on the grid approaches
Tier-1 utility. The averaging curve of Appendix~B is exactly the composition
cost such a budget would meter."

## 4. Reconciliation table (R3-W4/Q3, R1-W6)

```latex
\begin{table}[t]
\centering\small
\caption{Where each quoted utility number comes from. Arm: e2e = trained
through the channel, post-hoc = channel applied to a frozen representation.
Channel: sub = subspace-confined (isolate-then-noise), full = full-rank
noise. Convention: own-head accuracy on the training partition (in-sample)
vs.\ retrained linear probe on a held-out 25\% split.}
\label{tab:recon}
\begin{tabular}{lllll}
\toprule
number & arm & channel & $\sigma$ (e/m/h) & convention \\
\midrule
102 / 100 / 85\%          & e2e      & sub  & 16/16/16 (Tab.~5) & in-sample \\
108.8 / 101.6 / 87.3\%    & e2e      & sub  & 16/16/16          & held-out (Tab.~4) \\
94.6 / 56.6 / $-0.5$\%    & e2e      & full & 8/12/32 (Fig.~5)  & in-sample \\
55.5 / 21.7 / $-7.0$\%    & e2e      & full & 20/24/64 (Fig.~5) & in-sample \\
88 / 81 / 53\%            & post-hoc & sub  & rel.\ 16/16/16    & held-out \\
\bottomrule
\end{tabular}
\end{table}
```

Caption note to add to Table 5: "the subspace-confined (isolate-then-noise)
channel; the full-rank points of Fig.~5/Table~3 use $\sigma$ = 8–64."

## 5. Small text fixes

- **The 36 arithmetic (R3-Q1):** footnote to Table 2 — "36 = 6 method variants
  (LAFTR validated reimplementation, LAFTR official release, VFAE, DANN-scrub,
  LEACE, Obliviator) × 3 cells × 2 tiers; LAFTR's two variants share one table
  row."
- **MMD definition (R3-W9):** first use — "class-conditional maximum mean
  discrepancy (MMD), a kernel two-sample statistic matching class
  distributions".
- **"Causal" softening (R3-W3):** "which supports a causal relationship" →
  "consistent with a causal effect of coupling on these two base datasets".
- **Abstract "upper bound" (R3-W1):** "The prediction is an upper bound." →
  "The prediction is an upper bound, tight when the task is well learned."
  (three external cells sit 0.03–0.04 above the diagonal at the first passing
  grid level — already explained in Results; the abstract just needs the
  qualifier).
- **Appendix B 60k sentence (claims-audit residual):** "For cells above 60,000
  rows, attacker evaluation stratified-subsamples to 60,000 rows (seed 42)." →
  "External-source cells are loaded with a stratified 60,000-row cap (seed 42);
  the headline cells are evaluated on their full train partitions."
- **Dangling sweep pointer (R2-W4/Q6):** "the per-method sweeps are in the
  appendix" → "the per-method sweep grids and all attacker readings are in the
  reproducibility package (one JSON per method)."
- **Fig. 5 label:** regenerated — near-zero bars now print one decimal
  (−0.5%), matching the text (commit alongside this kit).

## 6. Remaining compute item (offered, not yet run)

R2's headline condition — TPR@{0.1%, 1%} FPR for the 8 surviving audit
configurations and the 36 baseline combos — needs score-capture re-runs of
those configs (sealed-channel and baseline models). Same pattern as
`run_tpr.py`; est. 60–90 min on MPS. Not started; say the word.
