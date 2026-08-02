# Consolidated report — FNF and Fair PCA under the two-tier gauntlet

2026-07-24. Two runs completed under full discipline; one optional run skipped.
Predictions were committed at **8b7d873 before any implementation**; every
verdict below is from our own runs — no shipped number is borrowed for any
cell result. Nothing here changes the submission. Paper use requires author
sign-off.

| artifact | file |
|---|---|
| FNF registration / results / data | `fnf_prediction.md`, `fnf_results.md`, `fnf_gauntlet_hard_pca16.json` |
| Fair PCA registration / results / data | `fairpca_prediction.md`, `fairpca_results.md`, `fairpca_gauntlet.json` |
| Code | `fnf_stage_a.py`, `fnf_stage_b.py`, `run_fairpca_gauntlet.py`, `utils/fnf_shim.py` |
| Commits | 8b7d873 (registration) → a9aa64d (fair PCA) → 393edc3 (FNF) |

## Run 1 — FNF (eth-sri/fnf @ f574012, Apache-2.0)

**Gate: PASSED.** Their Table-1 Adult row reproduced with their own scripts at
their seed 100: at γ = 0.02, balanced accuracy **0.8447** vs published 0.8443
and adversarial accuracy **0.5971** vs published 0.5956 — both inside the 0.01
gate; their bound reproduces at 0.6121 vs 0.6112.

**Applicability.** FNF needs binary s *and* needs s at inference to pick the
encoder, so the 5-class-race cells are structurally out of scope; they were
**not** binarized. On the hard cell at its own 105 features FNF is
**inapplicable**: its density-estimation step fails (5/15 configs abort on a
non-PD covariance; the other 10 reach Δ = 1.000 with FNF's own adversary at
1.000). A best-shot variant (PCA to 16 dims, 85% variance retained — the
analogue of their own documented feature-dropping practice) trains cleanly and
carries the scoring.

**Scoring.** P1 **FALSIFIED** (I predicted Tier-1 passes at 1–2 gammas; **zero**
pass — every gamma reads rep Tier-1 0.999–1.000). P2 **HELD** but only
trivially, since Tier 2 never had to do any work. P3 **HELD** at 4/5 gammas —
measured recovery exceeds FNF's own density-conditional bound by up to
**0.174**.

**The finding worth keeping.** The bound violation appears in *FNF's own logged
outputs*: their own MLP adversary reads 0.989–0.999 at configs where their own
logged Δ implies a bound of 0.807–0.900. Our battery widens the gap but did not
create it. This is **not** a refutation of their theorem — their guarantee is
stated over estimated densities and Theorem 5.5 explicitly leaves an
intractable ε for the true-density gap, and on their own datasets the bounds
held. It shows the guarantee's *precondition* failing on a realistic
higher-dimensional cell. Contrast with FARE, whose certificate was coherent
inside its own scope: FNF's is honest in statement but unsound in practice
here.

## Run 2 — Fair PCA (amazon-science/fair-pca @ 18fac69, Apache-2.0)

**Gate: PASSED.** Their shipped pure-Python example runs unmodified, and the
method's exact defining invariant reproduces: projected group-conditional means
coincide to **1.5e-16**, and their Section-3.6 variant cuts covariance
deviation **185.20 → 0.206**. Their Tables 1–4 need Matlab and R and were not
used; their example is unseeded, so it publishes no fixed number to match
within 0.01 — stated rather than papered over.

**Applicability.** Their code raises `ValueError` on multiclass s (verified by
calling it), so — the same wall as FNF — only the hard cell runs, with race
cells left un-binarized.

**Scoring.** P1 **FALSIFIED as literally registered**, but with the substantive
claim intact: fair PCA fails Tier 1 at *every one* of 30 sweep points (minimum
0.696, never within 0.14 of the bar), yet my registered "≥ 0.90, LEACE-like"
threshold is wrong at the smallest target dimension. P1b **FALSIFIED**:
protection *does* improve monotonically as the dimension shrinks
(0.996 → 0.986 → 0.889 → 0.764 → 0.696 for dim 32 → 2), which is plain
dimensional bottlenecking, not the fairness constraint, and it costs utility
(certified point keeps 44.9% and still fails both tiers at 0.677).
It passes nowhere, so there is no surprise to report.

## Run 3 — PAT / vertical data minimization (eth-sri/datamin): SKIPPED

Not run, and deliberately not started: doing it properly needs its own
registration-before-implementation and a real gate, and the repo vendors its
own scikit-learn and sktree behind a conda `environment.yml` — a FARE-class
environment build. Reconnaissance only (no run, no numbers): **MIT licensed**,
so unlike FARE it may be reused freely. It remains a scope sentence either way.

## Cross-cutting observations

1. **Two independent published methods, one identical structural wall.** FNF
   and fair PCA both hard-require binary s. Our channel handles the 5-class
   race cells natively. That is a scope advantage that has not been stated in
   the paper and costs nothing to state.
2. **Two certificates, two different failure modes.** FARE: coherent inside its
   scope, expensive frontier. FNF: honest statement, precondition unmet, bound
   exceeded by up to 0.17 on a real cell. Both support verification-relativity,
   but for different reasons — worth distinguishing rather than merging.
3. **A registered numeric threshold got falsified in the useful direction.**
   Fair PCA is meaningfully more protective than LEACE (0.70 vs 0.995) while
   still failing everywhere. Had the threshold not been registered in advance,
   the temptation would have been to report the qualitative claim as a clean
   win.

## Accounting delta if adopted (reported, NOT applied)

Published-method combinations **42 → 46** (FNF × hard × 2 tiers; fair PCA ×
hard × 2 tiers). Passes remain **7** — neither method certifies anywhere. Four
cell-level entries are recorded as *structurally inapplicable* rather than as
failures (FNF and fair PCA on easy and middle), plus one *inapplicable at full
dimension* entry for FNF on hard. The submission does not change.
