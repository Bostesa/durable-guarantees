# FNF (Fair Normalizing Flows) under the two-tier gauntlet — results and scoring

Run 2026-07-24, local CPU. Registered prediction committed at 8b7d873 BEFORE
any run on our cells and before implementation (`fnf_prediction.md`).
Pipeline: `fnf_stage_a.py` (FNF venv, trains their encoders on our cell) →
`fnf_stage_b.py` (project venv, scores the battery), the same two-stage
structure as the FARE run. Data: `fnf_gauntlet_hard_pca16.json`,
`fnf_cells/*_stage_a.json`, gate CSV `fnf_cells/gate_adult_seed100.csv`.

Code: official eth-sri/fnf @ f574012, **Apache-2.0** (verified — unlike FARE,
which ships no license, so this code may be reused freely). Their tree is NOT
edited. Run in a period-appropriate venv (py3.11, numpy 1.23.5, pandas 1.5.3),
under which their code runs natively; the only shim
(`utils/fnf_shim.py`) restores `torch.load(weights_only=False)`, needed because
torch ≥ 2.6 flipped that default and their pipeline reloads the MADE prior
pickle it wrote itself minutes earlier in the same run.

## a. Gate — PASSED

Their Table-1 Adult FNF row (Acc **84.43**, Adv Acc **59.56**, Max Adv Acc
**61.12**, implying Δ ≈ 0.2224) reproduced with their own scripts
(`train_gen_categorical.py` → `train_enc_categorical.py` at the shipped
`scripts/run_adult_gamma.py` settings), their seed 100, their shipped gamma
grid. The paper does not say which gamma produced Table 1, so the gate searched
their grid:

| gamma | Δ (stat_dist) | test bal-acc | adv test acc | bound (1+Δ)/2 |
|---|---|---|---|---|
| 0 | 0.2275 | 0.8458 | 0.5975 | 0.6137 |
| **0.02** | **0.2242** | **0.8447** | **0.5971** | **0.6121** |
| 0.1 | 0.2111 | 0.8406 | 0.5950 | 0.6056 |
| 0.2 | 0.1951 | 0.8351 | 0.5844 | 0.5976 |
| 0.9 | 0.1196 | 0.7944 | 0.5475 | 0.5598 |

γ = 0.02 matches Table 1 on **both** required quantities: accuracy 0.8447 vs
0.8443 (|Δ| = 0.0004) and adversarial accuracy 0.5971 vs 0.5956 (|Δ| = 0.0015);
their bound reproduces at 0.6121 vs 0.6112. All within the 0.01 gate. Table 1's
"Acc" is their *balanced* test accuracy (unbalanced is 0.8217), recorded so the
convention is not ambiguous. Nothing is borrowed — every cell number below is
our own run.

## b. Applicability — hard cell only; race cells NOT binarized

FNF trains two group-conditional bijective encoders and structurally requires
binary s; it also requires s **at inference time** to select the encoder (a
deployment-scope caveat worth stating on its own). So easy and middle
(5-class race) are out of scope and were not binarized. Only hard
(adult/sex/income) is run.

**Full-dimension result (the faithful-to-cell run): FNF is INAPPLICABLE.**
On the cell's own 105 features, their continuous path fails at its density
estimation prerequisite: 5/15 configs abort outright
(`ValueError: covariance_matrix ... PositiveDefinite` — PCRL's adult encoding
contains near-constant one-hot columns, so the full-covariance GMM is
singular), and the 10 that do fit reach **Δ = 1.000 with their own adversary at
1.000** at every gamma. That is the failure mode registered in advance; per the
registration the honest reading is "not applicable to this input distribution",
NOT "FNF fails".

**Best-shot variant (reported as a deviation in FNF's favour):** PCA to 16
dims first (retaining 85.2% of variance), the analogue of their own documented
practice of "dropping uninformative features to facilitate a good density
estimate" (their continuous cells use ~6 features). Here FNF trains cleanly at
all 15 configs and its gamma knob behaves as published (Δ falls monotonically
with gamma). **All scoring below is on this best-shot variant.**

## c. Readings — hard cell, PCA-16, clean lift +0.1435

3 training seeds (their 100/101/102), probe seeds [0,1,2], bar 0.55.
AUC is our standing metric; balanced accuracy is theirs.

| gamma | rep T1 / T2 (AUC) | out T1 (AUC) | μ* (AUC) | best bal-acc | Δ | their bound | utility kept |
|---|---|---|---|---|---|---|---|
| 0 | 1.000 / 1.000 | 0.784 | 1.000 | 0.998 | 0.999 | 1.000 | 66.6% ± 0.5 |
| 0.02 | 1.000 / 1.000 | 0.772 | 0.946 | 0.991 | 0.800 | 0.900 | 66.2% ± 0.5 |
| 0.1 | 1.000 / 1.000 | 0.722 | 0.900 | 0.985 | 0.682 | 0.841 | 63.8% ± 1.1 |
| 0.2 | 0.999 / 0.999 | 0.700 | 0.880 | 0.981 | 0.631 | 0.816 | 61.9% ± 0.2 |
| 0.9 | 0.999 / 0.999 | 0.630 | 0.862 | 0.981 | 0.615 | 0.807 | 46.1% ± 3.5 |

## d. Certificate coherence — the headline finding

At every gamma except 0 (where the bound is the vacuous 1.000), our measured
best-attacker balanced accuracy **exceeds FNF's own density-conditional
bound**, by 0.091 / 0.144 / 0.165 / 0.174 at gamma 0.02 / 0.1 / 0.2 / 0.9.

Crucially, **this is visible in FNF's own logged outputs before our battery is
involved**: their own trained MLP adversary reads 0.989–0.999 at exactly the
configs where their own logged Δ implies a bound of 0.807–0.900. Our attackers
make the gap wider, but they did not create it.

**Precise scope — this is not a refutation of their theorem.** FNF's guarantee
is stated over the *estimated* densities p̂₀, p̂₁; their Theorem 5.5 explicitly
adds an ε for the TV gap to the true densities and notes the constants are
intractable. On their own datasets (their Figure 5) the bounds held. What this
run shows is that the guarantee's *precondition* — a good density estimate —
fails on a realistic higher-dimensional cell, and that when it fails the
certificate is off by up to 0.17 absolute rather than by a negligible margin.
This is the paper's verification-relativity point in its sharpest form so far:
FARE's certificate was coherent within its own scope; FNF's is honest in
statement but unsound in practice on this cell, because the quantity it is
conditioned on cannot be estimated well here.

## Prediction scoring

- **P1 (Tier-1 passes at 1–2 gammas, all with ≤ 40% utility) — FALSIFIED.**
  **Zero** gammas pass Tier 1; every gamma reads rep T1 = 0.999–1.000, i.e. the
  release is *fully* attribute-readable by a black-box XGB/MLP. My predicted
  range was far too generous. The two sub-conditions I attached did hold
  vacuously (no passing point has > 40% utility; every > 60%-utility point
  fails Tier 1), but the headline claim is falsified: FNF does not reach our
  bar on this cell at any operating point. Mechanism: the encoders are
  *bijections*, so the release retains all information in x; protection depends
  entirely on distribution matching, and on this cell the match is poor
  (Δ ≥ 0.615 even at gamma 0.9).
- **P2 (Tier-2 fails at every point retaining > 40% utility) — HELD.**
  All five gammas retain > 40% utility and all five fail Tier 2 (0.999–1.000).
  Held, though trivially so given P1's collapse — Tier 2 never had to do any
  work, which weakens the claim's evidential value rather than strengthening it.
- **P3 (at least one config measured above FNF's own bound) — HELD**, at 4 of
  5 gammas, with margins up to 0.174, and visible in their own adversary's
  numbers. This is the run's substantive contribution.
- **P4 (failure modes named in advance):** the registered global failure mode —
  "the flows may fail to train at all on PCRL-preprocessed features, in which
  case the honest reading is *inapplicable*, not *defeated*" — is exactly what
  happened at full dimension, and is reported as such rather than scored as a
  win.

## The Table-2-shaped row this would produce

| method | cell | tier | operating point | utility kept | rep T1 / T2 | out T1 | verdict | own certificate |
|---|---|---|---|---|---|---|---|---|
| FNF | easy, middle | — | — | — | — | — | **inapplicable** (binary s required; s needed at inference) | — |
| FNF | hard (105-d, faithful) | T1/T2 | — | — | — | — | **inapplicable** (density estimate not PD; Δ = 1.000) | vacuous |
| FNF | hard (PCA-16, best shot) | T1 | γ = 0.9 | 46.1% | 0.999 / 0.999 | 0.630 | **FAIL** | bound 0.807 < measured 0.981 |
| FNF | hard (PCA-16, best shot) | T2 | γ = 0.9 | 46.1% | 0.999 / 0.999 | 0.630 | **FAIL** | " |

## Accounting delta if adopted (reported, not applied)

Published-method combinations 42 → 44 (FNF × hard × 2 tiers, best-shot
variant); passes unchanged at 7. Two cells recorded structurally inapplicable,
plus one cell-level inapplicability at full dimension. Requires Nathan +
advisor sign-off before any paper use.
