# Fair PCA under the two-tier gauntlet — results and scoring

Run 2026-07-24, 4.8 min, local MPS. Registered prediction committed at 8b7d873
BEFORE implementation (`fairpca_prediction.md`). Script
`run_fairpca_gauntlet.py`, log `run_fairpca_gauntlet.log`, data
`fairpca_gauntlet.json`. Code: official amazon-science/fair-pca @ 18fac69,
**Apache-2.0** (verified), called unmodified from `src/fair_pca/fair_PCA.py`.

## Gate — PASSED

Their pure-Python shipped example runs unmodified (the only adjustment is
`PYTHONPATH=<repo root>`, which is their own documented "set the project root
as your working directory" requirement — Python puts the *script's* directory
on `sys.path`, not the cwd). Their Matlab/R paths for Tables 1–4 are not
reproducible on this machine and were not used; their example seeds its data
with an unseeded `np.random.normal`, so it publishes no fixed number to match
within 0.01 — stated rather than papered over. The gate therefore checks the
method's exact, seed-independent defining invariant:

| check | result | verdict |
|---|---|---|
| shipped `illustration_Figure1.py` completes and prints its diagnostics | rc=0; recon err 270.69 / 444.09, cov dev 228.12 / 0.124 | PASS |
| projected group-conditional means coincide (their `null_space(Z^T X)` constraint) | relative max gap **1.465e-16** | PASS |
| Section-3.6 variant reduces group-conditional covariance deviation vs Section 2 | **185.20 → 0.206** | PASS |

## Applicability — hard cell only, race cells NOT binarized

`FairPCA.fit` / `check_inputs` raise `ValueError: Array with potected
attributes must be a 1-dim numpy array with values in {0,1}` on a multiclass
attribute — verified by calling it, not assumed. So, exactly as with FNF, the
5-class-race cells (easy, middle) are **structurally out of scope** and were
not binarized (that would change the cell definition and break comparability
with every other Table-2 row). Only hard (adult/sex/income) is run.

## Readings — hard adult/sex/income, clean lift +0.1435

Sweep (train seed 0, probe seeds [0,1]), both exposure surfaces, 30 points.
Tier-1 max by target dimension at `tradeoff_param = 0` (the most protective
setting; larger values append `tradeoff³ ×` the standard-PCA block and are
strictly less protective by construction):

| target_dim | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| min Tier-1 max over surfaces | **0.696** | 0.764 | 0.889 | 0.986 | 0.996 |

At `tradeoff_param ∈ {0.25, 0.5}` every point reads 0.977–1.000 on both
surfaces. Full sweep range: **0.696 – 1.000**; no point anywhere reaches the
0.55 bar.

Certified operating point (best available, 3 training seeds, probe seeds
[0,1,2]) — the least-leaky point in the whole sweep:

| point | rep T1 / T2 | out T1 / T2 | utility kept | verdict |
|---|---|---|---|---|
| post-hoc P, dim 2, tradeoff 0 | 0.677 / 0.677 | 0.599 / 0.599 | 44.9% | **FAILS both tiers** |

## Prediction scoring

- **P1 — FALSIFIED as literally registered; substantive claim HELD.** I
  registered "fails Tier 1 on every cell, at or near the unprotected level
  (**Tier-1 max ≥ 0.90**)". The two halves separate cleanly:
  - *Fails Tier 1 everywhere*: **HELD decisively** — the minimum reading
    anywhere in a 30-point sweep is 0.696, never within 0.14 of the 0.55 bar,
    and the certified point fails both tiers on both surfaces.
  - *At or near unprotected (≥ 0.90)*: **FALSIFIED** — at the smallest target
    dimension the reading is 0.696 (raw-X) / 0.677 (certified post-hoc), not
    ≥ 0.90. Fair PCA is meaningfully more protective than LEACE
    (0.995–0.998), just nowhere near enough to certify. The literal numeric
    threshold was too strong; the scoring reports FALSIFIED and this is why
    thresholds get registered rather than asserted afterwards.
- **P1b — FALSIFIED.** I registered "protection does not improve monotonically
  as target_dim shrinks; Tier-1 stays ≥ 0.80 at every swept dimension". It
  does improve, cleanly and monotonically (0.996 → 0.986 → 0.889 → 0.764 →
  0.696 as dim goes 32 → 2), breaching 0.80 at both dim 2 and dim 4. The
  mechanism is not the fairness constraint but plain dimensional bottlenecking:
  a 2-dim projection of a 105-dim input destroys attribute signal the way any
  aggressive compression would, and it costs utility (44.9% kept) without
  approaching the bar.
- **"If it passes anywhere, that is the surprise":** it does not pass
  anywhere. No surprise to report.

## Where this lands

Fair PCA joins the paper's "≈ unprotected at high utility" quadrant with
LEACE, but as a distinctly softer version of it: a single linear constraint
plus dimensional bottlenecking moves recovery from ~1.00 down to ~0.70 at the
extreme, and no further. Its constraint is scoped to downstream selection-rate
parity, not attribute unrecoverability — the same certificate-scope pattern
the FARE run found, now in a closed-form linear method.

## Accounting delta if adopted (reported, not applied)

Published-method combinations 42 → 44 (fair PCA × hard cell × 2 tiers);
passes unchanged at 7. Two cells (easy, middle) are recorded as structurally
inapplicable rather than as failures. Requires author sign-off.
