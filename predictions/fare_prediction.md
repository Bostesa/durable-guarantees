# Registered prediction — FARE under the two-tier gauntlet (committed before any FARE run)

Addresses blind-reviewer R1 W4: FARE (Jovanovic et al. 2023) is the closest
certified competitor (practical certificates for restricted downstream families)
and is cited but not evaluated in Table 2. This runs FARE under the identical
two-tier protocol used for the other published baselines.

## Pinned protocol

- Code: official eth-sri/fare at commit 89cb1b66ed268c16659cbf7428c43e60da2df641,
  patched fair-Gini sklearn (v1.2.dev0 @ fd60379f) built in
  $FARE_ROOT/.venv (py3.9.12, arm64, no OpenMP). No license file in
  the repo — run-and-cite only, no redistribution of their code.
- Reproduction gate FIRST: shipped result ACSIncome-CA-2014, config
  k=50, min_ni=100, alpha=0.9, val_split=0.3 → must reproduce dp_ub within 0.01
  of 0.15712 and tree test accuracy within 0.01 of 0.7959 before any cell run.
  If the gate fails, stop and report; borrow nothing.
- Cells: the three headline cells (easy hmda/race/loan_decision, middle
  hmda/race/loan_amount_band, hard adult/sex/income), raw features from
  load_cell — the same from-raw-x input the other trained baselines consume;
  cat_pos = [] (features are already preprocessed floats).
- FARE grid (their shell sweeps, condensed): max_k in {5, 10, 25, 50, 100} x
  alpha in {0.5, 0.75, 0.9, 0.975, 0.999}, min_samples_leaf=100, internal
  train/val split 70/30 (numpy seed 42+ts), tree random_state 43+ts.
  "Training seeds" ts in {0,1,2}: sweep at ts=0, certification at all three —
  mirroring baseline_gauntlet.py (probe seeds [0,1] sweep / [0,1,2] cert,
  bar 0.55, rep-gated verdicts, MAX_CERT_CANDIDATES=2 per tier).
- Exposures (LEACE-shaped deterministic row): P = leaf-median embedding z for
  ALL rows (T.apply + per-leaf train medians); L = log-probs of a logistic
  head retrained on z (lift_own), retrained-LR probe utility (lift_lr);
  ncov = 0 (deterministic channel → the Tier-2 LRT degenerates to an informed
  QDA, exactly as for LAFTR/LEACE/DANN).
- Per-config FARE certificate dp_ub (AlphaBetaAdversary, err_budget 0.05,
  eps 0.005/0.005, method 'cp'; all-pairs with Bonferroni-split budget for
  multiclass s) logged next to the attacker readings.
- Two-stage execution (venv incompatibility): stage A fits all 25 x 3 trees per
  cell in the FARE venv and writes leaf assignments + leaf medians + dp_ub to
  npz; stage B scores the battery in the project venv.

## Registered claims

1. **Gate.** The reproduction gate holds — dp_ub within 0.01 of 0.15712 (seeds
   are hard-coded in their code, so near bit-exact) and accuracy within 0.01.
2. **Frontier by coupling.** On the easy cell (coupling 0.514) some operating
   point passes the Tier-1 rep bar at >= 40% utility kept (vs the cell's stored
   clean lift). On the middle (0.584) and hard (0.603) cells, any config passing
   the rep bar retains <= 50% (middle) and <= 35% (hard) utility — the tree can
   only balance its cells in s by discarding the y-information that carries s.
   Whether the passing configs survive 3-seed certification is genuinely
   uncertain; if none certifies, FARE joins the does-not-certify rows like the
   other four non-VFAE baselines.
3. **Tiers collapse.** Tier-2 adds <= 0.02 AUC over Tier-1 max at every FARE
   operating point (finite-support deterministic z: the informed QDA has no
   channel noise to exploit and XGB is already near-Bayes on a leaf-partition),
   so no config passes Tier 1 but fails Tier 2 by more than 0.02.
4. **Certificate coherence (the interesting one).** FARE's dp_ub certifies
   downstream-classifier DP, not attribute recovery; still, because both track
   how s-imbalanced the leaf cells are, predict: per cell, Spearman rank
   correlation >= 0.6 between dp_ub and rep tier1_max across the 25-point grid,
   and every config with dp_ub <= 0.10 reads rep tier1_max <= 0.60. Unlike the
   R^2 and second-moment certificates the paper attacks, FARE's certificate
   should NOT be flatly contradicted by the battery in its own scope — the
   expected finding is "honest certificate, costly frontier", not "lying
   certificate".

Acknowledged uncertainties: FARE was designed for binary s (multiclass DP
criterion is data-driven but less exercised — hmda race is 5-class); leaf-median
embeddings on standardized continuous features may behave differently from
their ordinal-coded categorical setting; claim-2 thresholds are coarse
judgement calls, not derived.
