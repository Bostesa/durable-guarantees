# Rebuttal evidence bundle — 2026-07-24 (self-contained)

Both reviewer-objection runs, complete: registered predictions (committed BEFORE
each run), scored results, and full raw readings. Single-file copy of:
analysis/yus_extras/{fresh_partition_prediction.md, fresh_partition_results.md,
fresh_partition_generalization.json, fare_prediction.md, fare_results.md,
fare_gauntlet.json, fare_cells/*_fare_manifest.json}.

RUN 1 — Fresh-partition generalization (blind-reviewer R2 W1): attackers scored
on rows the encoder never saw. Verdict: NO flips, max attacker delta 0.013 AUC.
Prediction commit 592c551; results commit da1c94d.

RUN 2 — FARE baseline under the two-tier gauntlet (blind-reviewer R1 W4):
eth-sri/fare @ 89cb1b6, reproduction gate held bit-exact (dp_ub 0.1571 vs
0.15712). FARE passes both tiers on all three cells at 0% / 28.3% / 39.0%
utility. Prediction commit ab76813; results commit 099cc2d.

---

# ============================================================
# FILE: analysis/yus_extras/fresh_partition_prediction.md
# ============================================================

# Registered prediction — fresh-partition generalization (committed before implementation)

Addresses blind-reviewer R2 W1: the audit measures attacker generalization across
held-out rows of the same partition the representation was trained on; an encoder
could in principle memorize the attribute for training rows, inflating in-partition
attacker AUC. This experiment re-scores the surviving operating points on rows the
representation model never saw.

## Pinned protocol

- Cells: easy = hmda/race/loan_decision, middle = hmda/race/loan_amount_band,
  hard = adult/sex/income.
- Split: 50/50 disjoint row partitions (representation partition / evaluation
  partition), single fixed split seed **20260724**, stratified on (attr, task)
  jointly. Split happens before any training.
- Training: encoder + head (+ subspace Q for the surgical channel) trained on the
  representation partition ONLY, at the already-registered operating points, no
  re-tuning:
  - e2e full-rank (train_channel replica): Tier-1 points sigma = 8 / 12 / 32,
    Tier-2 points sigma = 20 / 24 / 64 (easy/middle/hard).
  - e2e subspace (train_e2e_surgical replica): easy r=32 sigma=16 lam=10,
    middle r=16 sigma=16 lam=100, hard r=32 sigma=16 lam=100 (the
    hardening_seeds sigma=16 operating points behind the 102/100/85 triple).
  - Clean baseline sigma=0 per cell for fresh-partition clean lift.
  - 3 training seeds [0, 1, 2] per point.
- Scoring: full suite — XGB, MLP(256-256), rank-32 LoRA, channel-aware
  Gaussian-LRT — fit and scored entirely within the evaluation partition
  (standard 75/25 internal probe split, probe seeds [0, 1, 2]); the encoder never
  saw any evaluation-partition row. Bar 0.55 on the 3-seed mean, same as the
  paper. Utility = own-head lift on the evaluation partition relative to the
  sigma=0 clean lift on the same partition.
- Comparators: stored in-partition readings in results/two_tier_certification.json
  (e2e_cells tier1/tier2), results/hardening_seeds.json (subspace rep maxes,
  5-seed), analysis/yus_extras/isolate_vs_fullrank.json (subspace LRT, 3-seed).

## Registered claims

1. **No verdict flips.** All six full-rank points keep their tier verdicts
   (Tier-1 pass at the Tier-1 points; Tier-1 and Tier-2 pass at the Tier-2
   points), and all three subspace points still pass Tier-1 and still fail
   Tier-2 (LRT >= 0.60).
2. **Magnitude.** Fresh-partition Tier-1 max and LRT readings within +-0.02 of
   the stored in-partition means at every point (stored T1 maxes: full-rank
   0.535/0.541/0.531 at sigma=8/12/32; LRT at Tier-2 points 0.538/0.550/0.534;
   subspace rep maxes 0.509/0.531/0.536). Direction: fresh readings at or below
   in-partition + 0.01 — i.e., no memorization-driven inflation is present,
   because the protective mechanism is per-row fresh Gaussian noise, which
   applies identically to unseen rows, and there is no headroom for a large drop
   between these readings and chance.
3. **Risk cases named.** The only plausible flips are the two points nearest the
   bar: the hard-cell subspace point (stored 0.536) and the middle-cell
   full-rank Tier-2 LRT (stored 0.550). If any point flips, it is one of these,
   by <= 0.02.
4. **Utility.** Fresh-partition clean lift will be lower than the full-data
   clean lift (half the training rows), but the utility-kept ratios stay within
   +-10pp of the stored values (full-rank T1 95/57/-0.5%, T2 55/22/-7%;
   subspace 102/100/85%). The hard cell's near-zero denominators make its
   percentage the most volatile; the claim for the hard cell is on lift sign and
   magnitude (within +-0.02 absolute lift), not the ratio.

Main acknowledged uncertainty: half-data training shifts BN statistics and clean
lift, which can move readings near the bar by ~0.01-0.02 — this is why claim 3
names the two nearest-bar points rather than promising zero flips at 0.001
precision.

# ============================================================
# FILE: analysis/yus_extras/fresh_partition_results.md
# ============================================================

# Fresh-partition generalization — results and scoring of the registered claims

Run: 2026-07-24, analysis/yus_extras/run_fresh_partition.py, split seed 20260724,
50/50 stratified (attr, task); prediction committed at 592c551 BEFORE the run.
Data: fresh_partition_generalization.json. Verdicts on 3-seed means, bar 0.55,
probes fit and scored entirely inside the evaluation partition (never seen by
the encoder).

## Readings (fresh partition vs stored in-partition)

| cell | channel | T1 max (fresh / stored / delta) | LRT (fresh / stored / delta) | verdicts | util_ev% (fresh / stored) |
|---|---|---|---|---|---|
| easy | fullrank sigma=8 (T1 pt) | 0.533 / 0.535 / -0.003 | 0.591 / 0.583 / +0.008 | T1 pass | 70.8 / 94.6 |
| easy | fullrank sigma=20 (T2 pt) | 0.510 / 0.509 / +0.001 | 0.538 / 0.538 / +0.000 | T1+T2 pass | 69.7 / 55.5 |
| easy | subspace r32 s16 l10 | 0.517 / 0.509 / +0.008 | 0.861 / 0.854 / +0.007 | T1 pass, T2 fail | 64.0 / 102.2 |
| middle | fullrank sigma=12 (T1 pt) | 0.541 / 0.541 / +0.000 | 0.585 / 0.585 / +0.000 | T1 pass | 57.2 / 56.6 |
| middle | fullrank sigma=24 (T2 pt) | 0.516 / 0.512 / +0.004 | 0.543 / 0.550 / -0.006 | T1+T2 pass | 21.6 / 21.7 |
| middle | subspace r16 s16 l100 | 0.534 / 0.531 / +0.003 | 0.858 / 0.864 / -0.006 | T1 pass, T2 fail | 96.6 / 100.0 |
| hard | fullrank sigma=32 (T1 pt) | 0.525 / 0.531 / -0.007 | 0.565 / 0.578 / -0.013 | T1 pass | -3.0 / -0.5 |
| hard | fullrank sigma=64 (T2 pt) | 0.511 / 0.506 / +0.005 | 0.529 / 0.534 / -0.005 | T1+T2 pass | -9.6 / -7.0 |
| hard | subspace r32 s16 l100 | 0.543 / 0.536 / +0.008 | 0.672 / 0.662 / +0.010 | T1 pass, T2 fail | 96.8 / 85.3 |

Fresh-partition clean lifts (3-seed, eval partition): easy +0.0152, middle
+0.3966, hard +0.0687 (in-sample rep-partition: +0.0273 / +0.4163 / +0.1853).
Utility convention here is out-of-sample own-head over out-of-sample clean; the
stored percentages are the in-sample full-data convention — denominators differ.

## Claim scoring

1. **No verdict flips — HELD.** flips list is empty. All six full-rank points
   keep their tier verdicts; all three subspace points pass Tier-1 and fail
   Tier-2 (LRT 0.861 / 0.858 / 0.672).
2. **Magnitude within +-0.02, direction <= stored+0.01 — HELD.** Largest
   |delta T1 max| = 0.008; largest |delta LRT| = 0.013; largest positive T1
   delta +0.008; largest positive LRT delta +0.010. No memorization-driven
   inflation: fresh readings are statistically indistinguishable from
   in-partition readings.
3. **Risk cases — consistent (vacuous).** Neither named near-bar point
   (hard subspace, middle fullrank-T2 LRT) flipped; hard subspace read 0.543
   (bar 0.55), middle LRT read 0.543.
4. **Utility ratios within +-10pp — PARTIALLY FALSIFIED.** Held on the middle
   cell (max 3.4pp) and hard full-rank (max 2.6pp; absolute lifts within 0.005
   of stored, inside the +-0.02 carve-out). Falsified on the easy cell
   (fullrank-T1 -23.8pp, fullrank-T2 +14.2pp, subspace -38.2pp) and hard
   subspace (+11.5pp). Cause is the denominator, not the mechanism: the easy
   cell's clean lift is ~0.02 and shrinks to 0.0152 out-of-sample on half the
   training data, so percentage swings of tens of pp correspond to absolute
   lift movements of <= 0.012; the prediction carved this out only for the hard
   cell and should have carved it out for every near-zero-lift denominator.
   The in-sample >100% readings (noise-as-regularizer) do not survive
   out-of-sample evaluation on the easy cell (102.2% -> 64.0%).

## Bottom line numbers

59-of-67-style audit conclusions are unaffected: every surviving operating
point still passes on rows the encoder never saw (max attacker-side delta
0.013 AUC), and the subspace channel's Tier-2 failure reproduces (0.66-0.86).
The only quantity that moves is the utility percentage on near-zero-lift cells,
which is a denominator artifact of the convention, not attacker-side drift.

# ============================================================
# FILE: analysis/yus_extras/fare_prediction.md
# ============================================================

# Registered prediction — FARE under the two-tier gauntlet (committed before any FARE run)

Addresses blind-reviewer R1 W4: FARE (Jovanovic et al. 2023) is the closest
certified competitor (practical certificates for restricted downstream families)
and is cited but not evaluated in Table 2. This runs FARE under the identical
two-tier protocol used for the other published baselines.

## Pinned protocol

- Code: official eth-sri/fare at commit 89cb1b66ed268c16659cbf7428c43e60da2df641,
  patched fair-Gini sklearn (v1.2.dev0 @ fd60379f) built in
  /Users/nathansamson/fare/.venv (py3.9.12, arm64, no OpenMP). No license file in
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

# ============================================================
# FILE: analysis/yus_extras/fare_results.md
# ============================================================

# FARE under the two-tier gauntlet — results and scoring of the registered claims

Run: 2026-07-24. Prediction committed at ab76813 BEFORE any FARE run. Pipeline:
fare_stage0_export.py (cells) -> fare_stage_a.py (225 tree fits, FARE venv,
one process per config) -> fare_stage_b.py (attacker battery, project venv).
Backing data: fare_gauntlet.json, fare_cells/*_fare_manifest.json. FARE code:
eth-sri/fare @ 89cb1b6, patched fair-Gini sklearn v1.2.dev0 @ fd60379f.

## Reproduction gate — HELD (claim 1)

ACSIncome-CA-2014, k=50 ni=100 alpha=0.9 s=0.3, their entry point + eval:
dp_ub = 0.1571 vs shipped 0.15712 (bit-exact to 4dp); eval acc over 5 reps
0.796/0.792/0.798/0.797/0.797, mean 0.796 vs shipped mean 0.7959. Both within
the 0.01 gate. Nothing is borrowed — all cell numbers below are our runs.

## The Table-2 row (baseline conventions: best certified operating point per
## tier, 3 training seeds, probe seeds [0,1,2], bar 0.55 on 3-seed mean,
## utility = lift_best / cell clean_lift from baseline_gauntlet.json)

| cell | tier | operating point | utility kept | rep T1 / T2 | out T1 / T2 | verdict | FARE dp_ub |
|---|---|---|---|---|---|---|---|
| easy hmda/race/loan_decision | T1 | k=100 a=0.75 | **0.0%** (lift +0.0000) | 0.539 / 0.539 | 0.539 / 0.539 | PASS | 0.000 |
| easy | T2 | same point | 0.0% | 0.539 / 0.539 | 0.539 / 0.539 | PASS | 0.000 |
| middle hmda/race/loan_amount_band | T1 | k=100 a=0.975 | **28.3%** (lift +0.1156) | 0.535 / 0.535 | 0.536 / 0.536 | PASS | unavailable* |
| middle | T2 | same point | 28.3% | 0.535 / 0.535 | 0.536 / 0.536 | PASS | unavailable* |
| hard adult/sex/income | T1 | k=5 a=0.75 | **39.0%** (lift +0.0560) | 0.546 / 0.546 | 0.546 / 0.546 | PASS | 0.185 |
| hard | T2 | same point | 39.0% | 0.546 / 0.546 | 0.546 / 0.546 | PASS | 0.185 |

*their shipped certificate machinery (AlphaBetaAdversary) asserts every leaf
cell appears in every split of every race pair and aborts on sparse minority
races; dp_ub is null at 20/75 easy and 34/75 middle configs, including the
middle certified point. We did not patch their bound; verdicts never use dp_ub.

FARE passes BOTH tiers on BOTH surfaces on all three cells — the only baseline
in the table to do so — but the frontier is expensive: 0% utility on easy
(every config in the 25-point grid is majority-degenerate: 0/25 rows with
lift_best > 0.005; the fair-Gini tree finds no split worth keeping on the
near-zero-lift cell), 28% on middle, 39% on hard. Highest-utility FAILING
sweep rows for scale: middle k=25 a=0.75 keeps 81% at rep T1 = 0.606; hard
k=100 a=0.5 keeps 63% at 0.637.

Accounting update: published-method combinations go 36 -> 42; passes go 1 -> 7
(VFAE easy T1 at incidental noise + FARE 6/6 at 0/28/39% utility). Under the
floor partition, FARE's middle and hard passes sit on floor-infeasible cells
and clear the output surface precisely by paying the predicted utility price
(out readings 0.536 and 0.546, below the cells' coupling floors 0.584/0.603).

## Claim scoring

2. **Frontier by coupling — MIXED.** Middle HELD (28.3% <= 50%). Easy
   FALSIFIED in the interesting direction: predicted a pass at >= 40% utility;
   actual passes exist only at 0% — FARE is degenerate on the entire easy grid,
   so it joins the paper's only-noise-or-degenerate pattern rather than beating
   it. Hard FALSIFIED by 4pp (39.0% vs predicted <= 35%).
3. **Tiers collapse — HELD.** Tier-2 max equals Tier-1 max at every certified
   point (delta 0.000 <= 0.02; the informed QDA never beats XGB on a
   finite-support deterministic z). No config passes one tier and fails the
   other.
4. **Certificate coherence — MIXED, with a finding.** Hard cell: Spearman
   rho(dp_ub, rep T1) = 0.652 (n=25, p=4e-4) >= 0.6, HELD; threshold form
   (dp_ub <= 0.10 -> rep T1 <= 0.60) holds 25/25. Easy: all 25 dp_ub = 0.000
   (rho undefined); threshold form holds. Middle: FALSIFIED — the 14
   certificate-available configs all read dp_ub = 0.000, yet three of them
   (k10 a0.5, k25 a0.5, k25 a0.75) leak race pairs at rep T1 = 0.603-0.610.
   Precise scope note: dp_ub bounds downstream-classifier demographic parity,
   not attribute recovery, so 0.000-vs-0.61 is not a contradiction of FARE's
   theorem — it is the paper's verification-relativity point appearing again
   (a certificate scoped to one property says nothing about the adjacent
   property an attacker actually exploits). The multiclass Bonferroni path
   drives the per-pair balanced-accuracy bound to 0 on heavily imbalanced race
   pairs; on the binary cell, where FARE's certificate is native, coherence
   held exactly as predicted.

## Operational caveats (for the repro package)

- eth-sri/fare has NO license file — run-and-cite; do not redistribute code.
- The patched sklearn build corrupts memory across sequential in-process fits
  (segfault/abort on the 2nd+ fit; clean in a fresh process). Stage A runs one
  process per config; nothing in their code was modified.
- Compiled-in printf debugging in their Cython splitter floods fd 1; routed to
  /dev/null at the OS level in run_one.

# ============================================================
# FILE: analysis/yus_extras/fresh_partition_generalization.json
# ============================================================

```json
{
  "experiment": "fresh_partition_generalization",
  "bar": 0.55,
  "split_seed": 20260724,
  "split": "50/50 stratified (attr,task)",
  "train_seeds": [
    0,
    1,
    2
  ],
  "probe_seeds": [
    0,
    1,
    2
  ],
  "steps": 1500,
  "smoke": false,
  "prediction_commit": "592c551",
  "cells": [
    {
      "cell": "hmda/race/loan_decision",
      "n_rep": 31873,
      "n_eval": 31874,
      "task_maj_eval": 0.8929535044236682,
      "clean_lift_eval": 0.015237079333207873,
      "clean_lift_eval_std": 0.0014396163955926987,
      "clean_lift_rep_insample": 0.027348121189303336,
      "fullrank_t1": {
        "xgb_mean": 0.5294754524768851,
        "mlp_mean": 0.5237992894919353,
        "lora_mean": 0.5325028564859177,
        "lrt_mean": 0.5909175493232597,
        "tier1_max_mean": 0.5325028564859177,
        "tier2_max_mean": 0.5909175493232597,
        "xgb_std": 0.003793430095072782,
        "mlp_std": 0.011283748787999523,
        "lora_std": 0.006543607898532583,
        "lrt_std": 0.011987431682986513,
        "tier1_max_std": 0.006543607898532583,
        "tier2_max_std": 0.011987431682986513,
        "per_seed": {
          "xgb": [
            0.5348291382336519,
            0.5270964425091246,
            0.5265007766878788
          ],
          "mlp": [
            0.5382128555940426,
            0.5106619656366077,
            0.5225230472451557
          ],
          "lora": [
            0.5416069344538825,
            0.5293878386428789,
            0.5265137963609915
          ],
          "lrt": [
            0.6031877778893863,
            0.5949130271123638,
            0.574651842968029
          ],
          "tier1_max": [
            0.5416069344538825,
            0.5293878386428789,
            0.5265137963609915
          ],
          "tier2_max": [
            0.6031877778893863,
            0.5949130271123638,
            0.574651842968029
          ]
        },
        "tier1_pass": true,
        "tier2_pass": false,
        "sigma": 8.0,
        "lift_eval_mean": 0.010792495450837714,
        "lift_eval_std": 0.0022660003639350937,
        "lift_rep_insample_mean": 0.025110072265972172,
        "utility_kept_eval_pct": 70.83047357584087,
        "stored_inpartition": {
          "tier1_max": 0.5351162084403658,
          "lrt": 0.5833593117214804,
          "util_pct": 94.57899183328631
        },
        "delta_tier1_max": -0.0026133519544481176,
        "delta_lrt": 0.007558237601779316
      },
      "fullrank_t2": {
        "xgb_mean": 0.5087976919035749,
        "mlp_mean": 0.5067214813260948,
        "lora_mean": 0.5072171602287225,
        "lrt_mean": 0.5380351289088533,
        "tier1_max_mean": 0.5100466187530678,
        "tier2_max_mean": 0.5380351289088533,
        "xgb_std": 0.0042542805362451065,
        "mlp_std": 0.008067034963343826,
        "lora_std": 0.006441345424589497,
        "lrt_std": 0.011759032560306035,
        "tier1_max_std": 0.005506367515685198,
        "tier2_max_std": 0.011759032560306035,
        "per_seed": {
          "xgb": [
            0.514654325370326,
            0.5046763799276649,
            0.507062370412734
          ],
          "mlp": [
            0.5177677734424327,
            0.4987284865863027,
            0.503668183949549
          ],
          "lora": [
            0.5158850029122147,
            0.5053097124040365,
            0.5004567653699165
          ],
          "lrt": [
            0.5531627013467562,
            0.5364529548666908,
            0.5244897305131131
          ],
          "tier1_max": [
            0.5177677734424327,
            0.5053097124040365,
            0.507062370412734
          ],
          "tier2_max": [
            0.5531627013467562,
            0.5364529548666908,
            0.5244897305131131
          ]
        },
        "tier1_pass": true,
        "tier2_pass": true,
        "sigma": 20.0,
        "lift_eval_mean": 0.010614712095542878,
        "lift_eval_std": 0.0010138177656520315,
        "lift_rep_insample_mean": 0.012131480145159412,
        "utility_kept_eval_pct": 69.66369251887431,
        "stored_inpartition": {
          "tier1_max": 0.5087894478825772,
          "lrt": 0.5377418140931699,
          "util_pct": 55.47733032948471
        },
        "delta_tier1_max": 0.0012571708704905982,
        "delta_lrt": 0.0002933148156833987
      },
      "subspace": {
        "xgb_mean": 0.5103112697615654,
        "mlp_mean": 0.5101351168481464,
        "lora_mean": 0.5174599914901922,
        "lrt_mean": 0.8609905387290399,
        "tier1_max_mean": 0.5174599914901922,
        "tier2_max_mean": 0.8609905387290399,
        "xgb_std": 0.0030914619522051058,
        "mlp_std": 0.004372517451280862,
        "lora_std": 0.0049573398215500344,
        "lrt_std": 0.011796177200563302,
        "tier1_max_std": 0.0049573398215500344,
        "tier2_max_std": 0.011796177200563302,
        "per_seed": {
          "xgb": [
            0.5107077848467468,
            0.506342364085919,
            0.5138836603520301
          ],
          "mlp": [
            0.5045750045325774,
            0.5105716230125817,
            0.5152587229992801
          ],
          "lora": [
            0.5135314899568016,
            0.5143955263237588,
            0.5244529581900158
          ],
          "lrt": [
            0.853218210497109,
            0.852093202533589,
            0.8776602031564217
          ],
          "tier1_max": [
            0.5135314899568016,
            0.5143955263237588,
            0.5244529581900158
          ],
          "tier2_max": [
            0.853218210497109,
            0.852093202533589,
            0.8776602031564217
          ]
        },
        "tier1_pass": true,
        "tier2_pass": false,
        "rank": 32,
        "sigma": 16.0,
        "lam": 10.0,
        "lift_eval_mean": 0.009757168852356187,
        "lift_eval_std": 0.0036673499896348336,
        "lift_rep_insample_mean": 0.029345632144239058,
        "utility_kept_eval_pct": 64.03568977350729,
        "stored_inpartition": {
          "tier1_max": 0.5091121213015587,
          "util_pct": 102.23880597014919,
          "lrt": 0.8542237934281701
        },
        "delta_tier1_max": 0.008347870188633477,
        "delta_lrt": 0.0067667453008697365
      },
      "minutes": 3.3
    },
    {
      "cell": "hmda/race/loan_amount_band",
      "n_rep": 31873,
      "n_eval": 31874,
      "task_maj_eval": 0.20963794942586433,
      "clean_lift_eval": 0.39662420781828445,
      "clean_lift_eval_std": 0.0014301651904425324,
      "clean_lift_rep_insample": 0.41631893242974716,
      "fullrank_t1": {
        "xgb_mean": 0.5333255859677131,
        "mlp_mean": 0.5239439901499562,
        "lora_mean": 0.5385464656562736,
        "lrt_mean": 0.5848544017405826,
        "tier1_max_mean": 0.5409260086545974,
        "tier2_max_mean": 0.5848544017405826,
        "xgb_std": 0.0019250443558168839,
        "mlp_std": 0.007553990387070939,
        "lora_std": 0.008782903709642381,
        "lrt_std": 0.003890688299629453,
        "tier1_max_std": 0.005708002630246516,
        "tier2_max_std": 0.003890688299629453,
        "per_seed": {
          "xgb": [
            0.5307206798932443,
            0.5353132702908581,
            0.5339428077190366
          ],
          "mlp": [
            0.5333991376072883,
            0.5235227252899175,
            0.5149101075526626
          ],
          "lora": [
            0.5409107416369233,
            0.5479244766078324,
            0.526804178724065
          ],
          "lrt": [
            0.5882538885809379,
            0.5869014969990779,
            0.5794078196417322
          ],
          "tier1_max": [
            0.5409107416369233,
            0.5479244766078324,
            0.5339428077190366
          ],
          "tier2_max": [
            0.5882538885809379,
            0.5869014969990779,
            0.5794078196417322
          ]
        },
        "tier1_pass": true,
        "tier2_pass": false,
        "sigma": 12.0,
        "lift_eval_mean": 0.22682018782288596,
        "lift_eval_std": 0.004827062769172054,
        "lift_rep_insample_mean": 0.23559125278448842,
        "utility_kept_eval_pct": 57.187681274060026,
        "stored_inpartition": {
          "tier1_max": 0.5407797396408048,
          "lrt": 0.5846459792438328,
          "util_pct": 56.59499087437309
        },
        "delta_tier1_max": 0.00014626901379255486,
        "delta_lrt": 0.00020842249674979385
      },
      "fullrank_t2": {
        "xgb_mean": 0.5122340710450289,
        "mlp_mean": 0.5056730703745848,
        "lora_mean": 0.5124254087860748,
        "lrt_mean": 0.5433889338358425,
        "tier1_max_mean": 0.5162414057802303,
        "tier2_max_mean": 0.5433889338358425,
        "xgb_std": 0.0016833005323249116,
        "mlp_std": 0.00570048872820905,
        "lora_std": 0.0066259932475980715,
        "lrt_std": 0.0028225854361764616,
        "tier1_max_std": 0.003929547245463898,
        "tier2_max_std": 0.0028225854361764616,
        "per_seed": {
          "xgb": [
            0.5098567791344201,
            0.5135304824957538,
            0.5133149515049129
          ],
          "mlp": [
            0.513613311448931,
            0.5004955615934706,
            0.5029103380813528
          ],
          "lora": [
            0.5077584699353231,
            0.521795954386847,
            0.5077218020360542
          ],
          "lrt": [
            0.5444239971170969,
            0.5462101094030464,
            0.5395326949873844
          ],
          "tier1_max": [
            0.513613311448931,
            0.521795954386847,
            0.5133149515049129
          ],
          "tier2_max": [
            0.5444239971170969,
            0.5462101094030464,
            0.5395326949873844
          ]
        },
        "tier1_pass": true,
        "tier2_pass": true,
        "sigma": 24.0,
        "lift_eval_mean": 0.08563928802995129,
        "lift_eval_std": 0.002540330745229946,
        "lift_rep_insample_mean": 0.08736757339022579,
        "utility_kept_eval_pct": 21.5920476717819,
        "stored_inpartition": {
          "tier1_max": 0.5121384001460152,
          "lrt": 0.5496316188324943,
          "util_pct": 21.727427493443354
        },
        "delta_tier1_max": 0.004103005634215062,
        "delta_lrt": -0.006242684996651815
      },
      "subspace": {
        "xgb_mean": 0.5251562546962093,
        "mlp_mean": 0.5231481678486064,
        "lora_mean": 0.5336727899729348,
        "lrt_mean": 0.8584424202812239,
        "tier1_max_mean": 0.5337910636439961,
        "tier2_max_mean": 0.8584424202812239,
        "xgb_std": 0.0028490658273063697,
        "mlp_std": 0.009732358361158494,
        "lora_std": 0.0019810783960241044,
        "lrt_std": 0.003060398853920401,
        "tier1_max_std": 0.0021287331797213874,
        "tier2_max_std": 0.003060398853920401,
        "per_seed": {
          "xgb": [
            0.5212535827120331,
            0.5279751318518352,
            0.5262400495247593
          ],
          "mlp": [
            0.5194649191435079,
            0.5364747152804501,
            0.5135048691218612
          ],
          "lora": [
            0.5336306640174068,
            0.5361198942672664,
            0.5312678116341311
          ],
          "lrt": [
            0.8583169990947604,
            0.8547584972037979,
            0.8622517645451132
          ],
          "tier1_max": [
            0.5336306640174068,
            0.5364747152804501,
            0.5312678116341311
          ],
          "tier2_max": [
            0.8583169990947604,
            0.8547584972037979,
            0.8622517645451132
          ]
        },
        "tier1_pass": true,
        "tier2_pass": false,
        "rank": 16,
        "sigma": 16.0,
        "lam": 100.0,
        "lift_eval_mean": 0.3831649620380248,
        "lift_eval_std": 0.0011674439810579638,
        "lift_rep_insample_mean": 0.4263483198945816,
        "utility_kept_eval_pct": 96.60654959658285,
        "stored_inpartition": {
          "tier1_max": 0.5311576317190265,
          "util_pct": 99.97392678026411,
          "lrt": 0.8639938598132509
        },
        "delta_tier1_max": 0.002633431924969565,
        "delta_lrt": -0.00555143953202708
      },
      "minutes": 3.2
    },
    {
      "cell": "adult/sex/income",
      "n_rep": 12072,
      "n_eval": 12073,
      "task_maj_eval": 0.7519257848090781,
      "clean_lift_eval": 0.06866561749358073,
      "clean_lift_eval_std": 0.0052048717902869225,
      "clean_lift_rep_insample": 0.18530483764082176,
      "fullrank_t1": {
        "xgb_mean": 0.5245080591655934,
        "mlp_mean": 0.5067008435501585,
        "lora_mean": 0.5179080179080179,
        "lrt_mean": 0.5648947653057242,
        "tier1_max_mean": 0.5245080591655934,
        "tier2_max_mean": 0.5648947653057242,
        "xgb_std": 0.006232790649451111,
        "mlp_std": 0.005688774016846988,
        "lora_std": 0.004693385647712242,
        "lrt_std": 0.004966664246544262,
        "tier1_max_std": 0.006232790649451111,
        "tier2_max_std": 0.004966664246544262,
        "per_seed": {
          "xgb": [
            0.5241550002508907,
            0.5170571361667252,
            0.5323120410791643
          ],
          "mlp": [
            0.49944051382407545,
            0.5133324970311272,
            0.5073295197952731
          ],
          "lora": [
            0.5168870322979912,
            0.5127387224647498,
            0.5240982989613127
          ],
          "lrt": [
            0.5615113653469818,
            0.5612557913927777,
            0.5719171391774132
          ],
          "tier1_max": [
            0.5241550002508907,
            0.5170571361667252,
            0.5323120410791643
          ],
          "tier2_max": [
            0.5615113653469818,
            0.5612557913927777,
            0.5719171391774132
          ]
        },
        "tier1_pass": true,
        "tier2_pass": false,
        "sigma": 32.0,
        "lift_eval_mean": -0.002070736353847432,
        "lift_eval_std": 0.0014202292884521225,
        "lift_rep_insample_mean": 0.001546277888226218,
        "utility_kept_eval_pct": -3.0156815440289555,
        "stored_inpartition": {
          "tier1_max": 0.5310746483134677,
          "lrt": 0.5781671367423106,
          "util_pct": -0.5368585118051253
        },
        "delta_tier1_max": -0.0065665891478743355,
        "delta_lrt": -0.013272371436586328
      },
      "fullrank_t2": {
        "xgb_mean": 0.5077524099441907,
        "mlp_mean": 0.49799078394968815,
        "lora_mean": 0.5061753112438044,
        "lrt_mean": 0.5290161740846673,
        "tier1_max_mean": 0.5108989133646668,
        "tier2_max_mean": 0.5290161740846673,
        "xgb_std": 0.007242271182445907,
        "mlp_std": 0.007195654862733657,
        "lora_std": 0.003768938687590856,
        "lrt_std": 0.004503882189680981,
        "tier1_max_std": 0.004538045110915141,
        "tier2_max_std": 0.004503882189680981,
        "per_seed": {
          "xgb": [
            0.507022931406493,
            0.4992697409135765,
            0.5169645575125027
          ],
          "mlp": [
            0.4885806278956964,
            0.5060503955709436,
            0.49934132838242434
          ],
          "lora": [
            0.5096817870105541,
            0.5009456069045111,
            0.507898539816348
          ],
          "lrt": [
            0.5241079164366836,
            0.5279547393246024,
            0.5349858664927158
          ],
          "tier1_max": [
            0.5096817870105541,
            0.5060503955709436,
            0.5169645575125027
          ],
          "tier2_max": [
            0.5241079164366836,
            0.5279547393246024,
            0.5349858664927158
          ]
        },
        "tier1_pass": true,
        "tier2_pass": true,
        "sigma": 64.0,
        "lift_eval_mean": -0.006571136696209197,
        "lift_eval_std": 7.809235828559227e-05,
        "lift_rep_insample_mean": -0.007234371548486866,
        "utility_kept_eval_pct": -9.569762766385237,
        "stored_inpartition": {
          "tier1_max": 0.506037438872595,
          "lrt": 0.5338598632379427,
          "util_pct": -6.984933325636464
        },
        "delta_tier1_max": 0.004861474492071749,
        "delta_lrt": -0.004843689153275377
      },
      "subspace": {
        "xgb_mean": 0.519247969179476,
        "mlp_mean": 0.5235239544828586,
        "lora_mean": 0.5433488885543679,
        "lrt_mean": 0.67220836748234,
        "tier1_max_mean": 0.5433488885543679,
        "tier2_max_mean": 0.67220836748234,
        "xgb_std": 0.005258058476026564,
        "mlp_std": 0.011678945036532886,
        "lora_std": 0.006343606977215548,
        "lrt_std": 0.007857975722509524,
        "tier1_max_std": 0.006343606977215548,
        "tier2_max_std": 0.007857975722509524,
        "per_seed": {
          "xgb": [
            0.5119433154364661,
            0.521695184571897,
            0.5241054075300651
          ],
          "mlp": [
            0.5089537859400873,
            0.5240726244835834,
            0.5375454530249051
          ],
          "lora": [
            0.5369383812534497,
            0.541118972351849,
            0.5519893120578052
          ],
          "lrt": [
            0.6807885493501932,
            0.6740344054727615,
            0.6618021476240654
          ],
          "tier1_max": [
            0.5369383812534497,
            0.541118972351849,
            0.5519893120578052
          ],
          "tier2_max": [
            0.6807885493501932,
            0.6740344054727615,
            0.6618021476240654
          ]
        },
        "tier1_pass": true,
        "tier2_pass": false,
        "rank": 32,
        "sigma": 16.0,
        "lam": 100.0,
        "lift_eval_mean": 0.06645683204947679,
        "lift_eval_std": 0.0051772642178530915,
        "lift_rep_insample_mean": 0.16431963772918048,
        "utility_kept_eval_pct": 96.78327301970243,
        "stored_inpartition": {
          "tier1_max": 0.5358373793475304,
          "util_pct": 85.34384666391999,
          "lrt": 0.6617114916174585
        },
        "delta_tier1_max": 0.007511509206837519,
        "delta_lrt": 0.010496875864881572
      },
      "minutes": 1.4
    }
  ],
  "flips": []
}
```

# ============================================================
# FILE: analysis/yus_extras/fare_gauntlet.json
# ============================================================

```json
{
  "experiment": "fare_gauntlet",
  "bar": 0.55,
  "prediction_commit": "ab76813",
  "fare_commit": "89cb1b66ed268c16659cbf7428c43e60da2df641",
  "grid": "max_k {5,10,25,50,100} x alpha {0.5,0.75,0.9,0.975,0.999}, min_ni=100",
  "cells": [
    {
      "cell": "hmda/race/loan_decision",
      "key": "easy",
      "clean_lift": 0.02228183287056653,
      "rows": [
        {
          "xgb": 0.5523335453526641,
          "mlp": 0.5429709605600831,
          "lrt": 0.550091661436451,
          "tier1_max": 0.5523335453526641,
          "tier2_max": 0.5523335453526641,
          "out_xgb": 0.552296090917501,
          "out_mlp": 0.523963711658417,
          "out_max": 0.552296090917501,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k100_a0.5_ts0",
          "max_k": 100,
          "alpha": 0.5,
          "nb_cells": 9,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k100_a0.75_ts0",
          "max_k": 100,
          "alpha": 0.75,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k100_a0.975_ts0",
          "max_k": 100,
          "alpha": 0.975,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k100_a0.999_ts0",
          "max_k": 100,
          "alpha": 0.999,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k100_a0.9_ts0",
          "max_k": 100,
          "alpha": 0.9,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5523335453526641,
          "mlp": 0.5429709605600831,
          "lrt": 0.550091661436451,
          "tier1_max": 0.5523335453526641,
          "tier2_max": 0.5523335453526641,
          "out_xgb": 0.552296090917501,
          "out_mlp": 0.523963711658417,
          "out_max": 0.552296090917501,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k10_a0.5_ts0",
          "max_k": 10,
          "alpha": 0.5,
          "nb_cells": 9,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k10_a0.75_ts0",
          "max_k": 10,
          "alpha": 0.75,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k10_a0.975_ts0",
          "max_k": 10,
          "alpha": 0.975,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k10_a0.999_ts0",
          "max_k": 10,
          "alpha": 0.999,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k10_a0.9_ts0",
          "max_k": 10,
          "alpha": 0.9,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5523335453526641,
          "mlp": 0.5429709605600831,
          "lrt": 0.550091661436451,
          "tier1_max": 0.5523335453526641,
          "tier2_max": 0.5523335453526641,
          "out_xgb": 0.552296090917501,
          "out_mlp": 0.523963711658417,
          "out_max": 0.552296090917501,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k25_a0.5_ts0",
          "max_k": 25,
          "alpha": 0.5,
          "nb_cells": 9,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k25_a0.75_ts0",
          "max_k": 25,
          "alpha": 0.75,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k25_a0.975_ts0",
          "max_k": 25,
          "alpha": 0.975,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k25_a0.999_ts0",
          "max_k": 25,
          "alpha": 0.999,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k25_a0.9_ts0",
          "max_k": 25,
          "alpha": 0.9,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5523335453526641,
          "mlp": 0.5429709605600831,
          "lrt": 0.550091661436451,
          "tier1_max": 0.5523335453526641,
          "tier2_max": 0.5523335453526641,
          "out_xgb": 0.552296090917501,
          "out_mlp": 0.523963711658417,
          "out_max": 0.552296090917501,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k50_a0.5_ts0",
          "max_k": 50,
          "alpha": 0.5,
          "nb_cells": 9,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k50_a0.75_ts0",
          "max_k": 50,
          "alpha": 0.75,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k50_a0.975_ts0",
          "max_k": 50,
          "alpha": 0.975,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k50_a0.999_ts0",
          "max_k": 50,
          "alpha": 0.999,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k50_a0.9_ts0",
          "max_k": 50,
          "alpha": 0.9,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5527659741913045,
          "mlp": 0.536915386350644,
          "lrt": 0.5538378773649152,
          "tier1_max": 0.5527659741913045,
          "tier2_max": 0.5538378773649152,
          "out_xgb": 0.5527659741913045,
          "out_mlp": 0.5244204419686699,
          "out_max": 0.5527659741913045,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k5_a0.5_ts0",
          "max_k": 5,
          "alpha": 0.5,
          "nb_cells": 5,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k5_a0.75_ts0",
          "max_k": 5,
          "alpha": 0.75,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k5_a0.975_ts0",
          "max_k": 5,
          "alpha": 0.975,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k5_a0.999_ts0",
          "max_k": 5,
          "alpha": 0.999,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5495186066294739,
          "mlp": 0.5435919694172409,
          "lrt": 0.5495186066294739,
          "tier1_max": 0.5495186066294739,
          "tier2_max": 0.5495186066294739,
          "lora": 0.5376633706336233,
          "out_xgb": 0.5495186066294739,
          "out_mlp": 0.5125038818217593,
          "out_max": 0.5495186066294739,
          "lift_own": 0.0,
          "lift_lr": -2.969380139228317e-05,
          "lift_best": 0.0,
          "tag": "k5_a0.9_ts0",
          "max_k": 5,
          "alpha": 0.9,
          "nb_cells": 4,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        }
      ],
      "spearman_dpub_tier1": {
        "rho": NaN,
        "p": NaN,
        "n": 25,
        "n_grid": 25
      },
      "tier1": {
        "certified": true,
        "cert": {
          "rep_xgb_mean": 0.5393853765046093,
          "rep_mlp_mean": 0.5322063684253636,
          "rep_lora_mean": 0.5306653559802096,
          "rep_lrt_mean": 0.538585892142356,
          "out_xgb_mean": 0.5393772781978581,
          "out_mlp_mean": 0.5140428344465269,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.5226446390025955,
          "lift_own_mean": 0.0,
          "lift_lr_mean": -2.969380139228317e-05,
          "rep_xgb_std": 0.012535931126589695,
          "rep_mlp_std": 0.009632214194788758,
          "rep_lora_std": 0.009404193232466659,
          "rep_lrt_std": 0.0126099550965866,
          "out_xgb_std": 0.012530496464468379,
          "out_mlp_std": 0.0002861318464152543,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0024229914932378206,
          "lift_own_std": 0.0,
          "lift_lr_std": 0.0,
          "per_seed": {
            "rep_xgb": [
              0.5486875598147415,
              0.5478043419674147,
              0.5216642277316716
            ],
            "rep_mlp": [
              0.5398638413263167,
              0.5381342531635882,
              0.5186210107861859
            ],
            "rep_lora": [
              0.5353958779758328,
              0.5390646152581489,
              0.5175355747066471
            ],
            "rep_lrt": [
              0.5486875598147415,
              0.5462623393977558,
              0.5208077772145708
            ],
            "out_xgb": [
              0.5486875598147415,
              0.5477800470471612,
              0.5216642277316716
            ],
            "out_mlp": [
              0.5143258283008438,
              0.5141518230660844,
              0.5136508519726524
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.5260676791473458,
              0.5210688716098462,
              0.5207973662505947
            ],
            "lift_own": [
              0.0,
              0.0,
              0.0
            ],
            "lift_lr": [
              -2.969380139228317e-05,
              -2.969380139228317e-05,
              -2.969380139228317e-05
            ]
          },
          "rep_tier1_max": 0.5393853765046093,
          "rep_tier2_max": 0.5393853765046093,
          "out_tier1_max": 0.5393772781978581,
          "out_tier2_max": 0.5393772781978581,
          "lift_best": 0.0,
          "tag": "k100_a0.75_ts0",
          "max_k": 100,
          "alpha": 0.75,
          "dp_ub": 0.0
        },
        "tried": [
          "k100_a0.75_ts0"
        ]
      },
      "tier2": {
        "certified": true,
        "cert": {
          "rep_xgb_mean": 0.5393853765046093,
          "rep_mlp_mean": 0.5322063684253636,
          "rep_lora_mean": 0.5306653559802096,
          "rep_lrt_mean": 0.538585892142356,
          "out_xgb_mean": 0.5393772781978581,
          "out_mlp_mean": 0.5140428344465269,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.5226446390025955,
          "lift_own_mean": 0.0,
          "lift_lr_mean": -2.969380139228317e-05,
          "rep_xgb_std": 0.012535931126589695,
          "rep_mlp_std": 0.009632214194788758,
          "rep_lora_std": 0.009404193232466659,
          "rep_lrt_std": 0.0126099550965866,
          "out_xgb_std": 0.012530496464468379,
          "out_mlp_std": 0.0002861318464152543,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0024229914932378206,
          "lift_own_std": 0.0,
          "lift_lr_std": 0.0,
          "per_seed": {
            "rep_xgb": [
              0.5486875598147415,
              0.5478043419674147,
              0.5216642277316716
            ],
            "rep_mlp": [
              0.5398638413263167,
              0.5381342531635882,
              0.5186210107861859
            ],
            "rep_lora": [
              0.5353958779758328,
              0.5390646152581489,
              0.5175355747066471
            ],
            "rep_lrt": [
              0.5486875598147415,
              0.5462623393977558,
              0.5208077772145708
            ],
            "out_xgb": [
              0.5486875598147415,
              0.5477800470471612,
              0.5216642277316716
            ],
            "out_mlp": [
              0.5143258283008438,
              0.5141518230660844,
              0.5136508519726524
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.5260676791473458,
              0.5210688716098462,
              0.5207973662505947
            ],
            "lift_own": [
              0.0,
              0.0,
              0.0
            ],
            "lift_lr": [
              -2.969380139228317e-05,
              -2.969380139228317e-05,
              -2.969380139228317e-05
            ]
          },
          "rep_tier1_max": 0.5393853765046093,
          "rep_tier2_max": 0.5393853765046093,
          "out_tier1_max": 0.5393772781978581,
          "out_tier2_max": 0.5393772781978581,
          "lift_best": 0.0,
          "tag": "k100_a0.75_ts0",
          "max_k": 100,
          "alpha": 0.75,
          "dp_ub": 0.0
        },
        "tried": [
          "k100_a0.75_ts0"
        ]
      },
      "certs": {
        "k100_a0.75_ts0": {
          "rep_xgb_mean": 0.5393853765046093,
          "rep_mlp_mean": 0.5322063684253636,
          "rep_lora_mean": 0.5306653559802096,
          "rep_lrt_mean": 0.538585892142356,
          "out_xgb_mean": 0.5393772781978581,
          "out_mlp_mean": 0.5140428344465269,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.5226446390025955,
          "lift_own_mean": 0.0,
          "lift_lr_mean": -2.969380139228317e-05,
          "rep_xgb_std": 0.012535931126589695,
          "rep_mlp_std": 0.009632214194788758,
          "rep_lora_std": 0.009404193232466659,
          "rep_lrt_std": 0.0126099550965866,
          "out_xgb_std": 0.012530496464468379,
          "out_mlp_std": 0.0002861318464152543,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0024229914932378206,
          "lift_own_std": 0.0,
          "lift_lr_std": 0.0,
          "per_seed": {
            "rep_xgb": [
              0.5486875598147415,
              0.5478043419674147,
              0.5216642277316716
            ],
            "rep_mlp": [
              0.5398638413263167,
              0.5381342531635882,
              0.5186210107861859
            ],
            "rep_lora": [
              0.5353958779758328,
              0.5390646152581489,
              0.5175355747066471
            ],
            "rep_lrt": [
              0.5486875598147415,
              0.5462623393977558,
              0.5208077772145708
            ],
            "out_xgb": [
              0.5486875598147415,
              0.5477800470471612,
              0.5216642277316716
            ],
            "out_mlp": [
              0.5143258283008438,
              0.5141518230660844,
              0.5136508519726524
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.5260676791473458,
              0.5210688716098462,
              0.5207973662505947
            ],
            "lift_own": [
              0.0,
              0.0,
              0.0
            ],
            "lift_lr": [
              -2.969380139228317e-05,
              -2.969380139228317e-05,
              -2.969380139228317e-05
            ]
          },
          "rep_tier1_max": 0.5393853765046093,
          "rep_tier2_max": 0.5393853765046093,
          "out_tier1_max": 0.5393772781978581,
          "out_tier2_max": 0.5393772781978581,
          "lift_best": 0.0,
          "tag": "k100_a0.75_ts0",
          "max_k": 100,
          "alpha": 0.75,
          "dp_ub": 0.0
        }
      },
      "minutes": 5.4
    },
    {
      "cell": "hmda/race/loan_amount_band",
      "key": "middle",
      "clean_lift": 0.40912356659921256,
      "rows": [
        {
          "xgb": 0.616415524549075,
          "mlp": 0.6159639068352927,
          "lrt": 0.5986094920197109,
          "tier1_max": 0.616415524549075,
          "tier2_max": 0.616415524549075,
          "out_xgb": 0.6159780287607306,
          "out_mlp": 0.5896627026087522,
          "out_max": 0.6159780287607306,
          "lift_own": 0.3025083533342746,
          "lift_lr": 0.30544943149215636,
          "lift_best": 0.30544943149215636,
          "tag": "k100_a0.5_ts0",
          "max_k": 100,
          "alpha": 0.5,
          "nb_cells": 100,
          "dp_ub": null,
          "cert_failed_pairs": 3
        },
        {
          "xgb": 0.6053802714111737,
          "mlp": 0.6096290532482652,
          "lrt": 0.5945734228552977,
          "tier1_max": 0.6096290532482652,
          "tier2_max": 0.6096290532482652,
          "out_xgb": 0.6052154502544631,
          "out_mlp": 0.5844776364314977,
          "out_max": 0.6052154502544631,
          "lift_own": 0.315387390779174,
          "lift_lr": 0.3114104028167469,
          "lift_best": 0.315387390779174,
          "tag": "k100_a0.75_ts0",
          "max_k": 100,
          "alpha": 0.75,
          "nb_cells": 100,
          "dp_ub": null,
          "cert_failed_pairs": 4
        },
        {
          "xgb": 0.5357788290729054,
          "mlp": 0.5358595943318589,
          "lrt": 0.5304750446964356,
          "tier1_max": 0.5358595943318589,
          "tier2_max": 0.5358595943318589,
          "lora": 0.5341446639926694,
          "out_xgb": 0.5357765541513596,
          "out_mlp": 0.5293994164277716,
          "out_max": 0.5357765541513596,
          "lift_own": 0.11167584356910912,
          "lift_lr": 0.11105901924392891,
          "lift_best": 0.11167584356910912,
          "tag": "k100_a0.975_ts0",
          "max_k": 100,
          "alpha": 0.975,
          "nb_cells": 11,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.5,
          "mlp": 0.5,
          "lrt": 0.5,
          "tier1_max": 0.5,
          "tier2_max": 0.5,
          "lora": 0.5,
          "out_xgb": 0.5,
          "out_mlp": 0.5,
          "out_max": 0.5,
          "lift_own": 0.0,
          "lift_lr": -3.2885931796899914e-06,
          "lift_best": 0.0,
          "tag": "k100_a0.999_ts0",
          "max_k": 100,
          "alpha": 0.999,
          "nb_cells": 1,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5968025200764073,
          "mlp": 0.6018933177838273,
          "lrt": 0.5779768254417006,
          "tier1_max": 0.6018933177838273,
          "tier2_max": 0.6018933177838273,
          "out_xgb": 0.5966836089764749,
          "out_mlp": 0.579953129074692,
          "out_max": 0.5966836089764749,
          "lift_own": 0.30210049100349823,
          "lift_lr": 0.30083752209892045,
          "lift_best": 0.30210049100349823,
          "tag": "k100_a0.9_ts0",
          "max_k": 100,
          "alpha": 0.9,
          "nb_cells": 100,
          "dp_ub": null,
          "cert_failed_pairs": 3
        },
        {
          "xgb": 0.6028374528797631,
          "mlp": 0.5992926994978522,
          "lrt": 0.5984775746589689,
          "tier1_max": 0.6028374528797631,
          "tier2_max": 0.6028374528797631,
          "out_xgb": 0.6028374528797631,
          "out_mlp": 0.5900791048434382,
          "out_max": 0.6028374528797631,
          "lift_own": 0.3217249439189296,
          "lift_lr": 0.3128849588812509,
          "lift_best": 0.3217249439189296,
          "tag": "k10_a0.5_ts0",
          "max_k": 10,
          "alpha": 0.5,
          "nb_cells": 10,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5987741326930961,
          "mlp": 0.5971421223287694,
          "lrt": 0.5927331524802048,
          "tier1_max": 0.5987741326930961,
          "tier2_max": 0.5987741326930961,
          "out_xgb": 0.5987741326930961,
          "out_mlp": 0.5951002601684705,
          "out_max": 0.5987741326930961,
          "lift_own": 0.3032299559194943,
          "lift_lr": 0.3025316928964357,
          "lift_best": 0.3032299559194943,
          "tag": "k10_a0.75_ts0",
          "max_k": 10,
          "alpha": 0.75,
          "nb_cells": 10,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5357595468576801,
          "mlp": 0.5363256018900476,
          "lrt": 0.5318294882347515,
          "tier1_max": 0.5363256018900476,
          "tier2_max": 0.5363256018900476,
          "lora": 0.5343008862579921,
          "out_xgb": 0.5357572719361343,
          "out_mlp": 0.5295903331479187,
          "out_max": 0.5357572719361343,
          "lift_own": 0.11104836306022245,
          "lift_lr": 0.11055704271133182,
          "lift_best": 0.11104836306022245,
          "tag": "k10_a0.975_ts0",
          "max_k": 10,
          "alpha": 0.975,
          "nb_cells": 10,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.5,
          "mlp": 0.5,
          "lrt": 0.5,
          "tier1_max": 0.5,
          "tier2_max": 0.5,
          "lora": 0.5,
          "out_xgb": 0.5,
          "out_mlp": 0.5,
          "out_max": 0.5,
          "lift_own": 0.0,
          "lift_lr": -3.2885931796899914e-06,
          "lift_best": 0.0,
          "tag": "k10_a0.999_ts0",
          "max_k": 10,
          "alpha": 0.999,
          "nb_cells": 1,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5999234564126982,
          "mlp": 0.5951297040756062,
          "lrt": 0.5946334948863046,
          "tier1_max": 0.5999234564126982,
          "tier2_max": 0.5999234564126982,
          "out_xgb": 0.5999234564126982,
          "out_mlp": 0.5849811151651767,
          "out_max": 0.5999234564126982,
          "lift_own": 0.3040143065556027,
          "lift_lr": 0.31040644975155274,
          "lift_best": 0.31040644975155274,
          "tag": "k10_a0.9_ts0",
          "max_k": 10,
          "alpha": 0.9,
          "nb_cells": 10,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.6098490357314919,
          "mlp": 0.606903489690411,
          "lrt": 0.5983141098021986,
          "tier1_max": 0.6098490357314919,
          "tier2_max": 0.6098490357314919,
          "out_xgb": 0.6098490357314919,
          "out_mlp": 0.5855197908207246,
          "out_max": 0.6098490357314919,
          "lift_own": 0.3058810610695406,
          "lift_lr": 0.3026258134962977,
          "lift_best": 0.3058810610695406,
          "tag": "k25_a0.5_ts0",
          "max_k": 25,
          "alpha": 0.5,
          "nb_cells": 25,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.6057914949924214,
          "mlp": 0.6053751515028629,
          "lrt": 0.6006071328231166,
          "tier1_max": 0.6057914949924214,
          "tier2_max": 0.6057914949924214,
          "out_xgb": 0.6057826604759655,
          "out_mlp": 0.5859579166669086,
          "out_max": 0.6057826604759655,
          "lift_own": 0.3330666541170565,
          "lift_lr": 0.33139534352076905,
          "lift_best": 0.3330666541170565,
          "tag": "k25_a0.75_ts0",
          "max_k": 25,
          "alpha": 0.75,
          "nb_cells": 25,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5357788290729054,
          "mlp": 0.5358595943318589,
          "lrt": 0.5304750446964356,
          "tier1_max": 0.5358595943318589,
          "tier2_max": 0.5358595943318589,
          "lora": 0.5341446639926694,
          "out_xgb": 0.5357765541513596,
          "out_mlp": 0.5293994164277716,
          "out_max": 0.5357765541513596,
          "lift_own": 0.11167584356910912,
          "lift_lr": 0.11105901924392891,
          "lift_best": 0.11167584356910912,
          "tag": "k25_a0.975_ts0",
          "max_k": 25,
          "alpha": 0.975,
          "nb_cells": 11,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.5,
          "mlp": 0.5,
          "lrt": 0.5,
          "tier1_max": 0.5,
          "tier2_max": 0.5,
          "lora": 0.5,
          "out_xgb": 0.5,
          "out_mlp": 0.5,
          "out_max": 0.5,
          "lift_own": 0.0,
          "lift_lr": -3.2885931796899914e-06,
          "lift_best": 0.0,
          "tag": "k25_a0.999_ts0",
          "max_k": 25,
          "alpha": 0.999,
          "nb_cells": 1,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.6012876388154342,
          "mlp": 0.6008805491977433,
          "lrt": 0.5955813831622518,
          "tier1_max": 0.6012876388154342,
          "tier2_max": 0.6012876388154342,
          "out_xgb": 0.6012969289112747,
          "out_mlp": 0.5850970474198717,
          "out_max": 0.6012969289112747,
          "lift_own": 0.33066654117056493,
          "lift_lr": 0.3287599667246342,
          "lift_best": 0.33066654117056493,
          "tag": "k25_a0.9_ts0",
          "max_k": 25,
          "alpha": 0.9,
          "nb_cells": 25,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.615210769151123,
          "mlp": 0.6140587324215863,
          "lrt": 0.6024865824060632,
          "tier1_max": 0.615210769151123,
          "tier2_max": 0.615210769151123,
          "out_xgb": 0.6150796428732659,
          "out_mlp": 0.5908359886817568,
          "out_max": 0.6150796428732659,
          "lift_own": 0.32426623997992066,
          "lift_lr": 0.3204773539367821,
          "lift_best": 0.32426623997992066,
          "tag": "k50_a0.5_ts0",
          "max_k": 50,
          "alpha": 0.5,
          "nb_cells": 50,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.606486416727241,
          "mlp": 0.6067782021918449,
          "lrt": 0.5981033228001815,
          "tier1_max": 0.6067782021918449,
          "tier2_max": 0.6067782021918449,
          "out_xgb": 0.6064436987558225,
          "out_mlp": 0.5747977249785212,
          "out_max": 0.6064436987558225,
          "lift_own": 0.3151834596137858,
          "lift_lr": 0.3132614412806987,
          "lift_best": 0.3151834596137858,
          "tag": "k50_a0.75_ts0",
          "max_k": 50,
          "alpha": 0.75,
          "nb_cells": 50,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.5357788290729054,
          "mlp": 0.5358595943318589,
          "lrt": 0.5304750446964356,
          "tier1_max": 0.5358595943318589,
          "tier2_max": 0.5358595943318589,
          "lora": 0.5341446639926694,
          "out_xgb": 0.5357765541513596,
          "out_mlp": 0.5293994164277716,
          "out_max": 0.5357765541513596,
          "lift_own": 0.11167584356910912,
          "lift_lr": 0.11105901924392891,
          "lift_best": 0.11167584356910912,
          "tag": "k50_a0.975_ts0",
          "max_k": 50,
          "alpha": 0.975,
          "nb_cells": 11,
          "dp_ub": null,
          "cert_failed_pairs": 1
        },
        {
          "xgb": 0.5,
          "mlp": 0.5,
          "lrt": 0.5,
          "tier1_max": 0.5,
          "tier2_max": 0.5,
          "lora": 0.5,
          "out_xgb": 0.5,
          "out_mlp": 0.5,
          "out_max": 0.5,
          "lift_own": 0.0,
          "lift_lr": -3.2885931796899914e-06,
          "lift_best": 0.0,
          "tag": "k50_a0.999_ts0",
          "max_k": 50,
          "alpha": 0.999,
          "nb_cells": 1,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.6002155355375021,
          "mlp": 0.6021400708550224,
          "lrt": 0.5919665468898687,
          "tier1_max": 0.6021400708550224,
          "tier2_max": 0.6021400708550224,
          "out_xgb": 0.6000988400375704,
          "out_mlp": 0.5784580885692507,
          "out_max": 0.6000988400375704,
          "lift_own": 0.31033617268263614,
          "lift_lr": 0.3088377730871868,
          "lift_best": 0.31033617268263614,
          "tag": "k50_a0.9_ts0",
          "max_k": 50,
          "alpha": 0.9,
          "nb_cells": 50,
          "dp_ub": null,
          "cert_failed_pairs": 3
        },
        {
          "xgb": 0.5921494536601338,
          "mlp": 0.5913445058505988,
          "lrt": 0.5921353014650435,
          "tier1_max": 0.5921494536601338,
          "tier2_max": 0.5921494536601338,
          "out_xgb": 0.5921494536601338,
          "out_mlp": 0.589859547856898,
          "out_max": 0.5921494536601338,
          "lift_own": 0.2645300955339075,
          "lift_lr": 0.26482070588507844,
          "lift_best": 0.26482070588507844,
          "tag": "k5_a0.5_ts0",
          "max_k": 5,
          "alpha": 0.5,
          "nb_cells": 5,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5944655818144319,
          "mlp": 0.5938163341577789,
          "lrt": 0.5906775385886762,
          "tier1_max": 0.5944655818144319,
          "tier2_max": 0.5944655818144319,
          "out_xgb": 0.5944655818144319,
          "out_mlp": 0.5941515297392486,
          "out_max": 0.5944655818144319,
          "lift_own": 0.28239760302445605,
          "lift_lr": 0.28242125805926427,
          "lift_best": 0.28242125805926427,
          "tag": "k5_a0.75_ts0",
          "max_k": 5,
          "alpha": 0.75,
          "nb_cells": 5,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5351375648750452,
          "mlp": 0.5340506055334484,
          "lrt": 0.5332689292269355,
          "tier1_max": 0.5351375648750452,
          "tier2_max": 0.5351375648750452,
          "lora": 0.5340598317645242,
          "out_xgb": 0.5351375648750452,
          "out_mlp": 0.5236145802838834,
          "out_max": 0.5351375648750452,
          "lift_own": 0.1092914176353397,
          "lift_lr": 0.10895699251367857,
          "lift_best": 0.1092914176353397,
          "tag": "k5_a0.975_ts0",
          "max_k": 5,
          "alpha": 0.975,
          "nb_cells": 5,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5,
          "mlp": 0.5,
          "lrt": 0.5,
          "tier1_max": 0.5,
          "tier2_max": 0.5,
          "lora": 0.5,
          "out_xgb": 0.5,
          "out_mlp": 0.5,
          "out_max": 0.5,
          "lift_own": 0.0,
          "lift_lr": -3.2885931796899914e-06,
          "lift_best": 0.0,
          "tag": "k5_a0.999_ts0",
          "max_k": 5,
          "alpha": 0.999,
          "nb_cells": 1,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5757189048668987,
          "mlp": 0.5745566774275239,
          "lrt": 0.5728524988973638,
          "tier1_max": 0.5757189048668987,
          "tier2_max": 0.5757189048668987,
          "out_xgb": 0.5757189048668987,
          "out_mlp": 0.5657213870653411,
          "out_max": 0.5757189048668987,
          "lift_own": 0.2470390763486909,
          "lift_lr": 0.24847509504238535,
          "lift_best": 0.24847509504238535,
          "tag": "k5_a0.9_ts0",
          "max_k": 5,
          "alpha": 0.9,
          "nb_cells": 5,
          "dp_ub": 0.0,
          "cert_failed_pairs": 0
        }
      ],
      "spearman_dpub_tier1": {
        "rho": NaN,
        "p": NaN,
        "n": 14,
        "n_grid": 25
      },
      "tier1": {
        "certified": true,
        "cert": {
          "rep_xgb_mean": 0.5354299036927069,
          "rep_mlp_mean": 0.5349432913054996,
          "rep_lora_mean": 0.5344743243891696,
          "rep_lrt_mean": 0.5300035506768004,
          "out_xgb_mean": 0.5355067206811625,
          "out_mlp_mean": 0.5292826671260498,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.531658841023327,
          "lift_own_mean": 0.11558190973692879,
          "lift_lr_mean": 0.11507483150470572,
          "rep_xgb_std": 0.002785133165514238,
          "rep_mlp_std": 0.0033667685527323755,
          "rep_lora_std": 0.003697144549473681,
          "rep_lrt_std": 0.0025917889429863845,
          "out_xgb_std": 0.0028886848592173517,
          "out_mlp_std": 0.0015426213478807104,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0015314490176490448,
          "lift_own_std": 0.005691208435827946,
          "lift_lr_std": 0.00581413514797026,
          "per_seed": {
            "rep_xgb": [
              0.532952927996522,
              0.5393205404195113,
              0.5340162426620872
            ],
            "rep_mlp": [
              0.5331610975671331,
              0.539658068972121,
              0.5320107073772448
            ],
            "rep_lora": [
              0.5323816190862534,
              0.5396702258252445,
              0.5313711282560111
            ],
            "rep_lrt": [
              0.5269671299501905,
              0.533299728071785,
              0.5297437940084255
            ],
            "out_xgb": [
              0.5329542892928217,
              0.5395452851638942,
              0.5340205875867711
            ],
            "out_mlp": [
              0.5279302223371526,
              0.5314413505669157,
              0.5284764284740812
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.5311242468349682,
              0.5337437356887021,
              0.5301085405463104
            ],
            "lift_own": [
              0.11167584356910912,
              0.12362934726340066,
              0.11144053837827664
            ],
            "lift_lr": [
              0.11078711528877216,
              0.12329469722598325,
              0.1111426819993618
            ]
          },
          "rep_tier1_max": 0.5354299036927069,
          "rep_tier2_max": 0.5354299036927069,
          "out_tier1_max": 0.5355067206811625,
          "out_tier2_max": 0.5355067206811625,
          "lift_best": 0.11558190973692879,
          "tag": "k100_a0.975_ts0",
          "max_k": 100,
          "alpha": 0.975,
          "dp_ub": null
        },
        "tried": [
          "k100_a0.975_ts0"
        ]
      },
      "tier2": {
        "certified": true,
        "cert": {
          "rep_xgb_mean": 0.5354299036927069,
          "rep_mlp_mean": 0.5349432913054996,
          "rep_lora_mean": 0.5344743243891696,
          "rep_lrt_mean": 0.5300035506768004,
          "out_xgb_mean": 0.5355067206811625,
          "out_mlp_mean": 0.5292826671260498,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.531658841023327,
          "lift_own_mean": 0.11558190973692879,
          "lift_lr_mean": 0.11507483150470572,
          "rep_xgb_std": 0.002785133165514238,
          "rep_mlp_std": 0.0033667685527323755,
          "rep_lora_std": 0.003697144549473681,
          "rep_lrt_std": 0.0025917889429863845,
          "out_xgb_std": 0.0028886848592173517,
          "out_mlp_std": 0.0015426213478807104,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0015314490176490448,
          "lift_own_std": 0.005691208435827946,
          "lift_lr_std": 0.00581413514797026,
          "per_seed": {
            "rep_xgb": [
              0.532952927996522,
              0.5393205404195113,
              0.5340162426620872
            ],
            "rep_mlp": [
              0.5331610975671331,
              0.539658068972121,
              0.5320107073772448
            ],
            "rep_lora": [
              0.5323816190862534,
              0.5396702258252445,
              0.5313711282560111
            ],
            "rep_lrt": [
              0.5269671299501905,
              0.533299728071785,
              0.5297437940084255
            ],
            "out_xgb": [
              0.5329542892928217,
              0.5395452851638942,
              0.5340205875867711
            ],
            "out_mlp": [
              0.5279302223371526,
              0.5314413505669157,
              0.5284764284740812
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.5311242468349682,
              0.5337437356887021,
              0.5301085405463104
            ],
            "lift_own": [
              0.11167584356910912,
              0.12362934726340066,
              0.11144053837827664
            ],
            "lift_lr": [
              0.11078711528877216,
              0.12329469722598325,
              0.1111426819993618
            ]
          },
          "rep_tier1_max": 0.5354299036927069,
          "rep_tier2_max": 0.5354299036927069,
          "out_tier1_max": 0.5355067206811625,
          "out_tier2_max": 0.5355067206811625,
          "lift_best": 0.11558190973692879,
          "tag": "k100_a0.975_ts0",
          "max_k": 100,
          "alpha": 0.975,
          "dp_ub": null
        },
        "tried": [
          "k100_a0.975_ts0"
        ]
      },
      "certs": {
        "k100_a0.975_ts0": {
          "rep_xgb_mean": 0.5354299036927069,
          "rep_mlp_mean": 0.5349432913054996,
          "rep_lora_mean": 0.5344743243891696,
          "rep_lrt_mean": 0.5300035506768004,
          "out_xgb_mean": 0.5355067206811625,
          "out_mlp_mean": 0.5292826671260498,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.531658841023327,
          "lift_own_mean": 0.11558190973692879,
          "lift_lr_mean": 0.11507483150470572,
          "rep_xgb_std": 0.002785133165514238,
          "rep_mlp_std": 0.0033667685527323755,
          "rep_lora_std": 0.003697144549473681,
          "rep_lrt_std": 0.0025917889429863845,
          "out_xgb_std": 0.0028886848592173517,
          "out_mlp_std": 0.0015426213478807104,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0015314490176490448,
          "lift_own_std": 0.005691208435827946,
          "lift_lr_std": 0.00581413514797026,
          "per_seed": {
            "rep_xgb": [
              0.532952927996522,
              0.5393205404195113,
              0.5340162426620872
            ],
            "rep_mlp": [
              0.5331610975671331,
              0.539658068972121,
              0.5320107073772448
            ],
            "rep_lora": [
              0.5323816190862534,
              0.5396702258252445,
              0.5313711282560111
            ],
            "rep_lrt": [
              0.5269671299501905,
              0.533299728071785,
              0.5297437940084255
            ],
            "out_xgb": [
              0.5329542892928217,
              0.5395452851638942,
              0.5340205875867711
            ],
            "out_mlp": [
              0.5279302223371526,
              0.5314413505669157,
              0.5284764284740812
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.5311242468349682,
              0.5337437356887021,
              0.5301085405463104
            ],
            "lift_own": [
              0.11167584356910912,
              0.12362934726340066,
              0.11144053837827664
            ],
            "lift_lr": [
              0.11078711528877216,
              0.12329469722598325,
              0.1111426819993618
            ]
          },
          "rep_tier1_max": 0.5354299036927069,
          "rep_tier2_max": 0.5354299036927069,
          "out_tier1_max": 0.5355067206811625,
          "out_tier2_max": 0.5355067206811625,
          "lift_best": 0.11558190973692879,
          "tag": "k100_a0.975_ts0",
          "max_k": 100,
          "alpha": 0.975,
          "dp_ub": null
        }
      },
      "minutes": 6.1
    },
    {
      "cell": "adult/sex/income",
      "key": "hard",
      "clean_lift": 0.14349140608821695,
      "rows": [
        {
          "xgb": 0.6356907275445585,
          "mlp": 0.6369177730764635,
          "lrt": 0.626261623452842,
          "tier1_max": 0.6369177730764635,
          "tier2_max": 0.6369177730764635,
          "out_xgb": 0.6356907275445585,
          "out_mlp": 0.6067531495402831,
          "out_max": 0.6356907275445585,
          "lift_own": 0.08589770138745079,
          "lift_lr": 0.09076297737775191,
          "lift_best": 0.09076297737775191,
          "tag": "k100_a0.5_ts0",
          "max_k": 100,
          "alpha": 0.5,
          "nb_cells": 85,
          "dp_ub": 0.8074477285507018,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5650363940462217,
          "mlp": 0.5660982060648293,
          "lrt": 0.5667888231608562,
          "tier1_max": 0.5660982060648293,
          "tier2_max": 0.5667888231608562,
          "out_xgb": 0.5650508904159792,
          "out_mlp": 0.5631439588669587,
          "out_max": 0.5650508904159792,
          "lift_own": 0.06605922551252841,
          "lift_lr": 0.06459103767260033,
          "lift_best": 0.06605922551252841,
          "tag": "k100_a0.75_ts0",
          "max_k": 100,
          "alpha": 0.75,
          "nb_cells": 71,
          "dp_ub": 0.7021485979597699,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5134159509267134,
          "mlp": 0.5156441245397559,
          "lrt": 0.5132601306405311,
          "tier1_max": 0.5163557017548264,
          "tier2_max": 0.5163557017548264,
          "lora": 0.5163557017548264,
          "out_xgb": 0.5134608206426298,
          "out_mlp": 0.5217609022741352,
          "out_max": 0.5217609022741352,
          "lift_own": 0.012797680679229595,
          "lift_lr": 0.01365514236035914,
          "lift_best": 0.01365514236035914,
          "tag": "k100_a0.975_ts0",
          "max_k": 100,
          "alpha": 0.975,
          "nb_cells": 39,
          "dp_ub": 0.4629606947548981,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.499660308010832,
          "mlp": 0.5003158451471426,
          "lrt": 0.4978032665655226,
          "tier1_max": 0.5003158451471426,
          "tier2_max": 0.5003158451471426,
          "lora": 0.49661550556811207,
          "out_xgb": 0.499660308010832,
          "out_mlp": 0.49671591332400883,
          "out_max": 0.499660308010832,
          "lift_own": 0.0,
          "lift_lr": -1.0585650242145306e-05,
          "lift_best": 0.0,
          "tag": "k100_a0.999_ts0",
          "max_k": 100,
          "alpha": 0.999,
          "nb_cells": 8,
          "dp_ub": 0.1661399878578278,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5301323675445986,
          "mlp": 0.5388467566788728,
          "lrt": 0.5361379436893223,
          "tier1_max": 0.5388467566788728,
          "tier2_max": 0.5388467566788728,
          "lora": 0.5347495554446607,
          "out_xgb": 0.530124648698364,
          "out_mlp": 0.5284409361416713,
          "out_max": 0.530124648698364,
          "lift_own": 0.055166701180368594,
          "lift_lr": 0.055894665302217705,
          "lift_best": 0.055894665302217705,
          "tag": "k100_a0.9_ts0",
          "max_k": 100,
          "alpha": 0.9,
          "nb_cells": 54,
          "dp_ub": 0.5773930255098385,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.616104406243756,
          "mlp": 0.6158031829760658,
          "lrt": 0.6154264656269108,
          "tier1_max": 0.616104406243756,
          "tier2_max": 0.616104406243756,
          "out_xgb": 0.616104406243756,
          "out_mlp": 0.5916202887425837,
          "out_max": 0.616104406243756,
          "lift_own": 0.08105197763512106,
          "lift_lr": 0.08479975060948952,
          "lift_best": 0.08479975060948952,
          "tag": "k10_a0.5_ts0",
          "max_k": 10,
          "alpha": 0.5,
          "nb_cells": 10,
          "dp_ub": 0.3214128253201012,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5719789314405801,
          "mlp": 0.5715029986776298,
          "lrt": 0.5682330945971591,
          "tier1_max": 0.5719789314405801,
          "tier2_max": 0.5719789314405801,
          "out_xgb": 0.5719789314405801,
          "out_mlp": 0.5587297515209265,
          "out_max": 0.5719789314405801,
          "lift_own": 0.06278732656864772,
          "lift_lr": 0.06450821507859672,
          "lift_best": 0.06450821507859672,
          "tag": "k10_a0.75_ts0",
          "max_k": 10,
          "alpha": 0.75,
          "nb_cells": 10,
          "dp_ub": 0.2762622480070902,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5127566484995567,
          "mlp": 0.5034347924421074,
          "lrt": 0.5135079495330537,
          "tier1_max": 0.5127566484995567,
          "tier2_max": 0.5135079495330537,
          "lora": 0.5064680479928992,
          "out_xgb": 0.5127566484995567,
          "out_mlp": 0.5172662745911145,
          "out_max": 0.5172662745911145,
          "lift_own": 0.006626630772416653,
          "lift_lr": 0.007443447810085879,
          "lift_best": 0.007443447810085879,
          "tag": "k10_a0.975_ts0",
          "max_k": 10,
          "alpha": 0.975,
          "nb_cells": 10,
          "dp_ub": 0.2141362758513079,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.499660308010832,
          "mlp": 0.5003158451471426,
          "lrt": 0.4978032665655226,
          "tier1_max": 0.5003158451471426,
          "tier2_max": 0.5003158451471426,
          "lora": 0.49661550556811207,
          "out_xgb": 0.499660308010832,
          "out_mlp": 0.49671591332400883,
          "out_max": 0.499660308010832,
          "lift_own": 0.0,
          "lift_lr": -1.0585650242145306e-05,
          "lift_best": 0.0,
          "tag": "k10_a0.999_ts0",
          "max_k": 10,
          "alpha": 0.999,
          "nb_cells": 8,
          "dp_ub": 0.1661399878578278,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5398519638250938,
          "mlp": 0.5398728611892898,
          "lrt": 0.5397505519916381,
          "tier1_max": 0.5398728611892898,
          "tier2_max": 0.5398728611892898,
          "lora": 0.5337656221917206,
          "out_xgb": 0.5398519638250938,
          "out_mlp": 0.5346199666847067,
          "out_max": 0.5398519638250938,
          "lift_own": 0.05123213915924618,
          "lift_lr": 0.051836358196039156,
          "lift_best": 0.051836358196039156,
          "tag": "k10_a0.9_ts0",
          "max_k": 10,
          "alpha": 0.9,
          "nb_cells": 10,
          "dp_ub": 0.23574916221951048,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.6316478406308017,
          "mlp": 0.6297237770084312,
          "lrt": 0.6258831803453826,
          "tier1_max": 0.6316478406308017,
          "tier2_max": 0.6316478406308017,
          "out_xgb": 0.6316478406308017,
          "out_mlp": 0.6001695698732553,
          "out_max": 0.6316478406308017,
          "lift_own": 0.0835369641747773,
          "lift_lr": 0.0867046702715733,
          "lift_best": 0.0867046702715733,
          "tag": "k25_a0.5_ts0",
          "max_k": 25,
          "alpha": 0.5,
          "nb_cells": 25,
          "dp_ub": 0.4523575084334692,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.575996465395973,
          "mlp": 0.5722976319332811,
          "lrt": 0.5716912318425125,
          "tier1_max": 0.575996465395973,
          "tier2_max": 0.575996465395973,
          "out_xgb": 0.575996465395973,
          "out_mlp": 0.5695987969644727,
          "out_max": 0.575996465395973,
          "lift_own": 0.06597639262787325,
          "lift_lr": 0.068400876996768,
          "lift_best": 0.068400876996768,
          "tag": "k25_a0.75_ts0",
          "max_k": 25,
          "alpha": 0.75,
          "nb_cells": 25,
          "dp_ub": 0.3911011442729859,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5147842262427719,
          "mlp": 0.5146673139619997,
          "lrt": 0.5121622659620719,
          "tier1_max": 0.5147842262427719,
          "tier2_max": 0.5147842262427719,
          "lora": 0.5129386689345432,
          "out_xgb": 0.5149199649777747,
          "out_mlp": 0.5196601523788104,
          "out_max": 0.5196601523788104,
          "lift_own": 0.008738869331124444,
          "lift_lr": 0.00909989969015873,
          "lift_best": 0.00909989969015873,
          "tag": "k25_a0.975_ts0",
          "max_k": 25,
          "alpha": 0.975,
          "nb_cells": 25,
          "dp_ub": 0.35519330957154427,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.499660308010832,
          "mlp": 0.5003158451471426,
          "lrt": 0.4978032665655226,
          "tier1_max": 0.5003158451471426,
          "tier2_max": 0.5003158451471426,
          "lora": 0.49661550556811207,
          "out_xgb": 0.499660308010832,
          "out_mlp": 0.49671591332400883,
          "out_max": 0.499660308010832,
          "lift_own": 0.0,
          "lift_lr": -1.0585650242145306e-05,
          "lift_best": 0.0,
          "tag": "k25_a0.999_ts0",
          "max_k": 25,
          "alpha": 0.999,
          "nb_cells": 8,
          "dp_ub": 0.1661399878578278,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5384250441041069,
          "mlp": 0.5416301851820242,
          "lrt": 0.53559844026592,
          "tier1_max": 0.5416301851820242,
          "tier2_max": 0.5416301851820242,
          "lora": 0.5405320381067515,
          "out_xgb": 0.5384250441041069,
          "out_mlp": 0.528558350461223,
          "out_max": 0.5384250441041069,
          "lift_own": 0.05334437771795397,
          "lift_lr": 0.053161519700097415,
          "lift_best": 0.05334437771795397,
          "tag": "k25_a0.9_ts0",
          "max_k": 25,
          "alpha": 0.9,
          "nb_cells": 25,
          "dp_ub": 0.37768342829975987,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.6361265285825738,
          "mlp": 0.6337823212566231,
          "lrt": 0.6277838677870834,
          "tier1_max": 0.6361265285825738,
          "tier2_max": 0.6361265285825738,
          "out_xgb": 0.6361265285825738,
          "out_mlp": 0.6019442769587294,
          "out_max": 0.6361265285825738,
          "lift_own": 0.08660178090702009,
          "lift_lr": 0.09018321921972644,
          "lift_best": 0.09018321921972644,
          "tag": "k50_a0.5_ts0",
          "max_k": 50,
          "alpha": 0.5,
          "nb_cells": 50,
          "dp_ub": 0.599943523654193,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5692504134289349,
          "mlp": 0.5681308669506868,
          "lrt": 0.569537516855952,
          "tier1_max": 0.5692504134289349,
          "tier2_max": 0.569537516855952,
          "out_xgb": 0.5692967265063423,
          "out_mlp": 0.5655110403347996,
          "out_max": 0.5692967265063423,
          "lift_own": 0.06605922551252841,
          "lift_lr": 0.06773829624473882,
          "lift_best": 0.06773829624473882,
          "tag": "k50_a0.75_ts0",
          "max_k": 50,
          "alpha": 0.75,
          "nb_cells": 50,
          "dp_ub": 0.5695227387113246,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5134159509267134,
          "mlp": 0.5156441245397559,
          "lrt": 0.5132601306405311,
          "tier1_max": 0.5163557017548264,
          "tier2_max": 0.5163557017548264,
          "lora": 0.5163557017548264,
          "out_xgb": 0.5134608206426298,
          "out_mlp": 0.5217609022741352,
          "out_max": 0.5217609022741352,
          "lift_own": 0.012797680679229595,
          "lift_lr": 0.01365514236035914,
          "lift_best": 0.01365514236035914,
          "tag": "k50_a0.975_ts0",
          "max_k": 50,
          "alpha": 0.975,
          "nb_cells": 39,
          "dp_ub": 0.4629606947548981,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.499660308010832,
          "mlp": 0.5003158451471426,
          "lrt": 0.4978032665655226,
          "tier1_max": 0.5003158451471426,
          "tier2_max": 0.5003158451471426,
          "lora": 0.49661550556811207,
          "out_xgb": 0.499660308010832,
          "out_mlp": 0.49671591332400883,
          "out_max": 0.499660308010832,
          "lift_own": 0.0,
          "lift_lr": -1.0585650242145306e-05,
          "lift_best": 0.0,
          "tag": "k50_a0.999_ts0",
          "max_k": 50,
          "alpha": 0.999,
          "nb_cells": 8,
          "dp_ub": 0.1661399878578278,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.532242405157344,
          "mlp": 0.5386127752302099,
          "lrt": 0.5361572408049087,
          "tier1_max": 0.5386127752302099,
          "tier2_max": 0.5386127752302099,
          "lora": 0.5317907585203512,
          "out_xgb": 0.5322047522488828,
          "out_mlp": 0.5268886008580846,
          "out_max": 0.5322047522488828,
          "lift_own": 0.055166701180368594,
          "lift_lr": 0.055894665302217705,
          "lift_best": 0.055894665302217705,
          "tag": "k50_a0.9_ts0",
          "max_k": 50,
          "alpha": 0.9,
          "nb_cells": 50,
          "dp_ub": 0.5538074430442157,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5918520423941627,
          "mlp": 0.5918520423941627,
          "lrt": 0.5918520423941627,
          "tier1_max": 0.5918520423941627,
          "tier2_max": 0.5918520423941627,
          "out_xgb": 0.5918520423941627,
          "out_mlp": 0.5805469448932113,
          "out_max": 0.5918520423941627,
          "lift_own": 0.06846137916752948,
          "lift_lr": 0.06956039331281899,
          "lift_best": 0.06956039331281899,
          "tag": "k5_a0.5_ts0",
          "max_k": 5,
          "alpha": 0.5,
          "nb_cells": 5,
          "dp_ub": 0.24207201864412853,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5484322144729747,
          "mlp": 0.5474846790315471,
          "lrt": 0.5484322144729747,
          "tier1_max": 0.5484322144729747,
          "tier2_max": 0.5484322144729747,
          "lora": 0.5469791887354547,
          "out_xgb": 0.5484322144729747,
          "out_mlp": 0.5483948125839033,
          "out_max": 0.5484322144729747,
          "lift_own": 0.05491820252640289,
          "lift_lr": 0.05606031049022503,
          "lift_best": 0.05606031049022503,
          "tag": "k5_a0.75_ts0",
          "max_k": 5,
          "alpha": 0.75,
          "nb_cells": 5,
          "dp_ub": 0.18540660092836592,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5156292830183375,
          "mlp": 0.5143613213259045,
          "lrt": 0.5102517573867476,
          "tier1_max": 0.5156292830183375,
          "tier2_max": 0.5156292830183375,
          "lora": 0.5143920084463005,
          "out_xgb": 0.5156292830183375,
          "out_mlp": 0.513525489512912,
          "out_max": 0.5156292830183375,
          "lift_own": 0.006626630772416653,
          "lift_lr": 0.007443447810085879,
          "lift_best": 0.007443447810085879,
          "tag": "k5_a0.975_ts0",
          "max_k": 5,
          "alpha": 0.975,
          "nb_cells": 5,
          "dp_ub": 0.15054163735949655,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.49599696241436475,
          "mlp": 0.5003001878127074,
          "lrt": 0.49599758996283905,
          "tier1_max": 0.5003001878127074,
          "tier2_max": 0.5003001878127074,
          "lora": 0.4977551649749633,
          "out_xgb": 0.49599696241436475,
          "out_mlp": 0.4962891489840241,
          "out_max": 0.4962891489840241,
          "lift_own": 0.0,
          "lift_lr": -1.0585650242145306e-05,
          "lift_best": 0.0,
          "tag": "k5_a0.999_ts0",
          "max_k": 5,
          "alpha": 0.999,
          "nb_cells": 5,
          "dp_ub": 0.12900657455077846,
          "cert_failed_pairs": 0
        },
        {
          "xgb": 0.5323950877011545,
          "mlp": 0.5311404300363778,
          "lrt": 0.5324132866069107,
          "tier1_max": 0.5323950877011545,
          "tier2_max": 0.5324132866069107,
          "lora": 0.5299096192136266,
          "out_xgb": 0.5323950877011545,
          "out_mlp": 0.53242847327999,
          "out_max": 0.53242847327999,
          "lift_own": 0.03959411886518949,
          "lift_lr": 0.040241195035528976,
          "lift_best": 0.040241195035528976,
          "tag": "k5_a0.9_ts0",
          "max_k": 5,
          "alpha": 0.9,
          "nb_cells": 5,
          "dp_ub": 0.164947682210294,
          "cert_failed_pairs": 0
        }
      ],
      "spearman_dpub_tier1": {
        "rho": 0.6516029354963305,
        "p": 0.00041783041863324445,
        "n": 25,
        "n_grid": 25
      },
      "tier1": {
        "certified": true,
        "cert": {
          "rep_xgb_mean": 0.545511007702279,
          "rep_mlp_mean": 0.5448858439121271,
          "rep_lora_mean": 0.5431039688285236,
          "rep_lrt_mean": 0.545511007702279,
          "out_xgb_mean": 0.545511007702279,
          "out_mlp_mean": 0.545492599613698,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.545492599613698,
          "lift_own_mean": 0.05491820252640289,
          "lift_lr_mean": 0.05603270295889048,
          "rep_xgb_std": 0.0,
          "rep_mlp_std": 0.0,
          "rep_lora_std": 1.1872611404801571e-05,
          "rep_lrt_std": 0.0,
          "out_xgb_std": 0.0,
          "out_mlp_std": 0.0,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0,
          "lift_own_std": 0.0,
          "lift_lr_std": 0.0,
          "per_seed": {
            "rep_xgb": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "rep_mlp": [
              0.5448858439121271,
              0.5448858439121271,
              0.5448858439121271
            ],
            "rep_lora": [
              0.543095573624489,
              0.5431207592365932,
              0.543095573624489
            ],
            "rep_lrt": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "out_xgb": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "out_mlp": [
              0.545492599613698,
              0.545492599613698,
              0.545492599613698
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.545492599613698,
              0.545492599613698,
              0.545492599613698
            ],
            "lift_own": [
              0.05491820252640289,
              0.05491820252640289,
              0.05491820252640289
            ],
            "lift_lr": [
              0.05603270295889048,
              0.05603270295889048,
              0.05603270295889048
            ]
          },
          "rep_tier1_max": 0.545511007702279,
          "rep_tier2_max": 0.545511007702279,
          "out_tier1_max": 0.545511007702279,
          "out_tier2_max": 0.545511007702279,
          "lift_best": 0.05603270295889048,
          "tag": "k5_a0.75_ts0",
          "max_k": 5,
          "alpha": 0.75,
          "dp_ub": 0.18540660092836592
        },
        "tried": [
          "k5_a0.75_ts0"
        ]
      },
      "tier2": {
        "certified": true,
        "cert": {
          "rep_xgb_mean": 0.545511007702279,
          "rep_mlp_mean": 0.5448858439121271,
          "rep_lora_mean": 0.5431039688285236,
          "rep_lrt_mean": 0.545511007702279,
          "out_xgb_mean": 0.545511007702279,
          "out_mlp_mean": 0.545492599613698,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.545492599613698,
          "lift_own_mean": 0.05491820252640289,
          "lift_lr_mean": 0.05603270295889048,
          "rep_xgb_std": 0.0,
          "rep_mlp_std": 0.0,
          "rep_lora_std": 1.1872611404801571e-05,
          "rep_lrt_std": 0.0,
          "out_xgb_std": 0.0,
          "out_mlp_std": 0.0,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0,
          "lift_own_std": 0.0,
          "lift_lr_std": 0.0,
          "per_seed": {
            "rep_xgb": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "rep_mlp": [
              0.5448858439121271,
              0.5448858439121271,
              0.5448858439121271
            ],
            "rep_lora": [
              0.543095573624489,
              0.5431207592365932,
              0.543095573624489
            ],
            "rep_lrt": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "out_xgb": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "out_mlp": [
              0.545492599613698,
              0.545492599613698,
              0.545492599613698
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.545492599613698,
              0.545492599613698,
              0.545492599613698
            ],
            "lift_own": [
              0.05491820252640289,
              0.05491820252640289,
              0.05491820252640289
            ],
            "lift_lr": [
              0.05603270295889048,
              0.05603270295889048,
              0.05603270295889048
            ]
          },
          "rep_tier1_max": 0.545511007702279,
          "rep_tier2_max": 0.545511007702279,
          "out_tier1_max": 0.545511007702279,
          "out_tier2_max": 0.545511007702279,
          "lift_best": 0.05603270295889048,
          "tag": "k5_a0.75_ts0",
          "max_k": 5,
          "alpha": 0.75,
          "dp_ub": 0.18540660092836592
        },
        "tried": [
          "k5_a0.75_ts0"
        ]
      },
      "certs": {
        "k5_a0.75_ts0": {
          "rep_xgb_mean": 0.545511007702279,
          "rep_mlp_mean": 0.5448858439121271,
          "rep_lora_mean": 0.5431039688285236,
          "rep_lrt_mean": 0.545511007702279,
          "out_xgb_mean": 0.545511007702279,
          "out_mlp_mean": 0.545492599613698,
          "out_lora_mean": 0.5,
          "out_lrt_mean": 0.545492599613698,
          "lift_own_mean": 0.05491820252640289,
          "lift_lr_mean": 0.05603270295889048,
          "rep_xgb_std": 0.0,
          "rep_mlp_std": 0.0,
          "rep_lora_std": 1.1872611404801571e-05,
          "rep_lrt_std": 0.0,
          "out_xgb_std": 0.0,
          "out_mlp_std": 0.0,
          "out_lora_std": 0.0,
          "out_lrt_std": 0.0,
          "lift_own_std": 0.0,
          "lift_lr_std": 0.0,
          "per_seed": {
            "rep_xgb": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "rep_mlp": [
              0.5448858439121271,
              0.5448858439121271,
              0.5448858439121271
            ],
            "rep_lora": [
              0.543095573624489,
              0.5431207592365932,
              0.543095573624489
            ],
            "rep_lrt": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "out_xgb": [
              0.545511007702279,
              0.545511007702279,
              0.545511007702279
            ],
            "out_mlp": [
              0.545492599613698,
              0.545492599613698,
              0.545492599613698
            ],
            "out_lora": [
              0.5,
              0.5,
              0.5
            ],
            "out_lrt": [
              0.545492599613698,
              0.545492599613698,
              0.545492599613698
            ],
            "lift_own": [
              0.05491820252640289,
              0.05491820252640289,
              0.05491820252640289
            ],
            "lift_lr": [
              0.05603270295889048,
              0.05603270295889048,
              0.05603270295889048
            ]
          },
          "rep_tier1_max": 0.545511007702279,
          "rep_tier2_max": 0.545511007702279,
          "out_tier1_max": 0.545511007702279,
          "out_tier2_max": 0.545511007702279,
          "lift_best": 0.05603270295889048,
          "tag": "k5_a0.75_ts0",
          "max_k": 5,
          "alpha": 0.75,
          "dp_ub": 0.18540660092836592
        }
      },
      "minutes": 2.0
    }
  ]
}
```

# ============================================================
# FILE: analysis/yus_extras/fare_cells/easy_fare_manifest.json
# ============================================================

```json
{
  "cell": "easy",
  "min_ni": 100,
  "err_budget": 0.05,
  "split": "60/20/20 seed 42+ts",
  "rows": [
    {
      "tag": "k5_a0.5_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.75_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.9_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 4.0,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.975_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.999_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.5_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 9,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.75_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 4.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.9_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.975_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.999_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.5_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 9,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.75_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.9_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.975_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.999_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.5_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 9,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.75_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.9_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.975_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.999_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.5_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 9,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.75_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.9_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.975_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.999_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 4,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8952156862745098,
      "seconds": 3.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.5_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.75_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.9_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.975_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.0,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.999_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.5_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.8935686274509804,
      "seconds": 2.1
    },
    {
      "tag": "k10_a0.75_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.9_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.975_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.999_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 3.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.5_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 15,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.8935686274509804,
      "seconds": 2.1
    },
    {
      "tag": "k25_a0.75_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.0
    },
    {
      "tag": "k25_a0.9_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.2
    },
    {
      "tag": "k25_a0.975_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.1
    },
    {
      "tag": "k25_a0.999_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.0
    },
    {
      "tag": "k50_a0.5_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 15,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.8935686274509804,
      "seconds": 2.1
    },
    {
      "tag": "k50_a0.75_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.0
    },
    {
      "tag": "k50_a0.9_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.975_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.1
    },
    {
      "tag": "k50_a0.999_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.0
    },
    {
      "tag": "k100_a0.5_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 15,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.8935686274509804,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.75_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.9_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.0
    },
    {
      "tag": "k100_a0.975_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.999_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8909019607843137,
      "seconds": 2.1
    },
    {
      "tag": "k5_a0.5_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.75_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 3,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.9_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.975_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.1,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.999_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.5_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k10_a0.75_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 3,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.3,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.9_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.975_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.999_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "tree_test_acc": 0.892,
      "seconds": 2.2,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.5_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 20,
      "dp_ub": null,
      "tree_test_acc": 0.8947450980392156,
      "seconds": 2.2,
      "cert_failed_pairs": 3
    },
    {
      "tag": "k25_a0.75_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 3,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k25_a0.9_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k25_a0.975_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.2
    },
    {
      "tag": "k25_a0.999_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.5_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 20,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8947450980392156,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.75_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 3,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.0
    },
    {
      "tag": "k50_a0.9_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k50_a0.975_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.999_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.5_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 20,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.8947450980392156,
      "seconds": 2.2
    },
    {
      "tag": "k100_a0.75_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 3,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.9_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.2
    },
    {
      "tag": "k100_a0.975_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.999_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 2,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.892,
      "seconds": 2.2
    }
  ]
}
```

# ============================================================
# FILE: analysis/yus_extras/fare_cells/middle_fare_manifest.json
# ============================================================

```json
{
  "cell": "middle",
  "min_ni": 100,
  "err_budget": 0.05,
  "split": "60/20/20 seed 42+ts",
  "rows": [
    {
      "tag": "k5_a0.5_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.48627450980392156,
      "seconds": 1.9
    },
    {
      "tag": "k5_a0.75_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.4992156862745098,
      "seconds": 1.9
    },
    {
      "tag": "k5_a0.9_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.46266666666666667,
      "seconds": 2.1
    },
    {
      "tag": "k5_a0.975_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.31898039215686275,
      "seconds": 2.2
    },
    {
      "tag": "k5_a0.999_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.20854901960784314,
      "seconds": 2.2
    },
    {
      "tag": "k10_a0.5_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5407058823529411,
      "seconds": 1.8
    },
    {
      "tag": "k10_a0.75_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5305882352941177,
      "seconds": 2.0
    },
    {
      "tag": "k10_a0.9_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5397647058823529,
      "seconds": 2.1
    },
    {
      "tag": "k10_a0.975_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.3211764705882353,
      "seconds": 2.2
    },
    {
      "tag": "k10_a0.999_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.20854901960784314,
      "seconds": 2.3
    },
    {
      "tag": "k25_a0.5_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 25,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5549803921568628,
      "seconds": 1.9
    },
    {
      "tag": "k25_a0.75_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 25,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5556078431372549,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.9_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 25,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.5568627450980392,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.975_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.32172549019607843,
      "seconds": 2.2
    },
    {
      "tag": "k25_a0.999_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.20854901960784314,
      "seconds": 2.3
    },
    {
      "tag": "k50_a0.5_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.5694117647058824,
      "seconds": 1.8
    },
    {
      "tag": "k50_a0.75_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.5650196078431372,
      "seconds": 1.9
    },
    {
      "tag": "k50_a0.9_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5664313725490197,
      "seconds": 1.8
    },
    {
      "tag": "k50_a0.975_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.32172549019607843,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.999_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.20854901960784314,
      "seconds": 2.3
    },
    {
      "tag": "k100_a0.5_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5756078431372549,
      "seconds": 2.0
    },
    {
      "tag": "k100_a0.75_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 4,
      "tree_test_acc": 0.5715294117647058,
      "seconds": 1.9
    },
    {
      "tag": "k100_a0.9_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5697254901960784,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.975_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 11,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.32172549019607843,
      "seconds": 2.3
    },
    {
      "tag": "k100_a0.999_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.20854901960784314,
      "seconds": 2.4
    },
    {
      "tag": "k5_a0.5_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.4772549019607843,
      "seconds": 2.0
    },
    {
      "tag": "k5_a0.75_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.4889411764705882,
      "seconds": 2.0
    },
    {
      "tag": "k5_a0.9_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.45623529411764707,
      "seconds": 2.1
    },
    {
      "tag": "k5_a0.975_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.32509803921568625,
      "seconds": 2.2
    },
    {
      "tag": "k5_a0.999_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21576470588235294,
      "seconds": 2.3
    },
    {
      "tag": "k10_a0.5_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5109803921568628,
      "seconds": 1.9
    },
    {
      "tag": "k10_a0.75_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.511921568627451,
      "seconds": 1.9
    },
    {
      "tag": "k10_a0.9_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5217254901960784,
      "seconds": 1.9
    },
    {
      "tag": "k10_a0.975_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.3381176470588235,
      "seconds": 2.2
    },
    {
      "tag": "k10_a0.999_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21576470588235294,
      "seconds": 2.2
    },
    {
      "tag": "k25_a0.5_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 25,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.5512156862745098,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.75_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 25,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.5494117647058824,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.9_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 25,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.544078431372549,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.975_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 12,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.340078431372549,
      "seconds": 2.1
    },
    {
      "tag": "k25_a0.999_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21576470588235294,
      "seconds": 2.3
    },
    {
      "tag": "k50_a0.5_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5587450980392157,
      "seconds": 1.9
    },
    {
      "tag": "k50_a0.75_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5604705882352942,
      "seconds": 1.8
    },
    {
      "tag": "k50_a0.9_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5552156862745098,
      "seconds": 1.9
    },
    {
      "tag": "k50_a0.975_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 12,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.340078431372549,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.999_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21576470588235294,
      "seconds": 2.3
    },
    {
      "tag": "k100_a0.5_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5686274509803921,
      "seconds": 2.0
    },
    {
      "tag": "k100_a0.75_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5694117647058824,
      "seconds": 2.0
    },
    {
      "tag": "k100_a0.9_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5624313725490196,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.975_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 12,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.340078431372549,
      "seconds": 2.2
    },
    {
      "tag": "k100_a0.999_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21576470588235294,
      "seconds": 2.2
    },
    {
      "tag": "k5_a0.5_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.4872156862745098,
      "seconds": 2.0
    },
    {
      "tag": "k5_a0.75_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.4916862745098039,
      "seconds": 1.9
    },
    {
      "tag": "k5_a0.9_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.39615686274509804,
      "seconds": 2.2
    },
    {
      "tag": "k5_a0.975_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.3176470588235294,
      "seconds": 2.3
    },
    {
      "tag": "k5_a0.999_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21129411764705883,
      "seconds": 2.3
    },
    {
      "tag": "k10_a0.5_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.532,
      "seconds": 1.8
    },
    {
      "tag": "k10_a0.75_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5331764705882353,
      "seconds": 1.9
    },
    {
      "tag": "k10_a0.9_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5254901960784314,
      "seconds": 2.0
    },
    {
      "tag": "k10_a0.975_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.319843137254902,
      "seconds": 2.1
    },
    {
      "tag": "k10_a0.999_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21129411764705883,
      "seconds": 2.2
    },
    {
      "tag": "k25_a0.5_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 25,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5511372549019607,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.75_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 25,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.548235294117647,
      "seconds": 1.8
    },
    {
      "tag": "k25_a0.9_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 25,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.5477647058823529,
      "seconds": 1.9
    },
    {
      "tag": "k25_a0.975_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 14,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.319843137254902,
      "seconds": 2.4
    },
    {
      "tag": "k25_a0.999_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21129411764705883,
      "seconds": 2.2
    },
    {
      "tag": "k50_a0.5_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5654901960784313,
      "seconds": 1.8
    },
    {
      "tag": "k50_a0.75_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.5617254901960784,
      "seconds": 2.0
    },
    {
      "tag": "k50_a0.9_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 50,
      "dp_ub": null,
      "cert_failed_pairs": 2,
      "tree_test_acc": 0.5608627450980392,
      "seconds": 1.9
    },
    {
      "tag": "k50_a0.975_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 14,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.319843137254902,
      "seconds": 2.4
    },
    {
      "tag": "k50_a0.999_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21129411764705883,
      "seconds": 2.3
    },
    {
      "tag": "k100_a0.5_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 5,
      "tree_test_acc": 0.5685490196078431,
      "seconds": 1.9
    },
    {
      "tag": "k100_a0.75_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5687058823529412,
      "seconds": 2.0
    },
    {
      "tag": "k100_a0.9_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 100,
      "dp_ub": null,
      "cert_failed_pairs": 3,
      "tree_test_acc": 0.5631372549019608,
      "seconds": 2.1
    },
    {
      "tag": "k100_a0.975_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 14,
      "dp_ub": null,
      "cert_failed_pairs": 1,
      "tree_test_acc": 0.319843137254902,
      "seconds": 2.3
    },
    {
      "tag": "k100_a0.999_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 1,
      "dp_ub": 0.0,
      "cert_failed_pairs": 0,
      "tree_test_acc": 0.21129411764705883,
      "seconds": 2.3
    }
  ]
}
```

# ============================================================
# FILE: analysis/yus_extras/fare_cells/hard_fare_manifest.json
# ============================================================

```json
{
  "cell": "hard",
  "min_ni": 100,
  "err_budget": 0.05,
  "split": "60/20/20 seed 42+ts",
  "rows": [
    {
      "tag": "k5_a0.5_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.24207201864412853,
      "tree_test_acc": 0.8198384758749223,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.75_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.18540660092836592,
      "tree_test_acc": 0.8082418720231932,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.9_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.164947682210294,
      "tree_test_acc": 0.7941602816318079,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.975_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.15054163735949655,
      "tree_test_acc": 0.7622696210395526,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.999_ts0",
      "ts": 0,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 5,
      "dp_ub": 0.12900657455077846,
      "tree_test_acc": 0.7570925657486022,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.5_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": 0.3214128253201012,
      "tree_test_acc": 0.8328846552081176,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.75_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.2762622480070902,
      "tree_test_acc": 0.8127976806792296,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.9_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.23574916221951048,
      "tree_test_acc": 0.8047214744253469,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.975_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": 0.2141362758513079,
      "tree_test_acc": 0.7622696210395526,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.999_ts0",
      "ts": 0,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 8,
      "dp_ub": 0.1661399878578278,
      "tree_test_acc": 0.7570925657486022,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.5_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 25,
      "dp_ub": 0.4523575084334692,
      "tree_test_acc": 0.8339200662663078,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.75_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 25,
      "dp_ub": 0.3911011442729859,
      "tree_test_acc": 0.8152826672188859,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.9_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 25,
      "dp_ub": 0.37768342829975987,
      "tree_test_acc": 0.8034789811555187,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.975_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 25,
      "dp_ub": 0.35519330957154427,
      "tree_test_acc": 0.7583350590184303,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.999_ts0",
      "ts": 0,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 8,
      "dp_ub": 0.1661399878578278,
      "tree_test_acc": 0.7570925657486022,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.5_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 50,
      "dp_ub": 0.599943523654193,
      "tree_test_acc": 0.8347483951128598,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.75_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 50,
      "dp_ub": 0.5695227387113246,
      "tree_test_acc": 0.8136260095257818,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.9_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 50,
      "dp_ub": 0.5538074430442157,
      "tree_test_acc": 0.8080347898115552,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.975_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 39,
      "dp_ub": 0.4629606947548981,
      "tree_test_acc": 0.7633050320977428,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.999_ts0",
      "ts": 0,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 8,
      "dp_ub": 0.1661399878578278,
      "tree_test_acc": 0.7570925657486022,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.5_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 85,
      "dp_ub": 0.8074477285507018,
      "tree_test_acc": 0.8353696417477738,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.75_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 71,
      "dp_ub": 0.7021485979597699,
      "tree_test_acc": 0.8136260095257818,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.9_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 54,
      "dp_ub": 0.5773930255098385,
      "tree_test_acc": 0.8080347898115552,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.975_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 39,
      "dp_ub": 0.4629606947548981,
      "tree_test_acc": 0.7633050320977428,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.999_ts0",
      "ts": 0,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 8,
      "dp_ub": 0.1661399878578278,
      "tree_test_acc": 0.7570925657486022,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.5_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.2460111133045202,
      "tree_test_acc": 0.8308138330917374,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.75_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.17489843540473693,
      "tree_test_acc": 0.8088631186581072,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.9_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.1613999315968686,
      "tree_test_acc": 0.7964381859598261,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.975_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.15330547260398597,
      "tree_test_acc": 0.7672395941188652,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.999_ts1",
      "ts": 1,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 5,
      "dp_ub": 0.11462085117557819,
      "tree_test_acc": 0.7686891696003313,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.5_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": 0.30122849564891463,
      "tree_test_acc": 0.8318492441499276,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.75_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.23733656882670306,
      "tree_test_acc": 0.8246013667425968,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.9_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.22652678966928175,
      "tree_test_acc": 0.8154897494305239,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.975_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": 0.22477660170263447,
      "tree_test_acc": 0.7711741561399875,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.999_ts1",
      "ts": 1,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 10,
      "dp_ub": 0.17017133476719182,
      "tree_test_acc": 0.7686891696003313,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.5_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 25,
      "dp_ub": 0.4306651438852258,
      "tree_test_acc": 0.8318492441499276,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.75_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 25,
      "dp_ub": 0.3929154779706807,
      "tree_test_acc": 0.8217022157796645,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.9_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 25,
      "dp_ub": 0.3547355944496031,
      "tree_test_acc": 0.8088631186581072,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.975_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 25,
      "dp_ub": 0.33543344230732175,
      "tree_test_acc": 0.7771795402774901,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.999_ts1",
      "ts": 1,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 10,
      "dp_ub": 0.17017133476719182,
      "tree_test_acc": 0.7686891696003313,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.5_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 50,
      "dp_ub": 0.618970730914181,
      "tree_test_acc": 0.8345413129012218,
      "seconds": 0.7,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.75_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 50,
      "dp_ub": 0.5613280513264343,
      "tree_test_acc": 0.8198384758749223,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.9_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 50,
      "dp_ub": 0.5438128251447996,
      "tree_test_acc": 0.8107268585628494,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.975_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 43,
      "dp_ub": 0.4827940296119153,
      "tree_test_acc": 0.7809070200869745,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.999_ts1",
      "ts": 1,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 10,
      "dp_ub": 0.17017133476719182,
      "tree_test_acc": 0.7686891696003313,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.5_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 82,
      "dp_ub": 0.7986326676467166,
      "tree_test_acc": 0.8345413129012218,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.75_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 74,
      "dp_ub": 0.7044520111722472,
      "tree_test_acc": 0.8198384758749223,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.9_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 59,
      "dp_ub": 0.6086848692207729,
      "tree_test_acc": 0.8107268585628494,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.975_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 43,
      "dp_ub": 0.4827940296119153,
      "tree_test_acc": 0.7809070200869745,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.999_ts1",
      "ts": 1,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 10,
      "dp_ub": 0.17017133476719182,
      "tree_test_acc": 0.7686891696003313,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.5_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.5,
      "nb_cells": 5,
      "dp_ub": 0.21389217067274457,
      "tree_test_acc": 0.8192172292400083,
      "seconds": 0.7,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.75_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.75,
      "nb_cells": 5,
      "dp_ub": 0.14848417300647654,
      "tree_test_acc": 0.8154897494305239,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.9_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.9,
      "nb_cells": 5,
      "dp_ub": 0.13509328020305267,
      "tree_test_acc": 0.7999585835576724,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.975_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.975,
      "nb_cells": 5,
      "dp_ub": 0.12924173343533085,
      "tree_test_acc": 0.765997100849037,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k5_a0.999_ts2",
      "ts": 2,
      "max_k": 5,
      "alpha": 0.999,
      "nb_cells": 5,
      "dp_ub": 0.1262943334296951,
      "tree_test_acc": 0.765997100849037,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.5_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.5,
      "nb_cells": 10,
      "dp_ub": 0.2667980783658308,
      "tree_test_acc": 0.8335059018430316,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.75_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.75,
      "nb_cells": 10,
      "dp_ub": 0.25178454218568236,
      "tree_test_acc": 0.8202526402981983,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.9_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.9,
      "nb_cells": 10,
      "dp_ub": 0.2138569751074535,
      "tree_test_acc": 0.806792296541727,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.975_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.975,
      "nb_cells": 10,
      "dp_ub": 0.19463136443749773,
      "tree_test_acc": 0.765997100849037,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k10_a0.999_ts2",
      "ts": 2,
      "max_k": 10,
      "alpha": 0.999,
      "nb_cells": 10,
      "dp_ub": 0.17710221720476915,
      "tree_test_acc": 0.7672395941188652,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.5_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.5,
      "nb_cells": 25,
      "dp_ub": 0.4336890322052871,
      "tree_test_acc": 0.8337129840546698,
      "seconds": 0.6,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.75_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.75,
      "nb_cells": 25,
      "dp_ub": 0.3692280249638853,
      "tree_test_acc": 0.8229447090494927,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.9_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.9,
      "nb_cells": 25,
      "dp_ub": 0.3410555558364148,
      "tree_test_acc": 0.8130047628908676,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.975_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.975,
      "nb_cells": 25,
      "dp_ub": 0.305289006814339,
      "tree_test_acc": 0.7842203354731828,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k25_a0.999_ts2",
      "ts": 2,
      "max_k": 25,
      "alpha": 0.999,
      "nb_cells": 13,
      "dp_ub": 0.19485049438015545,
      "tree_test_acc": 0.7672395941188652,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.5_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.5,
      "nb_cells": 50,
      "dp_ub": 0.5578364379419627,
      "tree_test_acc": 0.8355767239594118,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.75_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.75,
      "nb_cells": 50,
      "dp_ub": 0.5344959371064155,
      "tree_test_acc": 0.8233588734727687,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.9_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.9,
      "nb_cells": 50,
      "dp_ub": 0.517142799313348,
      "tree_test_acc": 0.8113481051977635,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.975_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.975,
      "nb_cells": 35,
      "dp_ub": 0.37678866872087147,
      "tree_test_acc": 0.784841582108097,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k50_a0.999_ts2",
      "ts": 2,
      "max_k": 50,
      "alpha": 0.999,
      "nb_cells": 13,
      "dp_ub": 0.19485049438015545,
      "tree_test_acc": 0.7672395941188652,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.5_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.5,
      "nb_cells": 82,
      "dp_ub": 0.747398749232302,
      "tree_test_acc": 0.8355767239594118,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.75_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.75,
      "nb_cells": 76,
      "dp_ub": 0.7052999795742447,
      "tree_test_acc": 0.8233588734727687,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.9_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.9,
      "nb_cells": 52,
      "dp_ub": 0.5260164889130754,
      "tree_test_acc": 0.8113481051977635,
      "seconds": 0.4,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.975_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.975,
      "nb_cells": 35,
      "dp_ub": 0.37678866872087147,
      "tree_test_acc": 0.784841582108097,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    },
    {
      "tag": "k100_a0.999_ts2",
      "ts": 2,
      "max_k": 100,
      "alpha": 0.999,
      "nb_cells": 13,
      "dp_ub": 0.19485049438015545,
      "tree_test_acc": 0.7672395941188652,
      "seconds": 0.5,
      "cert_failed_pairs": 0
    }
  ]
}
```

