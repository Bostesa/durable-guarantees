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
