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
