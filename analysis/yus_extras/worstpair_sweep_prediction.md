# Registered prediction — worst-pair sigma sweep on the two HMDA/race cells

Registered 2026-07-29 at repo HEAD 3ac75bd, BEFORE implementation and before
any sweep point was computed. Rebuttal-window diagnostic; the submission does
not change on the strength of this.

## Problem being addressed

Under worst-pair AUC (max over label pairs of the binary AUC on that pair's
subset) nine of our reported passing (config, surface) pairs fail, all on the
5-class HMDA/race cells, by 0.014 to 0.105 (maxseed_worstpair.json, sha256
d259629e). Macro-OVR averaging over five race classes hides a leaky pair. The
Adult cell is binary, so worst-pair equals macro there and it is unaffected.

## Pinned comparators (sha256)

- maxseed_worstpair.json **d259629e** — the failing worst-pair readings.
- isolate_vs_fullrank.json **471a4ccd** — subspace operating points
  (easy r=32 sigma=16 lam=10; middle r=16 sigma=16 lam=100).
- two_tier_certification.json **e380d61f** — clean lifts 0.02228 (easy),
  0.40912 (middle); full-rank operating sigmas easy T1=8 / T2=20,
  middle T1=12 / T2=24.
- fresh_partition_generalization.json **a79677ca** — out-of-partition
  convention and eval-half clean lifts 0.01524 (easy), 0.39663 (middle).

## Design (fixed a priori)

Two cells: easy hmda/race/loan_decision, middle hmda/race/loan_amount_band.
Three channels per cell: subspace-confined (train_e2e_surgical, k and lambda
held at their stored values), full-rank Tier-1, full-rank Tier-2
(mi_ceiling.train_channel). Sweep sigma upward from each channel's stored
operating point, DOUBLING, until worst-pair clears 0.55 on BOTH surfaces or
utility goes negative. Scoring is worst-pair AUC, MAX over seeds (3 training
seeds x 3 probe seeds, max over all of them), on the representation and the
output; macro-OVR reported alongside. Tier-1 suite = XGB + MLP + LoRA;
Tier-2 suite additionally includes the Gaussian LRT, whose per-class
posteriors are worst-paired the same way. Reproduction gate at each channel's
stored operating point within 0.01 before sweeping; gate failure invalidates
that cell.

The run will also compute the WORST-PAIR label-coupling predictor (worst-pair
AUC of the attribute from the task label alone) on both cells, because the
paper's footprint law makes that quantity the floor the output surface cannot
go below at positive utility. It is a label-only statistic and no sweep point
depends on it.

## Registered predictions

**P1 — does a passing sigma exist at non-negative utility, per (channel, cell)?**

| channel | easy (loan_decision) | middle (loan_amount_band) |
|---|---|---|
| subspace-confined | YES | NO |
| full-rank Tier-1 | YES | NO |
| full-rank Tier-2 | YES | NO |

Reasoning: the easy cell's clean lift is small (0.0223) and its utility decays
slowly in sigma (94.6% at 8, 55.5% at 20), so utility stays positive well past
the sigma where recovery collapses. The middle cell's label coupling (0.584
macro) already exceeds the bar, so by the paper's own footprint law its output
surface is floored above 0.55 wherever the task is learned at all; I expect the
sweep there to terminate on utility, not on protection.

**P2 — utility at the first passing sigma (percentage of clean lift,
in-sample), as a range.**

| channel | easy | middle |
|---|---|---|
| subspace-confined | 0–25% | n/a (no passing sigma expected) |
| full-rank Tier-1 | 0–20% | n/a |
| full-rank Tier-2 | 0–20% | n/a |

If any middle-cell point does pass against P1, I predict its utility is
below 5%.

**P3 — most likely failure mode.** The worst-pair label-coupling predictor
exceeds 0.55 on BOTH HMDA cells, not just the middle one. If so the output
surface is floored above the bar at every sigma that leaves positive utility,
every sweep terminates on utility rather than protection, and the answer is
0 of 6 rather than 3 of 6 — i.e. the worst-pair failure is not fixable by
adding noise on these cells at all, and the honest response is to report
worst-pair alongside macro-OVR and restrict the claim to the binary cell.
A second, milder failure mode: the sweep clears the representation surface
long before the output surface, so the binding constraint is the output and
the reported sigma is set entirely by the footprint law.

## Scoring

P1 is scored per (channel, cell) against the six-cell table above. P2 is
scored only where P1's YES is correct. P3 is scored by whether the worst-pair
label-coupling predictor exceeds 0.55 on each cell and by which surface binds
at the terminating sigma.

Outputs: analysis/yus_extras/worstpair_sweep.json and a results md scoring
each prediction, with old and new operating points side by side. No paper
edits.
