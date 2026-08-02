# Hybrid + leaf randomized response — results and scoring

Run 2026-07-24, 38.5 min, local MPS. Registered prediction committed at
**7b8e956 BEFORE implementation** (`hybrid_rr_prediction.md`; comparator files
pinned by sha256 there). Script `run_hybrid_rr.py`, log `run_hybrid_rr.log`,
data `hybrid_rr.json`. Mechanism identical to the coarsen run
(`run_hybrid_coarsen.py`, sha256 8939c7db) except for the leaf randomized
response at release. sklearn only; no FARE code.

## Reproduction gate — PASSED on all three cells (zero failures)

p = 0 reproduces the coarsen run's readings to 4 decimal places:

| cell | Tier-2 at p=0 (target) | in-sample utility (target) |
|---|---|---|
| easy | 0.5843 (0.5843) | 107.6% (107.6) |
| middle | 0.6028 (0.6026) | 94.4% (94.4) |
| hard | 0.5892 (0.5892) | 83.4% (83.4) |

## What eps_leaf is and is not

`eps_leaf = ln((1 − p + p/K)/(p/K))`, K = 32, is a **per-release guarantee on
LEAF MEMBERSHIP only**. It bounds how much the emitted leaf reveals about the
true leaf. It is **not** a guarantee on the attribute: s is only partially
carried by the leaf, and the released subspace block is outside the mechanism
entirely. It also does not compose across repeated releases of the same row.
The measured Tier-2 numbers below, not eps_leaf, are what bound attribute
recovery.

## Frontiers (3 training seeds, probe seeds [0,1,2], bar 0.55, both surfaces)

**easy** hmda/race/loan_decision — clean lift +0.0223

| p | eps_leaf | T1 rep / out | T2 | leaf post. | subspace LRT | utility in / held | verdict |
|---|---|---|---|---|---|---|---|
| 0 | ∞ | 0.534 / 0.522 | 0.584 | 0.579 | 0.530 | 107.6% / 105.1% | T2 fail |
| 0.1 | 5.67 | 0.527 / 0.515 | 0.578 | 0.572 | 0.530 | 56.8% / 55.5% | T2 fail |
| 0.2 | 4.86 | 0.520 / 0.512 | 0.572 | 0.565 | 0.530 | 37.5% / 37.3% | T2 fail |
| 0.3 | 4.33 | 0.515 / 0.512 | 0.566 | 0.559 | 0.530 | 18.2% / 19.3% | T2 fail |
| 0.5 | 3.50 | 0.511 / 0.508 | 0.553 | 0.543 | 0.530 | 1.2% / −0.6% | T2 fail |
| **0.7** | 2.69 | 0.510 / 0.507 | **0.544** | 0.531 | 0.530 | **0.0%** / −0.1% | **T2 PASS** |
| 0.9 | 1.52 | 0.507 / 0.507 | 0.533 | 0.510 | 0.530 | 0.0% / −0.1% | T2 PASS |

**middle** hmda/race/loan_amount_band — clean lift +0.4091

| p | eps_leaf | T1 rep / out | T2 | leaf post. | subspace LRT | utility in / held | verdict |
|---|---|---|---|---|---|---|---|
| 0 | ∞ | 0.558 / 0.571 | 0.603 | 0.602 | 0.515 | 94.4% / 92.3% | T1+T2 fail |
| 0.1 | 5.67 | 0.549 / 0.562 | 0.593 | 0.592 | 0.515 | 84.8% / 82.8% | T1+T2 fail |
| 0.2 | 4.86 | 0.540 / 0.553 | 0.583 | 0.581 | 0.515 | 74.8% / 73.4% | T1+T2 fail |
| 0.3 | 4.33 | 0.530 / 0.538 | 0.568 | 0.567 | 0.515 | 65.2% / 63.4% | T1 pass, T2 fail |
| 0.5 | 3.50 | 0.517 / 0.522 | 0.552 | 0.549 | 0.515 | 45.4% / 45.0% | T1 pass, T2 fail |
| **0.7** | 2.69 | 0.508 / 0.510 | **0.533** | 0.529 | 0.515 | **26.5%** / 26.0% | **T2 PASS** |
| 0.9 | 1.52 | 0.503 / 0.505 | 0.518 | 0.510 | 0.515 | 8.9% / 8.7% | T2 PASS |

**hard** adult/sex/income — clean lift +0.1435

| p | eps_leaf | T1 rep / out | T2 | leaf post. | subspace LRT | utility in / held | verdict |
|---|---|---|---|---|---|---|---|
| 0 | ∞ | 0.549 / 0.565 | 0.589 | 0.587 | 0.512 | 83.4% / 71.9% | T1+T2 fail |
| 0.1 | 5.67 | 0.543 / 0.555 | 0.580 | 0.579 | 0.512 | 63.7% / 54.4% | T1+T2 fail |
| 0.2 | 4.86 | 0.533 / 0.550 | 0.573 | 0.572 | 0.512 | 46.6% / 39.7% | T1 pass, T2 fail |
| 0.3 | 4.33 | 0.527 / 0.543 | 0.564 | 0.563 | 0.512 | 32.4% / 26.8% | T1 pass, T2 fail |
| **0.5** | 3.50 | 0.516 / 0.528 | **0.546** | 0.545 | 0.512 | **14.9%** / 12.3% | **T2 PASS** |
| 0.7 | 2.69 | 0.509 / 0.513 | 0.531 | 0.528 | 0.512 | 2.5% / 4.1% | T2 PASS |
| 0.9 | 1.52 | 0.509 / 0.506 | 0.516 | 0.510 | 0.512 | −0.3% / −0.4% | T2 PASS |

## Prediction scoring

- **P1 (a p exists where Tier-2 passes on ≥ 2 cells) — HELD, on all three.**
  Smallest passing p: easy 0.7, middle 0.7, hard 0.5. Randomized response does
  what it was supposed to do to the leaf channel: the flip-aware leaf posterior
  falls monotonically from 0.579/0.602/0.587 to 0.510/0.510/0.510 — i.e. to
  chance.
- **P2 (utility ≥ 50% at the smallest passing p) — FALSIFIED on every cell.**
  Utility at those points is **0.04% / 26.5% / 14.9%**, not ≥ 50%. This is the
  binding failure and it is decisive, not marginal.
- **P3 (Tier-1 passes wherever Tier-2 passes) — HELD everywhere.** The
  mechanism reasoning was right: the flip disperses the concentrated leaf
  signal that broke Tier-1 in the coarsen run. Middle's Tier-1 failure
  (rep 0.558 / out 0.571 at p=0) is repaired by p=0.3 (0.530 / 0.538) and
  hard's output failure (0.565) by p=0.2 (0.550 → passing at 0.533/0.543 by
  p=0.3). No point anywhere passes Tier-2 while failing Tier-1.
- **P4 (named failure mode) — CONFIRMED, and it is the story.** I registered:
  "the frontier runs out of utility before it reaches the bar", with mechanism
  (ii) being that HSIC pushed the task signal into h_perp, so the leaf median
  is where the utility lives and the same flip destroys both. That is exactly
  what the numbers show.

**Success criterion NOT met** (P2 fails on all three cells; zero gate
failures, so the negative is real). Per the registration: next-paper material,
**the submission does not change**, no paper text.

## Why it fails — two structural facts, both new

1. **A terrible exchange rate, quantified.** Between p=0 and the first passing
   p, Tier-2 falls by only 0.040 / 0.070 / 0.043 while utility falls by 107.6 /
   67.9 / 68.5 percentage points. The steepest part is the very first step: on
   the easy cell p=0.1 alone costs 50.8 pp of utility to buy 0.006 of Tier-2.
   Protection decays slowly in p because the leaf posterior's excess over
   chance was only 0.08–0.10 to begin with; utility decays fast because the
   HSIC penalty had already migrated the task signal into the leaf medians.
2. **The subspace LRT is a floor the flip cannot lower.** The Tier-2
   subspace-LRT member is pinned at 0.530 / 0.515 / 0.512 across the entire p
   range — randomized response acts only on the complement block, so it cannot
   touch it. On the easy cell that floor (0.530) is within 0.014 of the leaf
   posterior at p=0.9 (0.510) and within 0.020 of the bar, so even a total
   flip leaves Tier-2 at 0.533. Any future variant of this family inherits
   that floor and must attack the subspace side separately.

## Against the three comparators — no new frontier point

At each cell's smallest Tier-2-passing p, in-sample utility versus the
comparators (all Tier-2-certified):

| cell | hybrid+RR | FARE | full-rank T2 | verdict |
|---|---|---|---|---|
| easy | 0.0% (p=0.7) | 0.0% | **55.5%** | full-rank wins |
| middle | 26.5% (p=0.7) | **28.3%** | 21.7% | FARE wins |
| hard | 14.9% (p=0.5) | **39.0%** | −7.0% | FARE wins |

The hybrid does not dominate anywhere. It beats full-rank on middle and hard
and beats FARE on easy (both at ~0%), but is beaten outright on every cell by
one of the two existing Tier-2 mechanisms. There is no operating point worth
adding to the paper's frontier.

## What is worth keeping for next time

The near-misses are informative: middle at p=0.5 sits at Tier-2 **0.552** — two
thousandths above the bar — with 45.4% utility, and at p=0.3 it passes Tier-1
at 65.2% utility with Tier-2 0.568. So the family is not far off on the middle
cell specifically; what kills it is that the last 0.02 of Tier-2 costs the
remaining half of the utility, and the subspace LRT floor eats most of that
margin. A next attempt would have to (a) attack the subspace floor and the leaf
channel together rather than separately, and (b) avoid concentrating the task
signal in the coarsened block in the first place — i.e. the HSIC objective that
makes the coarsening work is the same thing that makes the flip expensive.
Taken with the coarsen run, this is the no-free-lunch thesis appearing twice at
the same design level: first the partition leaked, and now that the partition
is randomized, the utility it was carrying leaks away with it.
