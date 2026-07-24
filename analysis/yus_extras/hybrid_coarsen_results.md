# Isolate-noise-coarsen hybrid — results and scoring of the registered predictions

Run: 2026-07-24, 9.7 min, local MPS. Prediction committed at 364870e BEFORE
implementation (hybrid_coarsen_prediction.md; comparator files pinned by
sha256 there). Script run_hybrid_coarsen.py, log run_hybrid_coarsen.log, data
hybrid_coarsen.json. One fixed configuration per cell, no tuning anywhere.
sklearn coarsener only; no eth-sri/fare code.

## Reproduction gate — PASSED on all three cells (zero failures)

3-seed means vs the stored isolate_vs_fullrank values, tolerance 0.01:

| cell | T1 max (stored) | LRT (stored) |
|---|---|---|
| easy | 0.5094 (0.5094) | 0.8542 (0.8542) |
| middle | 0.5349 (0.5349) | 0.8640 (0.8640) |
| hard | 0.5369 (0.5369) | 0.6617 (0.6617) |

Bit-level reproduction of the base models — every reading below is on the
same channels the paper's numbers come from.

## Headline table (3-train-seed means, probe seeds [0,1,2], bar 0.55)

| cell | T1 rep / out | T2 (a) LRT-sub | (b) leaf post | (c) combined | (d) joint XGB | T2 max | comp-side | util in / held |
|---|---|---|---|---|---|---|---|---|
| easy | 0.534 / 0.522 | 0.530 | 0.579 | **0.584** | 0.538 | 0.584 | 0.579 | 107.6% / 105.1% |
| middle | **0.558** / **0.571** | 0.515 | 0.602 | **0.603** | 0.560 | 0.603 | 0.602 | 94.4% / 92.3% |
| hard | 0.549 / **0.565** | 0.512 | 0.587 | **0.589** | 0.546 | 0.589 | 0.588 | 83.4% / 71.9% |

Bold = above the 0.55 bar. Versus the comparators:

| channel | easy util / T2 | middle util / T2 | hard util / T2 | verdict |
|---|---|---|---|---|
| subspace (Tier-1 point) | 101.6% / 0.854 | 100.4% / 0.864 | 84.3% / 0.662 | T1 only |
| full-rank Tier-2 | 55.5% / ~0.52–0.58 | 21.7% / " | −7.0% / " | T2, costly |
| FARE | 0.0% / pass | 28.3% / pass | 39.0% / pass | both tiers, costly |
| **hybrid coarsen** | 107.6% / 0.584 | 94.4% / 0.603 | 83.4% / 0.589 | **fails both bars on 2 cells** |

## Prediction scoring

- **P1 (Tier-1 still passes everywhere) — FALSIFIED.** Easy holds
  (0.534/0.522). Middle breaches on BOTH surfaces (rep 0.558, out 0.571);
  hard breaches on the output (0.565; rep 0.549 holds). Coarsening is not
  Tier-1-neutral: replacing the HSIC-suppressed complement with 32 leaf
  medians CONCENTRATES the residual attribute signal into a discrete leaf
  identity that XGB/MLP read more easily than the raw h_perp they could not
  exploit — black-box protection got worse, not equal.
- **P2 (complement-side informed reading ≤ 0.60 on ≥ 2 cells) — HELD.**
  0.579 / 0.602 / 0.588 vs the base channel's 0.854 / 0.864 / 0.662. The
  coarsening does destroy most of the fine-grained side-information leak
  (drops of 0.27 / 0.26 / 0.07); middle misses the 0.60 threshold by 0.002.
- **P3 (combined Tier-2 passes on ≥ 2 cells) — FALSIFIED, 0 of 3.** Tier-2
  max 0.584 / 0.603 / 0.589, in every case driven by the per-leaf attribute
  posterior (b), not the subspace LRT (a), which sits at 0.51–0.53
  everywhere. The registered failure mode — the tree's y-pure leaves
  correlate with s — is exactly what happened, but it is NOT confined to the
  high-coupling hard cell: it appears on all three, including easy (coupling
  0.514, leaf posterior 0.579). The leaves leak more than label↔attribute
  coupling alone predicts: the tree partitions h_perp by y-usefulness, and
  the HSIC penalty left coarse-grained attribute structure in h_perp that
  leaf membership exposes even where the label itself barely carries s.
- **P4 (utility ≥ 70/70/50) — HELD, all three cells** (107.6 / 94.4 / 83.4
  in-sample; 105.1 / 92.3 / 71.9 held-out). The task-aware coarsening does
  preserve the migrated task signal — but this is moot for the success
  criterion because P3 passed nowhere.

## Verdict

**Success criterion NOT met** (P1 falsified, P3 falsified 0/3; gates all
passed, so the negatives are real, not artifacts). Per the registration: the
submission does not change, no paper text; this file and hybrid_coarsen.json
are committed as next-paper material.

What the negative result establishes, cleanly: the two halves of the
hypothesis pulled apart. Coarsening h_perp does kill the fine-grained
side-information channel (P2) while preserving task signal (P4) — but the
partition that preserves task signal is itself a new attribute channel. A
y-fit tree is s-correlated at the leaf level on every cell, enough to sit
0.03–0.05 above the bar for an attacker who just reads leaf statistics, and
on two cells the discretization even degrades Tier-1. FARE survives the
analogous attack only by paying 60–100 pp of utility to make its leaves
s-balanced; a task-pure partition at 94–108% utility does not get that for
free. Any next attempt needs an s-aware split criterion or per-leaf noise on
the medians — i.e., the utility/Tier-2 tradeoff reappears inside the
coarsener, which is the paper's no-free-lunch thesis showing up at a second
level of the design.
