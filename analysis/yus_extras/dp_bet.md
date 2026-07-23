# Registered bet — clipped full-rank DP channel (committed before implementation)

Registered 2026-07-23, before any implementation or measurement of the clipped
channel exists. The bet, verbatim as given:

> bet: clipped full-rank DP channel at delta=1e-5. Primary: at every (C, sigma)
> pair achieving epsilon <= 3, retained utility on each headline cell is at or
> below the existing full-rank Tier-2 point for that cell (55.5/21.7/-7.0).
> Secondary, lower confidence: no epsilon <= 1 point retains more than 20%
> utility on any cell. Falsified if any epsilon <= 3 point beats its Tier-2
> anchor on any cell.

## Operationalization (fixed now, so scoring cannot be litigated post-hoc)

- **Mechanism:** clip-in-the-loop full-rank Gaussian channel, per the DP
  feasibility audit (`dp_feasibility.md`). Same architecture and training
  protocol as the stored full-rank channel (`mi_ceiling.train_channel`:
  `_mlp` encoder → BatchNorm1d(64) → head, Adam 1e-3, 1500 steps, batch 1024),
  with the release changed to `clip_C(h) + sigma * randn(64)` where
  `clip_C(h) = h * min(1, C/||h||_2)` — clip applied in the training loop too
  (head learns through the released channel), noise in ALL 64 directions.
  New script; no existing experiment file modified.
- **Privacy accounting:** per-release Gaussian mechanism, delta = 1e-5,
  sensitivity 2C, classical calibration eps = (2C/sigma)·sqrt(2 ln(1.25/delta))
  = 9.6896·C/sigma. C is set from nominal eps targets; since the classical
  bound is conservative for eps > 1, every nominal-eps point's true eps is <=
  nominal — the "achieving eps <= 3" condition is satisfied conservatively.
- **Grid:** sigma in {8, 32} × nominal eps in {0.5, 1, 3}, plus eps = 6
  reference points OUTSIDE the bet (ramp context only). C = eps·sigma/9.6896:
  sigma=8 → C in {0.413, 0.826, 2.477, (4.954)};
  sigma=32 → C in {1.651, 3.303, 9.908, (19.816)}.
  Both sigmas at equal eps test whether utility at fixed eps is
  scale-invariant, as the audit predicts.
- **Cells / seeds / utility convention:** the three headline cells, 5 training
  seeds [0..4], probe-free (the bet is about utility; privacy is the DP
  guarantee itself). Utility = own-head lift over all rows (argmax accuracy −
  task majority), the e2e convention behind the anchors. Retained % =
  100 · lift_mean / clean_lift with the stored 5-seed clean lifts from
  `two_tier_certification.json`: easy 0.0223, middle 0.4091, hard 0.1435.
- **Anchors (stored Tier-2 retained utility, same file):** easy 55.5%,
  middle 21.7%, hard **−7.0%**.
- **Scoring:** primary holds iff, for every nominal eps <= 3 pair and every
  cell, mean retained % <= anchor. "Beats" = mean retained % strictly greater
  than the anchor; near-ties within seed noise (±100·lift_std/clean_lift) will
  be reported as within-noise but scored literally on the mean, as registered.
  Secondary holds iff every nominal eps <= 1 pair has mean retained <= 20% on
  every cell.
- **Pre-registered edge case, stated before running:** the hard-cell anchor is
  negative (−7.0%). A DP channel whose head collapses to the majority class
  has lift ≈ 0, i.e. retained ≈ 0% > −7.0% — which would falsify the primary
  claim on the hard cell by the literal reading above. That literal reading is
  the registered one.
- Also recorded per point (diagnostics, not part of the bet): fraction of rows
  clipped at eval (||h|| > C) and the median pre-clip ||h|| — this doubles as
  the empirical norm measurement the DP audit flagged as missing, for the
  full-rank channel's BN'd representation.

Falsified if: any nominal eps <= 3 point's mean retained % exceeds its cell's
anchor (primary), or any nominal eps <= 1 point exceeds 20% on any cell
(secondary, lower confidence).
