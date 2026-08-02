# Isolate-then-noise vs plain full-rank noise, matched at Tier-1 — result

2026-07-24. Script `run_isolate_vs_fullrank.py`, data `isolate_vs_fullrank.json`,
registered prediction `isolate_vs_fullrank_prediction.md` (committed before the
run). 3 train seeds, probe seeds [0,1], 11.5 min. Answers blind-reviewer A's
"isolate reduces to a Gaussian mechanism" objection with a measured table.

## Result: all three registered claims HELD

At Tier-1 protection matched within 0.01, per cell:

| cell (coupling) | matched T1 | isolate: util / LRT / σ | full-rank: util / LRT / σ | util gap |
|---|---|---|---|---|
| easy (0.514)  | 0.509 vs 0.508 | **101.6%** / 0.854 / 16 | 1.9% / 0.522 / 32 | +99.7 pp |
| middle (0.584)| 0.535 vs 0.534 | **100.4%** / 0.864 / 16 | 38.6% / 0.571 / 16 | +61.8 pp |
| hard (0.603)  | 0.537 vs 0.532 | 84.3% / 0.662 / 16 | 0.1% / 0.581 / 32 | +84.2 pp |

- **Claim 1 (isolate more utility): HELD, every cell**, gap 62–100 pp.
- **Claim 2 (isolate leaks Tier-2 more): HELD, every cell** — isolate LRT
  0.66–0.86 vs full-rank 0.52–0.58.
- **Claim 2b (isolate LRT ≥ 0.60): HELD** (0.662 / 0.854 / 0.864).

## What it proves (both directions of the tradeoff, quantified)

1. **Isolate buys Tier-1 utility at lower noise.** To match isolate's Tier-1
   protection, plain full-rank noise needs σ=32 (easy, hard) or σ=16 (middle)
   sprayed across all 64 dims, collapsing utility to 1.9% / 38.6% / 0.1% —
   because it perturbs every direction, not just the few carrying the
   attribute. Isolate confines σ=16 to span(Q) and keeps 84–102%. This is a
   genuine, large advantage where the reviewer implied there was none.
2. **Isolate does NOT buy Tier-2 protection.** At those same matched points
   the informed LRT reads isolate's untouched complement at 0.66–0.86, far
   above the 0.55 bar, while full-rank's LRT is 0.52–0.58 (near/at the bar).
   Against the informed attacker only full-rank noise works.
3. **No free lunch (claim 3):** no single channel achieves both isolate-level
   utility and full-rank Tier-2 protection. The honest recommendation is
   isolate-then-noise for the Tier-1 threat model, and the certified full-rank
   channel (Prop. 3) for Tier-2.

The reviewer's reduction ("isolate = a Gaussian mechanism") is half right and
now precisely bounded: against Tier-2 the subspace is inert, but against Tier-1
the subspace is worth 62–100 pp of utility over an equally-protective Gaussian
mechanism. That is a stated, measured scope limit, not a hidden weakness.

## Paper wiring

Replaces the asserted ablation sentence with a 3-row table (or inline). The
finalized "Protection and its cost" sentences:

"The isolating step does not add Tier-2 protection: against the informed
attacker the subspace is inert, and only full-rank noise passes. What it buys
is Tier-1 utility at lower noise. At matched black-box protection (Tier-1 max
within 0.01), isolate-then-noise retains 101.6% / 100.4% / 84.3% of clean
utility on the three cells, whereas plain full-rank noise of equal Tier-1
strength — which must perturb every direction rather than the few carrying the
attribute — retains only 1.9% / 38.6% / 0.1% (a 62–100 pp gap), while reading
the same informed LRT at 0.52–0.58 versus the subspace channel's 0.66–0.86.
The honest recommendation is isolate-then-noise under the Tier-1 threat model
and the certified full-rank channel (Prop. 3) under Tier-2."
