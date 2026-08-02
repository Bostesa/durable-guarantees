# Registered bet scored — clipped full-rank DP channel (δ = 1e-5)

2026-07-23. Bet registered at commit `e87332b` (`dp_bet.md`) BEFORE
implementation; script `run_dp_fullrank.py`; data `dp_fullrank.json`. 120
trainings (3 cells × 8 (C, σ) pairs × 5 seeds), 4.0 min on MPS.

## Verdicts

- **PRIMARY: FALSIFIED — 6/18 in-bet points beat their anchor. All six are the
  pre-registered hard-cell edge case.** Every hard-cell ε ≤ 3 point lands at
  kept ∈ [−2.0%, +0.0%], which "beats" the negative −7.0% Tier-2 anchor by
  collapsing to the majority class and doing nothing. The bet file registered
  exactly this failure mode before the run ("a DP channel whose head collapses
  to the majority class has lift ≈ 0, i.e. retained ≈ 0% > −7.0%"), and the
  literal reading was the registered one — so the verdict is falsified, scored
  as registered, no reinterpretation.
- **The substantive claim inside the primary held everywhere it could:** on the
  two cells whose Tier-2 anchors retain positive utility, every ε ≤ 3 point
  sits far below the anchor — easy: all 6 points at −0.1…0.0% vs 55.5%;
  middle: best ε=3 point 8.8% vs 21.7%. No DP point outperformed a *working*
  Tier-2 point anywhere.
- **SECONDARY: HELD — 0/12.** Every ε ≤ 1 point retains ≤ +0.3% on every cell
  (bound was 20%).

## Full grid (kept % = 100·lift/clean, 5 seeds; anchor in header; ε=6 outside the bet)

**easy hmda/loan_decision (clean 0.0223, anchor 55.5%)** — flat zero: even ε=6
retains −0.1%. The easy cell's small task signal does not survive clip+noise at
any audited privacy level.

| σ | ε=0.5 | ε=1 | ε=3 | (ε=6) |
|---|---|---|---|---|
| 8 | +0.0 | +0.0 | +0.0 | −0.1 |
| 32 | −0.0 | −0.0 | −0.0 | −0.1 |

**middle hmda/loan_amount_band (clean 0.4091, anchor 21.7%)** — a real ramp:

| σ | ε=0.5 | ε=1 | ε=3 | (ε=6) |
|---|---|---|---|---|
| 8 | −1.3 | +0.3 | **+8.8** | **+22.6** |
| 32 | −2.0 | −1.0 | +4.7 | +11.2 |

The DP ramp crosses the Tier-2 anchor between ε=3 and ε=6 (σ=8): a formally
private release matching the empirically-certified Tier-2 utility exists on
this cell at ε ≈ 5–6, δ=1e-5 per release — outside the bet's range, reported as
ramp context.

**hard adult/income (clean 0.1435, anchor −7.0%)** — the edge case:

| σ | ε=0.5 | ε=1 | ε=3 | (ε=6) |
|---|---|---|---|---|
| 8 | +0.0 (beats) | +0.0 (beats) | −0.4 (beats) | +4.4 |
| 32 | −0.4 (beats) | −0.4 (beats) | −2.0 (beats) | −2.8 |

## Diagnostics (the empirical norm data the DP audit flagged as missing)

- Median pre-clip ‖h‖ ranges 4.2–11.9 and **grows with C** (e.g. middle σ=32:
  5.15 at C=1.65 → 11.42 at C=19.8): the encoder expands into whatever ball the
  clip allows, so norms are equilibrium quantities of the clip, not fixed
  BN-scale constants. The audit's √64 ≈ 8 naive guess was the right order; the
  learned affine sits below it at tight clips.
- Fraction of rows clipped at ε ≤ 1: 76–100% — consistent with the audit's
  prediction that ε=1-grade clips distort most inputs.
- σ non-invariance at fixed ε: σ=8 beats σ=32 on the middle cell (8.8% vs 4.7%
  at ε=3) — tighter clip + less absolute noise wins over gentler clip + more
  noise, so the (C, σ) ratio alone does not determine utility.

## Plain summary

The bet is **falsified on the letter, held on the substance**: DP at ε ≤ 3
retains essentially nothing (≤ 8.8% everywhere, ≈ 0% at ε ≤ 1 — secondary held
comfortably), and never beats a functioning Tier-2 point; the falsification
comes entirely from the hard cell's negative anchor, where "doing nothing"
(0%) formally exceeds the Tier-2 point's −7.0%. That edge case was registered
in `dp_bet.md` before the run.
