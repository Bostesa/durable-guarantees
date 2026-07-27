# Registered prediction — knows-Q attacker (committed before implementation)

Registered 2026-07-27 at repo HEAD 819250f, BEFORE implementation. Tests whether
the subspace channel's Tier-1 protection depends on Q being secret — the missing
rung between Tier 1 (knows nothing) and Tier 2 (knows Q, sigma, AND clean class
statistics).

## Attacker

Knows Q. Does NOT know clean class statistics, sigma, or any pre-noise quantity.
Given the released representation h~ = h + sigma*eps*Q^T, it computes

    h~_perp = h~ - (h~ Q) Q^T

Because the channel confines noise to span(Q), this recovers h_perp EXACTLY and
noiselessly. It then trains XGB and MLP(256-256) on h_perp -> s directly.
3 training seeds, probe seeds [0,1,2], standard 75/25 stratified splits, bar 0.55,
at the three stored subspace operating points (easy r=32 s=16 lam=10; middle r=16
s=16 lam=100; hard r=32 s=16 lam=100). Reading = max(XGB, MLP) on the 3-seed mean.

## Comparators (pinned)

- Full-release Tier-1 max (what the paper reports): 0.509 / 0.535 / 0.537.
- Tier-2 Gaussian LRT on the same channels: 0.854 / 0.864 / 0.662.
  The LRT proves substantial attribute signal survives in h_perp; the open
  question is whether a probe WITHOUT clean statistics can find it.

## Registered predictions

- **K1.** The knows-Q reading exceeds the full-release Tier-1 max on all three
  cells (removing the r noised dimensions can only help a probe that would
  otherwise have to ignore them).
- **K2.** The knows-Q reading breaches the 0.55 bar on at least two of three
  cells — i.e. Tier-1 protection DOES depend on Q being secret.
- **K3.** The ordering follows the LRT ordering (middle >= easy > hard), since
  the LRT measures how much signal is there to find.

**Named failure mode.** The signal in h_perp may be diffuse and near-linear
across many dimensions — exactly what a Gaussian LRT with clean class statistics
aggregates well and what an axis-aligned tree finds badly. If so, XGB stays near
chance, the MLP does the work or nothing does, and K2 is FALSIFIED — which would
be a genuinely POSITIVE result for the paper: it would show Tier-1 protection
survives disclosure of Q and that the Tier-1/Tier-2 gap is about clean-statistics
side information, not about Q secrecy. That outcome must be reported as such.

## What each outcome means for the paper

- K2 HELD: the subspace channel is obscurity-dependent; the Tier-1 claim must be
  restated as conditional on Q being secret, and Table 2's "Ours, subspace
  passes Tier 1" needs that qualifier.
- K2 FALSIFIED: the paper gains a defensible answer to the obvious reviewer
  question, and the Tier-1 claim stands with Q public.

No paper text either way; numbers only, pending sign-off.
