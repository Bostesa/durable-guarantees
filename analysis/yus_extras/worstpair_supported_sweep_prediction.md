# Registered prediction — supported-pair sigma sweep

Registered 2026-07-29 at repo HEAD 0251ca9, BEFORE implementation and before
any supported-pair sweep point was computed.

## Why this run exists

The all-pairs sweep (worstpair_sweep.json) found 0 of 6 passing, but that
criterion is unusable at HMDA class sizes: its no-signal null is 0.5556, above
the 0.55 bar (worstpair_null.json). Restricted to pairs whose both classes have
>= 3000 rows — classes 0/1/2 of HMDA race — the null falls to 0.5217, leaving
real headroom. The previous sweep scored all-pairs only, so whether any sigma
clears the bar under the *usable* criterion is genuinely unknown.

## Design

Identical to run_worstpair_sweep.py — same channels, same cells, same seeds
(3 training x 3 probe, MAX over all), same doubling schedule, same both-surface
requirement, same gate — with two changes:
  1. scoring is worst-pair restricted to supported pairs (both classes >= 3000
     rows); all-pairs and macro-OVR are recorded alongside;
  2. MAX_SIGMA raised 2048 -> 8192, because middle/subspace was cut off by the
     old ceiling while still holding +73.1% utility.

## Registered predictions

**S1 — which combinations find a passing sigma at non-negative utility**

| channel | easy | middle |
|---|---|---|
| subspace-confined | YES | YES |
| full-rank Tier-1 | NO | NO |
| full-rank Tier-2 | NO | NO |

Reasoning, from the stored per-config restriction deltas: restricting to
supported pairs moved subspace_middle by −0.073 (rep) and −0.085 (out) and
subspace_easy by −0.038 (rep), but moved BOTH easy full-rank configs by
exactly **0.000** — their worst pair was already a supported pair. Applied to
the terminating sweep rows, middle/subspace at sigma=2048 (all-pairs
0.592/0.604, utility +73.1%) should land near 0.52/0.52, and easy/subspace at
sigma=256 (0.557/0.588, utility +49.7%) near 0.52/0.55, marginal on the output.
The four full-rank combinations die on utility within one or two doublings and
gain nothing from the restriction.

**S2 — utility at the first passing sigma (in-sample, % of clean lift)**

| channel | easy | middle |
|---|---|---|
| subspace-confined | 20–60% | >= 60% |
| full-rank | n/a | n/a |

**S3 — most likely failure mode.** The OUTPUT surface binds, not the
representation: on both cells the representation clears the supported bar one
or two doublings before the output does, and on middle the output is floored by
the label (worst-pair label-coupling predictor 0.6858, though its supported-pair
value is not yet measured and may be far lower). Secondary failure mode: the
restriction buys nothing where the leaky pair is already well-supported — as
already observed on both easy full-rank configs — so a combination can fail
supported-pair by exactly the margin it failed all-pairs.

**S4 — supported-pair label-coupling predictor.** Not yet measured. If it
exceeds 0.55 on a cell, no sigma with positive utility can pass on the output
surface there, and that cell's answer is NO regardless of channel.

## Scoring

S1 per combination against the six-cell table. S2 only where S1's YES is
correct. S3 by which surface binds at the terminating sigma. S4 by direct
measurement, reported whether or not it changes S1.

Outputs: worstpair_supported_sweep.json plus a results md. No paper edits.
