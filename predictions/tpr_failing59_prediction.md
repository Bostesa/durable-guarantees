# Registered prediction — worst-case TPR for the 59 AUC-failing audit configs

Registered 2026-07-24, before any implementation or measurement (option (b)
of the dual-verdict plan; completes the "all 67 audited configurations
measured under both criteria" sentence).

## Target

The 59 approved-but-suite-failing audit configurations: 18 existing
(honest_reaudit master rows minus the three adult noise survivors — the
projection configs and low-sigma noise) and 41 expansion
(expansion_reaudit approved-at-rest configs minus the five sigma=8 noise
survivors). Regeneration is deterministic (seed-0 clean reps, seed-0
projections — the same rebuild expansion_reaudit_paperframing.json's
provenance attests). Protocol identical to tpr_extension.json: score capture
for each config's originally scored suite, per-config headline = mean TPR at
FPR {0.001, 0.01} over the original (train seed x probe seed) grid, best
attacker, worst-class OvR for multiclass, reproduction gate at 0.01 against
each config's stored AUCs.

## Registered claims

1. Every one of the 59 reads ABOVE the survivor band's upper edge (5%) at
   the 1% budget: min over the 59 > 0.05.
2. The 59 read consistent with the measured failing-baseline distribution:
   mean TPR@1% in [50%, 75%] (baselines: AUC-failing mean 62.3%).
3. Monotone association with AUC holds loosely but with ordering
   exceptions, as in the baselines: at least one config with suite AUC
   above 0.60 reads below a config with suite AUC below 0.60.
4. The projection configs (which sit at the unprotected AUC level,
   0.68-1.0) read above 20% TPR@1% without exception.

Falsified if: any failing config reads inside the survivor band (<= 5%),
the mean falls outside [50%, 75%], or any projection config reads <= 20%.

Scoring is literal on the per-config means under the stated convention.
Reproduction gate failure (any stored-AUC delta > 0.01) invalidates that
config's reading rather than being averaged over silently.
