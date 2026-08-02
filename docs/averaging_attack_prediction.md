# Registered prediction — averaging attack on the isolate-then-noise defense

**Registered before implementation/running** (this file is committed before
`experiments/averaging_attack.py` produces any numbers).

## Target

Easy cell **hmda/race/loan_decision**, e2e surgical channel (Exp 10 arm D:
learned orthonormal subspace Q via differentiable QR, exposed rep
h' = h + sigma * (z @ Q^T), fresh z per draw), at its Tier-1 certified
operating point.

**Operating-point verification (flagged):** the task brief said k=32, sigma=8.
The repo's config for the easy-cell e2e surgical winner is
**rank=32, sigma=16, lambda=10** (`results/end_to_end_surgical.json` arm D
`best`, and `E2E_SUR_WINNERS` in `experiments/two_tier_certification.py`).
sigma=8 belongs to the *hard* cell's surgical winner (adult/sex/income,
r=32 sigma=8 lambda=100) and to the e2e-*blunt* easy Tier-1 point. The
experiment runs at the verified point: **r=32, sigma=16, lambda=10, train
seed 0** (the config whose single-draw numbers exist in the JSONs).

## Attack

The attacker observes, for every input, N independent draws of the exposed
representation from the same frozen model and averages them. Tier-1 suite
(XGB, MLP-256-256, rank-32 ReLU LoRA), 75/25 attribute-stratified splits,
probe seeds {0, 1} (the main experiments' protocol). N in {1, 2, 4, 8, 16}.

## Prediction

Averaging N i.i.d. Gaussian draws is *distributionally identical* to a single
draw at effective noise sigma_eff = sigma / sqrt(N) (exact for this channel:
the average of N draws of h + sigma*(z_i @ Q^T) is h + (sigma/sqrt(N))*(zbar @ Q^T)
with zbar standard normal). So Tier-1 recovery at N should approximately match
single-draw recovery at sigma/sqrt(N). The nearest stored single-draw anchors
are the r=32, lambda=10 grid rows of `end_to_end_surgical.json` (models
*retrained* at the lower sigma, so the match is approximate — training sigma
also shapes the learned rep — but they are the registered quantitative anchors):

| N  | sigma_eff | anchor (single-draw row)        | predicted Tier-1 reading |
|----|-----------|---------------------------------|--------------------------|
| 1  | 16.0      | winner r=32 s=16: XGB 0.5060, MLP 0.5017, LoRA 0.4994 | must REPRODUCE these (sanity gate) |
| 2  | 11.3      | between s=16 and s=8 rows       | ~0.51 (monotone interpolation) |
| 4  | 8.0       | r=32 s=8 row: XGB 0.5216, MLP 0.5146 | near 0.515–0.522 |
| 8  | 5.7       | between s=8 and s=4 rows        | ~0.53 |
| 16 | 4.0       | r=32 s=4 row: XGB 0.5411, MLP 0.5408 | near 0.54 |

Directional claims (the registered content):

1. Tier-1 max recovery is **monotonically increasing in N**.
2. N=4 reads near the sigma=8 single-draw row; N=16 near the sigma=4 row
   (the sqrt-N law).
3. At N=16 recovery approaches but likely does not yet cross the 0.55 bar
   (the sigma=4 retrained row reads 0.541); extrapolating the law, N ≈ 64+
   would breach. As N → infinity the average converges to the clean rep h,
   whose in-span(Q) attribute content the informed LRT already reads at 0.84
   — so the Tier-1 certificate at this point is meaningful only under
   single-draw exposure.

Falsified if: recovery is flat in N (no averaging gain), or deviates from the
sigma/sqrt(N) anchors by much more than the seed-noise scale (~0.01 AUC), or
the N=1 sanity gate fails to reproduce the stored winner numbers.
