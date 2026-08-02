# Registered prediction — isolate-then-noise vs plain full-rank noise, matched

Registered 2026-07-24, before implementation. Answers Reviewer A's objection
(NeurIPS blind review v35): "the isolate half fails Tier-2 on its own, so
isolate-then-noise reduces to the Gaussian mechanism." The head-to-head makes
the honest tradeoff a measured table instead of an asserted sentence.

## Protocol

For each of the three headline cells, train both channels at 3 seeds
(TRAIN_SEEDS 0,1,2) and report, on shared probe seeds [0,1]:
- isolate-then-noise: `train_e2e_surgical` at the Table-7 winner config
  (easy r=32 sigma=16 lam=10, middle r=16 sigma=16 lam=100,
  hard r=32 sigma=16 lam=100).
- plain full-rank noise: `mi_ceiling.train_channel` (e2e blunt, all-dim
  Gaussian), sigma tuned to MATCH the isolate channel's black-box Tier-1 max
  within 0.01 (grid-searched from the stored two_tier tier-1 sigmas
  8/12/32, adjusted per cell).

Both measured with the same instruments: black-box Tier-1 max = max(XGB, MLP,
LoRA-32) on the exposed rep; Tier-2 = channel-aware Gaussian LRT
(iso cov for full-rank, subspace cov QQ^T for isolate); utility = own-head
lift over all rows / clean lift (in-sample convention, the arm's convention).
"Matched Tier-1" = both channels' Tier-1 max within 0.01 of each other.

## Registered claims

1. **Utility at matched Tier-1: isolate wins on every cell.** Retained utility
   (isolate) > retained utility (full-rank) for easy, middle, and hard, and the
   gap widens with coupling (smallest on easy, largest on hard).
2. **Tier-2 at matched Tier-1: isolate leaks, full-rank does not.** Isolate's
   Tier-2 LRT AUC stays >= 0.60 at the matched point (the noiseless complement),
   while full-rank's Tier-2 LRT is strictly lower and closer to the 0.55 bar.
3. **No free lunch (the synthesis):** there is no single channel that both
   retains isolate-level utility AND reaches full-rank Tier-2 protection. To
   pass Tier-2 you give up the isolate utility advantage; the isolate subspace
   buys Tier-1 utility at lower noise, not Tier-2 protection.

Falsified if: full-rank retains >= isolate utility on any cell at matched
Tier-1 (claim 1), or isolate's Tier-2 LRT falls below full-rank's on any cell
(claim 2), or a single configuration achieves both isolate-utility and
full-rank-Tier-2 (claim 3). Literal scoring on the 3-seed means; per-config
reproduction sanity vs stored two_tier numbers where a stored point exists.
