# Registered prediction — hybrid + leaf randomized response (committed before implementation)

Registered 2026-07-24 at repo HEAD 819250f, BEFORE any implementation or run.
Direct follow-up to the isolate-noise-coarsen run (registered 364870e, run
eb989a1), which failed P3 on 0/3 cells because the coarsening tree's leaves are
s-correlated on every cell. This run attacks exactly that: randomize the leaf
identity before release.

Results enter the paper only if the success criterion below holds AND the
authors sign off. Anything less: next-paper material, submission
unchanged.

## Pinned provenance (sha256)

- hybrid_coarsen.json **7778a0fd** — the p=0 point this run must reproduce:
  Tier-2 max 0.5843 / 0.6026 / 0.5892, in-sample utility 107.6 / 94.4 / 83.4 %,
  Tier-1 rep 0.5337 / 0.5576 / 0.5489, Tier-1 out 0.5216 / 0.5711 / 0.5654,
  32 leaves on every cell (easy / middle / hard).
- run_hybrid_coarsen.py **8939c7db** — the mechanism this run modifies in
  exactly one place.
- isolate_vs_fullrank.json **471a4ccd** — subspace comparator (utility
  101.6 / 100.4 / 84.3 %, LRT 0.854 / 0.864 / 0.662).
- two_tier_certification.json **e380d61f** — clean-lift denominators
  0.02228 / 0.40912 / 0.14349.
- Comparators for the frontier table: full-rank Tier-2 utility 55.5 / 21.7 /
  −7.0 % (dp_fullrank.json 76239b88); FARE both-tier utility 0 / 28.3 / 39.0 %
  (fare_gauntlet.json be9dc53f).

## Mechanism — identical to hybrid_coarsen except one step

Everything is unchanged from the registered coarsen design: same three e2e
subspace operating points (easy r=32 σ=16 λ=10; middle r=16 σ=16 λ=100; hard
r=32 σ=16 λ=100), same 3 training seeds, same defense partition (stratified
75/25 on y, random_state = 1000 + train_seed), same
`DecisionTreeClassifier(max_leaf_nodes=32, min_samples_leaf=100,
random_state=train_seed)` fit on h_perp → y over defense-train rows, same
per-leaf medians.

The single change is at release: each row's leaf assignment is passed through
randomized response over the K = 32 leaves — with probability 1 − p the true
leaf is kept, with probability p the emitted leaf is drawn uniformly from all
K leaves. Release = [noised subspace part, unchanged, fresh ε per input] concat
[median of the EMITTED leaf]. The flip is fresh per release, matching the
single-release threat model. The task head is retrained through the flip noise
(fresh flips per training batch, the same way the subspace channel trains
through its Gaussian noise), same architecture and same budget as before;
encoder and Q stay frozen. Flip randomness uses a generator separate from the
channel-noise generator, so at p = 0 the channel draw is bit-identical to the
coarsen run.

## Guarantee accounting

Randomized response over K leaves carries a per-release ε on **leaf
membership**:

    eps_leaf = ln( (1 − p + p/K) / (p/K) ),  K = 32

logged at every sweep point. **This is a guarantee on leaf identity, not on the
attribute.** It bounds how much the emitted leaf reveals about the true leaf;
it says nothing directly about s, because s is a function of the data that the
leaf only partially carries. The results file must state this and must not
conflate the two. It is also per-release: it does not compose across repeated
releases of the same row.

## Sweep

p ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}, one config per cell, otherwise identical
to the registered coarsen design. 3 training seeds, probe seeds [0,1,2], bar
0.55 on the 3-seed mean, both surfaces (release and y-hat).

## Attackers

Tier-1 as before (XGB, MLP 256-256, LoRA rank 32) on the release and on y-hat.

Tier-2 is **fully informed**: it knows Q, σ, the clean statistics, the tree,
the leaf medians, AND the flip rate p. Members:
  (a) Gaussian LRT on the subspace part (knows Q, σ, clean stats);
  (b) **flip-aware** leaf posterior — the attacker recovers the emitted leaf ℓ'
      exactly from the released median, and computes
      P(s | ℓ') ∝ Σ_ℓ P(ℓ' | ℓ) · P(ℓ, s), with P(ℓ' | ℓ) the known randomized
      response kernel (1 − p + p/K on the diagonal, p/K off it) and P(ℓ, s)
      estimated from probe-train rows, on which the attacker knows both x
      (hence the true leaf, since it has the tree) and s;
  (c) the combined score, sum of the (a) and (b) log-posteriors, renormalized;
  (d) XGB on [LRT posterior, flip-aware leaf posterior, release].
Tier-2 reading = max over (a)–(d). An attacker that ignores the flip is a
strawman and is not scored.

## Gate

Reproduce the coarsen run's p = 0 readings within 0.01 before sweeping:
Tier-2 max 0.584 / 0.603 / 0.589 and in-sample utility 107.6 / 94.4 / 83.4 %.
Gate failure on a cell invalidates that cell; report, do not tune.

## Registered predictions

- **P1.** There exists a p at which Tier-2 passes (≤ 0.55, 3-seed mean, both
  surfaces) on at least two of the three cells.
- **P2.** At the smallest such p, in-sample utility stays ≥ 50% on those cells.
- **P3.** Tier-1 passes wherever Tier-2 passes — the flip also disperses the
  concentrated leaf signal that broke Tier-1 on the middle cell (rep 0.558,
  out 0.571) and the hard cell's output (0.565) in the coarsen run.
- **P4 — named failure mode.** The posterior over the TRUE leaf given the
  emitted leaf retains too much information at any p small enough to train
  through, so the frontier runs out of utility before it reaches the bar. Two
  concrete mechanisms: (i) at K = 32 the keep-probability is
  1 − p + p/K ≈ 1 − 0.97p, so even p = 0.5 leaves ~52% of rows carrying their
  true leaf, and the coarsen run's leaf-posterior excess over chance was only
  0.08–0.10 to begin with — halving it may not clear a bar that sits 0.03–0.05
  away; (ii) the HSIC penalty pushed the TASK signal into h_perp, so the leaf
  median is where the utility lives — the same flip that destroys the attribute
  signal destroys the task signal, and utility should fall roughly as fast as
  protection improves.

My own expectation, recorded for calibration: the crossing, if it exists, sits
near p ≈ 0.5, and the hard cell is again the most likely to miss — but the
binding risk here is P2 (utility), not P1, which is the reverse of the coarsen
run where the binding constraint was protection.

## Success criterion for any paper mention

P1 **and** P2 **and** P3 all hold on the SAME two-or-more cells, with zero gate
failures. Otherwise: commit as next-paper material and report; the submission
does not change.

Outputs: `results/hybrid_rr.json` + a results md scoring each
prediction, with the per-cell frontier (p, eps_leaf, Tier-1, Tier-2, utility)
against the three comparators (subspace, full-rank Tier-2, FARE).
