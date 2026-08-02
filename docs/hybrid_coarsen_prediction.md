# Registered prediction — isolate-noise-coarsen hybrid (committed before implementation)

Registered 2026-07-24, at repo HEAD 21ad18c, BEFORE any implementation or run.
Results enter the paper only if the success criterion below is met, and only
after author sign-off; anything less is next-paper material.

## Motivation

The subspace channel fails Tier-2 because the complement h_perp is released
noiselessly at full precision (informed LRT 0.66–0.86 at the three operating
points). FARE passes Tier-2 with a finite-support deterministic release.
Hypothesis: coarsening h_perp kills the residual fine-grained attribute signal
the HSIC penalty left behind, while a TASK-AWARE coarsening preserves the task
signal that migrated there. No code from eth-sri/fare is used (no license);
the coarsener is plain sklearn.

## Pinned comparators (sha256)

- isolate_vs_fullrank.json 471a4ccd — subspace operating points: Tier-1 max
  0.5094 / 0.5349 / 0.5369, LRT 0.8542 / 0.8640 / 0.6617, utility 101.6 /
  100.4 / 84.3 % (easy / middle / hard).
- two_tier_certification.json e380d61f — clean-lift denominators
  0.02228 / 0.40912 / 0.14349 (e2e own-head, the same denominators the
  comparators use).
- dp_fullrank.json 76239b88 — full-rank Tier-2 utility anchors 55.5 / 21.7 /
  −7.0 %.
- fare_gauntlet.json be9dc53f — FARE both-tier utility 0 / 28.3 / 39.0 %.
- Instruments: experiments/end_to_end_surgical.py 1870c688 (train_e2e_surgical
  unchanged), utils/battery.py 64162a62 (gaussian_lrt unchanged).

## Design — ONE configuration per cell, fixed a priori (no tuning anywhere)

Base models: the three existing e2e subspace operating points —
easy hmda/race/loan_decision r=32 σ=16 λ=10; middle hmda/race/loan_amount_band
r=16 σ=16 λ=100; hard adult/sex/income r=32 σ=16 λ=100 — retrained via
`train_e2e_surgical(return_clean=True)`, training seeds [0,1,2].

REPRODUCTION GATE (per cell, before any modification): 3-seed mean Tier-1 max
(XGB+MLP+LoRA32, probe seeds [0,1] — the stored instrument) and 3-seed mean
subspace-cov Gaussian LRT must each reproduce the stored isolate_vs_fullrank
values (T1 0.5094/0.5349/0.5369; LRT 0.8542/0.8640/0.6617) within 0.01.
Gate failure on a cell invalidates that cell; report, do not tune.

Defense partition: per training seed, stratified 75/25 split on the task label
y, random_state = 1000 + train_seed (distinct from probe seeds). Everything the
defense fits (tree, leaf medians, retrained head) uses the 75% defense-train
rows only; the release is constructed for all rows.

Coarsening: DecisionTreeClassifier(max_leaf_nodes=32, min_samples_leaf=100,
random_state=train_seed) fit on h_perp → y on defense-train rows, where
h_perp = h − (h@Q)@Q^T (clean pre-noise rep, learned Q). Release per row =
concat[ u_noised , m_leaf ] with u_noised = h@Q + σ·ε, ε ~ N(0, I_r) fresh per
input (identical in distribution to the subspace part of the original channel),
and m_leaf = elementwise median of h_perp over the row's leaf's defense-train
rows (64-dim). Release dim = r + 64.

Head retrain: same architecture as the e2e own head (single Linear(r+64 →
n_task)), same budget (Adam lr=1e-3, 1500 steps, batch 1024, seeded by
train_seed), trained on defense-train release rows. Encoder and Q frozen.

Attackers — both surfaces (release and y-hat = retrained-head logits over all
rows), probe conventions 75/25 stratified, probe seeds [0,1,2], bar 0.55, all
readings 3-train-seed means of per-seed probe-mean AUCs, max-of-means:

- Tier-1: XGB, MLP(256,256), LoRA rank 32 (the standing battery) on the
  release and on y-hat.
- Tier-2 (defense-aware, strongest honest attack), on the release:
  (a) Gaussian LRT on the subspace part knowing Q, σ, clean stats: class
      Gaussians fit on clean u = h@Q over probe-train rows, noise cov σ²I_r,
      scoring noised probe-test u (utils.battery.gaussian_lrt).
  (b) per-leaf attribute log-posterior: attacker knows the tree; per-leaf attr
      counts (Laplace +1) on probe-train rows → posterior for probe-test rows.
  (c) combined LLR: sum of the log-posteriors of (a) and (b), renormalized.
  (d) informed joint XGB on concat[LRT posterior, leaf posterior, release],
      fit on probe-train.
  Tier-2 reading = max of (a)–(d). Multiclass attr (hmda race): macro-OVR AUC
  throughout, as everywhere in the battery.
- Complement-side informed reading (for P2) = max of (b) and the Tier-1
  battery run on the m_leaf block alone.

Utility: own-head lift (accuracy − global task majority) over the stored clean
lifts above, reported in-sample (defense-train rows, the convention matching
the comparators' all-rows own-head numbers) and on the 25% defense-held-out
rows. P4 is scored on the in-sample convention.

## Registered predictions

P1. Tier-1 still passes on all three cells (coarsening does not break
    black-box protection): release T1 max ≤ 0.55 and y-hat T1 max ≤ 0.55,
    3-seed means, every cell.
P2. The complement-side informed reading drops from the stored 0.66–0.86 LRT
    (prompt range 0.66–0.94 across all operating points) to ≤ 0.60 on at
    least two of three cells.
P3. Combined Tier-2 reading (max of (a)–(d)) passes the 0.55 bar on at least
    two of three cells.
P4. Utility kept ≥ 70% on easy and middle, ≥ 50% on hard (in-sample
    convention) — i.e. beats FARE's 0 / 28.3 / 39.0 and full-rank Tier-2's
    55.5 / 21.7 / −7.0 wherever P3 passes.

Most likely failure mode, named a priori: the tree's y-pure leaves may
correlate with s exactly where label↔attribute coupling is high, so the HARD
cell (coupling 0.603) is the expected P3 failure — its leaf posterior (b)
reads above 0.55 even though its subspace LRT (a) stays near the bar.

## Success criterion for any paper mention

P1 holds everywhere, P3 holds on ≥ 2 cells, and P4 holds on the P3-passing
cells, with zero gate failures. Anything less: results are committed as
next-paper material and reported; the submission does not change.

Outputs: results/hybrid_coarsen.json +
predictions/hybrid_coarsen_results.md scoring each prediction
held/falsified with per-cell readings vs the three comparators.
