# Worst-case leakage — TPR at low FPR for the surviving operating points

Report-only analysis (2026-07-23). Per instruction, nothing was re-run; this file
answers whether per-example attacker scores exist, and — since they do not — states
exactly what re-execution would be needed and its estimated wall time, then stops.

## 1. Do per-example attacker scores exist? — No, nowhere

Every attacker in the codebase computes per-example scores in memory and returns
**only scalar AUC aggregates**. Checked exhaustively:

- `experiments/diagnostic.py::probe_sklearn` / `probe_lora` return
  `(mean_auc, std_auc)` floats; the `predict_proba` / softmax arrays are discarded
  after `roc_auc_score`.
- `experiments/hardening_test.py::_probe` returns `(acc, auc)`.
- `utils/battery.py::_lrt_scores` computes per-example Gaussian-LRT class posteriors
  but `gaussian_lrt` / `gaussian_lrt_channel` reduce them to `(mean, std)` AUC.
- No `np.save`/`np.savez`/pickle of scores anywhere in `experiments/` or `utils/`
  (the only `savez` calls are dataset caches in `utils/folktables_io.py`,
  `utils/expansion_io.py`, and the LAFTR export; `results/` contains no `.npy`/`.npz`).
- The results JSONs store aggregates only:
  - `results/hardening_seeds.json` → `exp10_5seed[].D.per_seed` and
    `hard_cell_sigma16_alternative.per_seed` are per-**train**-seed AUCs
    (probe-seed-averaged), per attacker, per surface — no scores.
  - `results/two_tier_certification.json` → `e2e_cells[].tier1/tier2.per_seed` are
    per-train-seed AUCs for xgb/mlp/lora/lrt — no scores.
  - `results/expansion_shards/*.json` are JSON aggregates only.

So TPR@FPR ∈ {0.01, 0.05} **cannot be computed from any stored artifact**: ROC
operating points below the full curve require the per-example score vectors, and
only the AUC (curve integral) was kept.

## 2. The surviving operating points the re-run would target

Subspace-confined channel, Tier-1 (5 train seeds × probe seeds {0,1,2}; source of
the current AUC aggregates: `results/hardening_seeds.json`):

| cell | surviving point | best stored attacker (rep surface, mean over seeds) |
|---|---|---|
| easy hmda/race/loan_decision | r=32, σ=16, λ=10 | XGB 0.5091 (`exp10_5seed[0].D`) |
| middle hmda/race/loan_amount_band | r=16, σ=16, λ=100 | LoRA 0.5312 (`exp10_5seed[1].D`) |
| hard adult/sex/income | r=32, σ=16, λ=100 (the σ=16 alternative — the stored σ=8 point flips at 5 seeds) | LoRA 0.5358 (`hard_cell_sigma16_alternative`) |

Config caveat (carried over from the operating-point report): the σ=16 alternative's
(rank, λ) are attested by the README/commit 8b218c1 message, not by keys inside the
JSON block.

Full-rank noise channel (5 train seeds × probe seeds {0,1,2}; source:
`results/two_tier_certification.json` → `e2e_cells`):

| cell | Tier-1 point | best T1-suite attacker | Tier-2 point | best in-suite attacker at T2 |
|---|---|---|---|---|
| easy | σ=8 | XGB 0.5351 | σ=20 | Gaussian LRT 0.5377 |
| middle | σ=12 | LoRA 0.5408 | σ=24 | Gaussian LRT 0.5496 |
| hard | σ=32 | LoRA 0.5311 | σ=64 | Gaussian LRT 0.5339 |

(At every Tier-2 point the channel-aware LRT is the strongest attacker, and its
per-example scores are closed-form class posteriors — no probe training needed,
just the score capture.)

## 3. What re-execution would be needed

A new script (no existing code path modified), mirroring
`experiments/averaging_attack.py`'s pattern of composing the existing probe
functions, but capturing the held-out `predict_proba` / softmax / LRT-posterior
vectors before the AUC reduction:

1. **Subspace points:** retrain `train_e2e_surgical` at the 3 surviving configs × 5
   train seeds = **15 trainings**, then per train seed run the battery per probe seed
   {0,1,2} on the exposed rep, saving scores (all three attackers, since the best
   attacker varies by train seed).
2. **Full-rank points:** retrain the e2e noised channel at 3 cells × 2 tiers × 5
   train seeds = **30 trainings**, battery + LRT score capture as above.
3. Compute TPR at FPR 0.01 / 0.05 per (train seed × probe seed), report mean and
   worst seed per point.

Determinism note: training seeds and probe splits are fully pinned
(`torch.manual_seed`, `train_test_split(random_state=seed, stratify=y)`), so the
re-run's AUCs should reproduce the stored `per_seed` aggregates — that reproduction
is the natural sanity gate, as in the averaging attack.

**Protocol decision required before running (flagging, not deciding):** hmda race is
5-class (`n_attr=5`; adult sex is binary). TPR@FPR is only defined for a binary
detection problem, so the two HMDA cells need a one-vs-rest convention — e.g.
worst-class OvR (most adversarial), or the minority-class-vs-rest reading. The AUCs
in the JSONs are macro-OvR, which does not commit to either.

## 4. Estimated wall time on MPS (grounded in stored runtimes)

Stored wall times for the *same* trainings + batteries (AUC-only; score capture adds
only array writes, negligible):

- `hardening_seeds.json` `exp10_5seed[].minutes` = 4.8 + 5.9 + 2.4 = 13.1 min for
  arms C **and** D at 5 seeds each → D-only ≈ **6–7 min**; the hard-cell σ=16
  alternative re-run ≈ **1–2 min** (hard cell is the cheapest, 2.4 min for C+D).
- `two_tier_certification.json` `e2e_cells[].minutes` = 8.2 + 5.6 + 2.3 = 16.1 min
  including the σ sweep + clean baseline + both tier certifications → tiers-only
  ≈ **8–9 min**.
- Cross-check: `averaging_attack.json` — one e2e-surgical training on the largest
  cell + 5 batteries × 2 probe seeds = 1.5 min total; each full battery ≈ 15 s.

**Total ≈ 16–18 min of measured compute; ≈ 25–30 min end-to-end on MPS** with
loader/startup overhead. Single local run, no AWS.

Stopping here per the job instruction ("if scores do not exist … stop. Do not re-run
anything in this job").
