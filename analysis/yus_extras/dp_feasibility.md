# DP feasibility audit — a differentially private isolate-then-noise release

Report-only (2026-07-23). No mechanism, experiment, or result file modified; no
runs executed. Scope: the per-input **release mechanism** of the e2e surgical
channel (the isolate-then-noise defense) at its three surviving Tier-1 operating
points (easy r=32 σ=16 λ=10, middle r=16 σ=16 λ=100, hard r=32 σ=16 λ=100).
DP granularity audited: per-release input privacy (local-model / per-record
Gaussian mechanism on the released representation). DP of the *trained model
itself* (DP-SGD) is a separate, larger question flagged in §4.

## 1. Is the released representation bounded? — No. Checked in code.

The full release path, `experiments/end_to_end_surgical.py::train_e2e_surgical`:

```python
130:    bn = nn.BatchNorm1d(PUB_DIM).to(device)
...
163:            h = bn(E(X_t[i:i + 4096]))
165:            hn = h + sigma * (z @ Q.T)   # one fresh draw per row (exposed)
```

There is **no clip, clamp, renorm, or normalization bound anywhere** on h or on
the subspace coordinates u = Qᵀh before noise: grepping
`clamp|clip|renorm|F.normalize|normalize(` across `end_to_end_surgical.py`,
`sealed_channel.py`, `mi_ceiling.py`, `targeted_noise.py` returns zero matches in
any release path. The only scale control is `BatchNorm1d` (line 130), which is
**not a bound**: eval-mode BN standardizes by *running statistics* and then
applies a *learnable* affine (γ, β) — per-example outputs are unbounded (an
outlier input maps to arbitrarily large h), and even the per-dimension scale is
γ-learned, not pinned to 1. The full-rank channel is the same shape
(`sealed_channel.py:123`, `mi_ceiling.py:256`: `bn(P) + σ·randn`): BN'd,
unbounded, unclipped.

## 2. Consequences, and the Gaussian-mechanism requirement

**(a) As built, no finite (ε, δ) exists at any σ.** Two independent blockers,
either of which alone is fatal:

- **Unbounded sensitivity.** The Gaussian mechanism's guarantee needs a finite
  L2 sensitivity Δ₂ = sup over neighboring inputs of ‖u(x) − u(x′)‖₂. With
  unclipped BN outputs Δ₂ = ∞, so ε = ∞ for every σ, including σ=16.
- **The noiseless complement — the mechanism's defining feature.** Noise lives
  *only* in span(Q); the complement h⊥ = h − QQᵀh (the task content, by the
  isolate-then-noise thesis) is released **deterministically** (line 165: the
  noise term is `z @ Q.T`, rank r of 64). Any two inputs differing in h⊥ are
  perfectly distinguishable from the release, so the *full 64-d release* has
  ε = ∞ regardless of σ and regardless of any clipping on u. A DP claim can
  therefore only ever cover the **subspace-projected view** Qᵀh̃ (a strictly
  smaller release), or the mechanism must add noise to all 64 directions —
  which is the full-rank channel, i.e. a different arm of the paper.

**(b) With a clip: the requirement.** Clip the subspace coordinates to an L2
ball of radius C before noise, ū = u · min(1, C/‖u‖₂), then release
ū + σz, z ∼ N(0, I_r). Worst-case per-release sensitivity between any two
inputs is Δ₂ = 2C, and the *subspace view* of one release is (ε, δ)-DP with the
classical Gaussian-mechanism calibration

    ε(C, σ, δ) = (2C / σ) · sqrt(2 ln(1.25/δ))     (per release; valid ε ≲ 1,
                                                    Balle–Wang analytic GM is
                                                    tighter above that)

All three surviving points share σ = 16, so at δ = 10⁻⁵
(sqrt(2 ln 1.25e5) = 4.845): **ε ≈ 0.606 · C**, and at δ = 10⁻⁶: ε ≈ 0.662 · C.

| clip C | ε @ δ=1e-5 | ε @ δ=1e-6 |
|---|---|---|
| 1.0 | 0.61 | 0.66 |
| 1.65 | **1.00** | 1.09 |
| 2.0 | 1.21 | 1.32 |
| 4.0 | 2.42 | 2.65 |
| √32 ≈ 5.66 | 3.43 | 3.75 |
| 8.0 | 4.84 | 5.30 |

So a respectable ε = 1 per release at the current σ = 16 requires **C ≈ 1.65**.
(δ = 10⁻⁵ is consistent with the ≲ 1/n convention at n = 63,747 / 24,145.)

**Composition — the averaging attack is exactly the missing DP ledger.** k
queries of the same input compose; for the Gaussian mechanism k identical
releases at σ are *exactly* one release at σ/√k — the same algebra the
averaging attack exploited empirically (N=16 ⇒ σ_eff=4 ⇒ middle-point T1
0.93). Under a DP release the same attack is not a silent break but a priced
ledger entry: ε grows ≈ √k·ε₁ (exact Gaussian RDP composition), and the vendor
can meter or cap queries per input. This is the strongest argument *for* the DP
retrofit: it converts the single-draw-exposure assumption from an unstated
caveat into an explicit, composable budget.

## 3. Empirical subspace-coordinate norms — saved representations do NOT exist

Checked: the repo contains **no saved representation arrays** for any operating
point. The only npz files anywhere are `analysis/yus_extras/tpr_scores/*.npz`
(attacker class-posterior scores + labels — probs only, verified by key
inspection) and the raw-data caches in `data_cache/`. Neither
`run_tpr.py` nor `averaging_attack.py` nor any experiment persists h, P, or Q.
So the empirical distribution of ‖Qᵀh‖₂ at the three surviving points cannot be
read off disk. Regenerating it would be deterministic and cheap (the three
frozen models rebuild bit-exactly — proven by the 230/230 gate — in ≈1 min
total, no training beyond the pinned pipeline), but that is a run, and this job
is no-runs; flagged as the immediate follow-up.

What can be said without the run:

- **The requested fraction is definitional: clipping at C = median ‖u‖ distorts
  exactly 50% of inputs** — that is what a median is. The empirical
  distribution is *not* needed for that number; it is needed for (i) the
  *magnitude* of distortion on the clipped half (the excess-norm distribution),
  and (ii) the ε that gentler quantiles buy (e.g. C = p99 distorts 1% of inputs
  but with a larger C, hence larger ε per the table above).
- **Scale proxy (estimate, not measurement):** h is eval-mode BN output, so
  per-dimension scale ≈ the learned γ (≈ 1 if BN's affine stays near init).
  Under that proxy E‖u‖² ≈ r, so median ‖u‖ ≈ √r ≈ **5.7** (r=32: easy, hard)
  or **4.0** (r=16: middle), putting C = median at **ε ≈ 3.4 / 2.4** per
  release (δ=1e-5) — weak DP. Reaching ε = 1 (C ≈ 1.65) would clip and
  distort the large majority of inputs at these scales. This proxy ignores the
  learned γ and any variance concentration in the learned Q directions; treat
  it as order-of-magnitude until the one-minute regeneration is run.

## 4. Retraining vs reanalysis

**Reanalysis only (no training):**
- The DP accounting itself: ε(C, σ, δ) is closed-form; the k-query composition
  ledger (averaging) is exact Gaussian RDP arithmetic.
- Measuring the ‖Qᵀh‖ distributions: deterministic regeneration of the three
  frozen models (pinned seeds, bit-exact), no gradient steps.
- Scoping decision: whether the paper's DP claim covers the subspace view only
  (complement stays empirically-certified, guarantee becomes explicitly hybrid:
  DP on span(Q), attacker-battery-only off it) — a framing choice, not a run.

**Re-measurement without retraining (runs, but no training):**
- A deployment-time clip (clip inserted into the frozen release path) changes
  the released distribution, so every certificate at the clipped release —
  Tier-1/Tier-2 batteries, LRT, utility — must be re-measured. Utility will
  likely degrade because the head D was trained on unclipped noisy inputs.

**Retraining required:**
- **Clip-in-the-loop** (the honest version): encoder and head must adapt to the
  clipped ball, so the 3 points × 5 seeds retrain (≈10 min on MPS at measured
  rates), and the (r, σ, λ) operating points must be re-derived — under DP, σ
  is dictated by (ε, δ, C), not by the attacker sweep, and if the DP-required
  σ/C ratio lands above what utility tolerates, the sweep moves.
- **Covering the complement** (unavoidable for a full-release DP claim): add
  noise in all 64 directions — a two-scale channel (σ_span from the DP budget,
  σ⊥ > 0) sitting between arm C (blunt) and arm D (surgical). New mechanism ⇒
  full retraining, new sweeps, new certification. Note the paper already owns
  the limiting case: the full-rank blunt channel is the DP-compatible shape,
  and its certified points (σ = 8–64) are the existing evidence for what
  all-direction noise costs in utility.
- **Out of scope but must be stated in any claim:** the encoder, BN running
  statistics, Q, and head are all functions of the training data, so "DP
  release" ≠ "DP model" — a guarantee against training-set leakage through the
  published model would additionally require DP training (DP-SGD), a different
  project.

**Bottom line:** the current mechanism admits no finite (ε, δ) at any σ — not
because σ is too small but structurally: unbounded BN outputs (no clip exists
in code) and a deterministic complement subspace. A DP retrofit of the subspace
view alone is cheap (clip at C ≈ 1.65 for ε = 1 at σ = 16, δ = 1e-5, plus a
3-point retrain), but at BN scale that clip likely distorts most inputs; a DP
guarantee for the *full* release requires abandoning the mechanism's defining
noiseless-complement feature — i.e., it is a different channel, and the
utility price of that channel is already measured in the paper's full-rank arm.
