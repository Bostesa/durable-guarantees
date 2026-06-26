# durable-guarantees

Empirical tests of whether ML "compliance guarantees" are *durable* — i.e.
whether they survive realistic downstream use — or are merely **training-time
artifacts** that an adapter can dissolve.

## Experiment 1 — Falsification attack on PCRL's compliance certificate

PCRL (Purpose-Conditioned Representation Learning) ships a per-purpose
representation with a **compliance certificate**: the linear R² between the
frozen 64-dim representation `h_p` and a forbidden sensitive attribute is low
(below τ = 0.05). This certificate is checked at training time on the frozen
pipeline.

**Claim under test:** the certificate does *not* survive downstream LoRA
fine-tuning.

**Setup** (`experiments/falsification_attack.py`):

1. Load the audited Adult seed-0 model (`checkpoints/v2_adult_s0/final.pt`,
   `weights_only=False`) and the Adult train data under purpose
   `income_prediction` (`purpose_idx = 0`). The PCRL pipeline stays **frozen**.
2. Compute the frozen 64-dim representation (before the task head) for every
   training sample.
3. **Baseline:** measure the certificate — linear R² and dominant-axis R² —
   between the frozen representation and `sex`, using **PCRL's own certificate
   code** (`pcrl.vision.r2_helper.linear_r2` and
   `pcrl.evaluation.certificates.compute_dominant_axis_r2`). This is the
   certificate at rest; it is low.
4. Attach a **fresh, trainable rank-8 LoRA** (PCRL's own `LoRAAdapter`) strictly
   *on top of* the frozen representation, plus a small linear head, and train
   300 steps to predict `sex`. The attacker only ever touches the certified
   representation — the same per-purpose-LoRA mechanism PCRL itself uses.
5. Every 10 steps, re-measure the certificate on the **LoRA-adapted**
   representation.
6. Save the R²-vs-step curves to `results/falsification_attack.json` and a plot
   to `results/falsification_attack.png` (baseline + τ marked).

**Expected:** baseline R² low, then R² climbs sharply within tens of steps —
the certificate is a training-time-only guarantee.

> Note: this backbone never reaches `per_dim_std >= 0.5` (documented, expected),
> so the frozen representation is partially low-variance. That is not a bug.

### Result (Adult s0, MPS)

| quantity | linear R² | dominant-axis R² | vs τ=0.05 |
|---|---|---|---|
| frozen repr (certificate at rest) | **0.035** | 0.035 | **PASS** |
| + strict-linear LoRA (control, trained) | 0.035 | 0.035 | PASS (invariant) |
| + post-ReLU LoRA (attack, 300 steps) | **0.104** | 0.104 | **FAIL** |

The certificate flips **PASS → FAIL** (≈3× baseline, breaching τ at the first
measured step) once a fresh rank-8 adapter is fit downstream — while the
PCRL pipeline never changes. See `results/falsification_attack.png`.

**Key subtlety (why the attack needs a nonlinearity — not fudging).** Linear
R² and dominant-axis R² are *invariant under any invertible affine map of the
features*. A strictly-linear LoRA on top of the frozen repr is exactly such a
map, so it leaves the certificate pinned at its baseline value forever — we
verified this empirically (the `CONTROL` curve is flat at 0.035 for all 300
steps). The certificate is **blind to linear re-bases**. The breach comes from
the single downstream nonlinearity (ReLU) that every real LoRA stack sits next
to — in PCRL's own backbone each LoRA-adapted `Linear` is followed by
`BatchNorm → ReLU`. The attacker mirrors that exactly (PCRL's own `LoRAAdapter`,
then a ReLU), and we keep the linear (pre-ReLU) output as a logged control so
the invariance stays visible. The takeaway: the certified low-R² geometry is a
property of the *frozen* representation only; one downstream nonlinear adapter
surfaces the nonlinearly-encoded sensitive information the linear certificate
never bounded. The certificate does not survive downstream fine-tuning.

### Layout

```
experiments/falsification_attack.py   # the experiment
utils/pcrl_io.py                       # PCRL-specific glue (imports the read-only PCRL repo)
results/                               # JSON curves + plot
requirements.txt
```

The PCRL repo is **never modified**. `utils/pcrl_io.py` adds it to `sys.path`
(default `/Users/nathansamson/PCRL`, override with the `PCRL_ROOT` env var) and
imports its code, including mirroring the `lora_target` auto-detection from
`scripts/crosspurp/run_eval_multi.py`.

### Run

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/falsification_attack.py
```

Runs locally on **MPS/CPU** only — never AWS.
