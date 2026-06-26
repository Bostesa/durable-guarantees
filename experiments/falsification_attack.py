"""Falsification attack: PCRL's compliance certificate does not survive
downstream LoRA fine-tuning.

Thesis under test
-----------------
PCRL ships a per-purpose representation whose *compliance certificate* is a
LOW linear R^2 between the frozen 64-dim representation h_p and a forbidden
sensitive attribute (here "sex" under purpose income_prediction, idx 0). The
certificate is checked at training time, on the frozen pipeline.

The attack
----------
We freeze the entire PCRL pipeline and attach a FRESH, trainable rank-8 LoRA
adapter strictly ON TOP of the frozen 64-dim representation, plus a small
linear head. We train (LoRA + head) for 300 steps to predict "sex". This is
exactly the downstream-adaptation mechanism PCRL itself uses (per-purpose
LoRA), now wielded by an attacker who only ever sees the certified frozen
representation.

Every 10 steps we re-measure the certificate — linear R^2 and dominant-axis
R^2 — between the LoRA-ADAPTED representation and "sex", using PCRL's OWN
certificate code (pcrl.vision.r2_helper.linear_r2 and
pcrl.evaluation.certificates.compute_dominant_axis_r2). The PCRL pipeline
never changes; only the attacker's adapter does.

Expected: baseline R^2 low (certificate at rest), then R^2 climbs sharply
within tens of steps — the certificate is a training-time-only guarantee.

Run locally on MPS/CPU. Never touches AWS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.pcrl_io import (  # noqa: E402
    ADULT_LORA_ALPHA,
    ADULT_LORA_RANK,
    CHECKPOINT,
    INCOME_PREDICTION_PURPOSE_IDX,
    LoRAAdapter,
    build_adult_train_loader,
    compute_dominant_axis_r2,
    extract_attr,
    extract_frozen_reps,
    get_device,
    linear_r2,
    load_adult_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Certificate threshold PCRL uses for the per-pair auditor metric (TAU).
TAU = 0.05

# Attack hyperparameters.
N_STEPS = 300
EVAL_EVERY = 10
BATCH_SIZE = 512
LR = 1e-2
SEED = 0
REPR_DIM = 64
SENSITIVE_ATTR = "sex"


class FreshLoRAAttacker(nn.Module):
    """A fresh rank-8 LoRA adapter on top of the frozen 64-dim repr, + a head.

    This mirrors EXACTLY the per-purpose-LoRA mechanism PCRL itself uses: a
    strict linear LoRA on a (frozen) Linear, followed by the downstream
    nonlinearity. In PCRL's backbone every LoRA-adapted Linear is followed by
    BatchNorm -> ReLU; we attach the same shape on top of the certified repr::

        linear_adapted = h + LoRAAdapter(h)        # affine re-basis of h (CONTROL)
        adapted        = ReLU(linear_adapted)      # 64-dim, what we probe (ATTACK)
        logits         = head(adapted)             # linear head -> sex

    LoRAAdapter is PCRL's own module (B(A(h)) * alpha/rank + bias), B
    zero-initialised, so at step 0 LoRAAdapter == 0 and:
      * linear_adapted == h  -> its R^2 equals the frozen certificate exactly;
      * adapted == ReLU(h)   -> the attack curve's starting point.

    Why the ReLU is necessary (and not fudging): linear/dominant-axis R^2 are
    INVARIANT under any invertible affine map of the features, so a strictly
    linear LoRA leaves the certificate at its baseline value forever (verified
    empirically: flat at 0.03494 for 300 steps). The certificate is blind to
    linear re-bases. A single downstream nonlinearity — present in every real
    LoRA stack — is what surfaces the nonlinearly-encoded sensitive info that
    the linear certificate never bounded. We keep ``linear_adapted`` as a logged
    control so the invariance stays visible.

    Only the LoRA + linear head are trainable; the frozen PCRL pipeline is not
    part of this module. The head is LINEAR so that minimising sex-CE directly
    pushes the adapted representation to be linearly sex-decodable.
    """

    def __init__(self, repr_dim: int = REPR_DIM, n_classes: int = 2,
                 rank: int = ADULT_LORA_RANK, alpha: float = ADULT_LORA_ALPHA):
        super().__init__()
        self.adapter = LoRAAdapter(
            in_features=repr_dim, out_features=repr_dim, rank=rank, alpha=alpha, dropout=0.0
        )
        self.head = nn.Linear(repr_dim, n_classes)

    def linear_adapted(self, h: torch.Tensor) -> torch.Tensor:
        """Pre-activation LoRA output: an affine re-basis of h (R^2-invariant control)."""
        return h + self.adapter(h)

    def adapt(self, h: torch.Tensor) -> torch.Tensor:
        """The LoRA-adapted representation probed by the certificate (post-activation)."""
        return torch.relu(self.linear_adapted(h))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapt(h))


def measure_certificate(H: np.ndarray, y: np.ndarray) -> dict:
    """Measure PCRL's certificate on representation H vs attribute y.

    Uses PCRL's own instruments: train-set Tikhonov one-hot linear R^2 and the
    dominant-axis (max one-vs-rest) R^2. We pass float64 to ``linear_r2`` (same
    representation values, just higher precision): post-ReLU features are
    rank-deficient and float32 makes the tiny-ridge Gram solve ill-conditioned,
    producing spurious R^2 dropouts. ``compute_dominant_axis_r2`` already casts
    to float64 internally; we match that so the two instruments agree (for a
    binary attribute they are mathematically equal). PCRL's code is unmodified.
    """
    H = np.asarray(H, dtype=np.float64)
    lin = linear_r2(H, y)
    da = compute_dominant_axis_r2(H, y)
    return {
        "linear_r2": float(lin),
        "dominant_axis_r2": float(da["r2_da"]),
        "da_argmax_class": int(da["argmax_class"]),
        "da_per_class_r2": [float(x) for x in da["per_class_r2"]],
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[setup] device={device}  checkpoint={CHECKPOINT}")

    # 1. Load the Adult s0 model and the Adult train data (purpose_idx 0).
    encoder, lora_target = load_adult_encoder(CHECKPOINT)
    print(f"[setup] encoder loaded; auto-detected lora_target={lora_target!r}")
    purposes, train_ds, train_loader = build_adult_train_loader(batch_size=BATCH_SIZE)
    print(f"[setup] N_train={len(train_ds)}  D={train_ds.info.num_features}  "
          f"purpose[0]={purposes[INCOME_PREDICTION_PURPOSE_IDX].name}")

    # 2. Frozen 64-dim representation for ALL training samples (before task head).
    H = extract_frozen_reps(encoder, train_loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    y = extract_attr(train_loader, SENSITIVE_ATTR)
    assert H.shape[0] == y.shape[0], (H.shape, y.shape)
    print(f"[setup] frozen repr H={H.shape}  per_dim_std mean={H.std(0).mean():.4f} "
          f"max={H.std(0).max():.4f}")
    sex_counts = np.bincount(y)
    print(f"[setup] sex distribution (0=F,1=M): {sex_counts.tolist()}  "
          f"majority={sex_counts.max() / sex_counts.sum():.4f}")

    # 3. Baseline certificate at rest — should be LOW.
    baseline = measure_certificate(H, y)
    print(f"\n[baseline] certificate at rest (frozen PCRL repr vs '{SENSITIVE_ATTR}'):")
    print(f"           linear R^2        = {baseline['linear_r2']:.5f}")
    print(f"           dominant-axis R^2 = {baseline['dominant_axis_r2']:.5f}")
    print(f"           threshold TAU     = {TAU}")

    # Guardrail: if the certificate is NOT low at rest, something loaded wrong.
    # Do not fudge — stop and report.
    if baseline["linear_r2"] > 0.20:
        print("\n[ABORT] Baseline linear R^2 is NOT low "
              f"({baseline['linear_r2']:.5f} > 0.20). Something loaded wrong "
              "(check weights_only=False, adapter layout, purpose_idx). "
              "Refusing to continue and fudge the result.")
        sys.exit(2)

    # 4. Attach a FRESH trainable LoRA (rank 8) + head; train 300 steps for sex.
    H_t = torch.from_numpy(H).to(device)
    y_t = torch.from_numpy(y).long().to(device)
    n = H_t.shape[0]

    attacker = FreshLoRAAttacker().to(device)
    opt = torch.optim.Adam(attacker.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(SEED)

    print(f"\n[attack] fresh LoRA rank={ADULT_LORA_RANK} alpha={ADULT_LORA_ALPHA} "
          f"+ linear head; {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR}")
    print(f"[attack] trainable params: "
          f"{sum(p.numel() for p in attacker.parameters() if p.requires_grad)}")

    curve = []

    def log_point(step: int, loss_val: float | None) -> None:
        attacker.eval()
        with torch.no_grad():
            H_attack = attacker.adapt(H_t).cpu().numpy().astype(np.float32)
            H_linear = attacker.linear_adapted(H_t).cpu().numpy().astype(np.float32)
        cert = measure_certificate(H_attack, y)              # post-ReLU (the attack)
        ctrl = measure_certificate(H_linear, y)              # pre-ReLU affine (control)
        attacker.train()
        rec = {
            "step": step,
            "loss": loss_val,
            "linear_r2": cert["linear_r2"],
            "dominant_axis_r2": cert["dominant_axis_r2"],
            "da_argmax_class": cert["da_argmax_class"],
            # Control: strictly-linear LoRA output — should stay ~flat at baseline.
            "control_linear_r2": ctrl["linear_r2"],
            "control_dominant_axis_r2": ctrl["dominant_axis_r2"],
        }
        curve.append(rec)
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "  --  "
        print(f"  step {step:3d} | loss {loss_str} | "
              f"ATTACK linR² {cert['linear_r2']:.5f} domR² {cert['dominant_axis_r2']:.5f} "
              f"| control(linear-LoRA) linR² {ctrl['linear_r2']:.5f}")

    # Step 0: untrained adapter (B=0) => LoRAAdapter==0, so the linear control
    # output == h (control R^2 == frozen baseline exactly) and the probed attack
    # representation == ReLU(h) (the climb's starting point).
    log_point(0, None)

    attacker.train()
    for step in range(1, N_STEPS + 1):
        idx = torch.randint(0, n, (BATCH_SIZE,), generator=g).to(device)
        hb, yb = H_t[idx], y_t[idx]
        logits = attacker(hb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            log_point(step, float(loss.item()))

    final = curve[-1]
    print(f"\n[result] frozen baseline linear R^2          = {baseline['linear_r2']:.5f}")
    print(f"[result] ATTACK final  linear R^2 (post-ReLU) = {final['linear_r2']:.5f}  "
          f"(x{final['linear_r2'] / max(baseline['linear_r2'], 1e-6):.1f} baseline)")
    print(f"[result] ATTACK final  dom-axis R^2           = {final['dominant_axis_r2']:.5f}")
    print(f"[result] CONTROL (linear LoRA) final linear R^2 = {final['control_linear_r2']:.5f}  "
          f"(affine-invariant: stays at baseline)")

    # First step where the certificate is breached (linear R^2 > TAU).
    breach = next((c["step"] for c in curve if c["linear_r2"] > TAU), None)
    print(f"[result] certificate breached (attack linear R^2 > {TAU}) at step: {breach}")

    # 6. Save curves (JSON) and plot.
    out = {
        "experiment": "falsification_attack",
        "checkpoint": str(CHECKPOINT),
        "purpose": purposes[INCOME_PREDICTION_PURPOSE_IDX].name,
        "purpose_idx": INCOME_PREDICTION_PURPOSE_IDX,
        "sensitive_attr": SENSITIVE_ATTR,
        "lora_target": lora_target,
        "tau": TAU,
        "device": str(device),
        "config": {
            "n_steps": N_STEPS, "eval_every": EVAL_EVERY, "batch_size": BATCH_SIZE,
            "lr": LR, "seed": SEED, "rank": ADULT_LORA_RANK, "alpha": ADULT_LORA_ALPHA,
        },
        "n_train": int(n),
        "repr_per_dim_std_mean": float(H.std(0).mean()),
        "repr_per_dim_std_max": float(H.std(0).max()),
        "baseline": baseline,
        "final": {
            "attack_linear_r2": final["linear_r2"],
            "attack_dominant_axis_r2": final["dominant_axis_r2"],
            "control_linear_r2": final["control_linear_r2"],
            "control_dominant_axis_r2": final["control_dominant_axis_r2"],
        },
        "breach_step_linear": breach,
        "notes": (
            "ATTACK = post-ReLU LoRA-adapted repr (probed). CONTROL = pre-ReLU "
            "strict-linear LoRA output; linear/dominant-axis R^2 are invariant "
            "under invertible affine maps, so the control stays at the frozen "
            "baseline — the nonlinearity is what breaks the certificate."
        ),
        "curve": curve,
    }
    json_path = RESULTS_DIR / "falsification_attack.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] curves -> {json_path}")

    _plot(curve, baseline, json_path.with_suffix(".png"))
    print(f"[save] plot   -> {json_path.with_suffix('.png')}")


def _plot(curve, baseline, png_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [c["step"] for c in curve]
    lin = [c["linear_r2"] for c in curve]
    da = [c["dominant_axis_r2"] for c in curve]
    ctrl = [c["control_linear_r2"] for c in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    # ATTACK: post-ReLU LoRA-adapted repr. linear R² == dominant-axis R² for a
    # binary attribute, so plot linear as the prominent solid line and overlay
    # dominant-axis as a thin dashed line to show they coincide.
    ax.plot(steps, lin, marker="o", ms=4, color="C3", lw=2,
            label="ATTACK linear R² (post-ReLU LoRA repr) — PCRL certificate")
    ax.plot(steps, da, color="C1", lw=1, ls="--",
            label="ATTACK dominant-axis R² (coincides — binary attr)")
    ax.plot(steps, ctrl, marker="^", ms=4, color="C0",
            label="CONTROL linear R² (strict-linear LoRA — affine-invariant)")
    ax.axhline(baseline["linear_r2"], color="gray", ls=":", lw=1.2,
               label=f"frozen baseline (certificate at rest) = {baseline['linear_r2']:.3f}")
    ax.axhline(TAU, color="black", ls="--", lw=1.2, label=f"τ = {TAU} certificate threshold")
    ax.fill_between(steps, TAU, lin, where=[v > TAU for v in lin],
                    color="C3", alpha=0.08)
    ax.set_xlabel("attacker LoRA training step")
    ax.set_ylabel("R²(representation, sex)")
    ax.set_ylim(0, max(max(lin), max(da)) * 1.15)
    ax.set_title("PCRL compliance certificate vs. downstream LoRA fine-tuning\n"
                 "(frozen pipeline; fresh attacker LoRA on top of certified repr)")
    ax.legend(loc="center right", fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
