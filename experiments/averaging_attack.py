"""Averaging attack on the isolate-then-noise (e2e surgical) defense.

Registered prediction: docs/averaging_attack_prediction.md (committed
before this script existed). Easy cell hmda/race/loan_decision at the
verified Tier-1 operating point of the e2e surgical channel — rank=32,
sigma=16, lambda=10, train seed 0 (results/end_to_end_surgical.json arm D
best; the task brief's "sigma=8" is the hard cell's winner and was corrected
during pre-registration).

Threat model: the attacker can query the frozen deployed model N times per
input and observes N INDEPENDENT draws of the exposed representation
h' = h + sigma * (z @ Q^T). It averages them and runs the standard Tier-1
suite (XGB + MLP-256-256 + rank-32 ReLU LoRA) on the averaged matrix, same
75/25 attribute-stratified protocol and probe seeds {0, 1} as the main
experiments. N in {1, 2, 4, 8, 16}, nested draws (the N=2 average contains
the N=1 draw, etc.) so the sequence is a coupled monotone path.

N=1 uses the model's own exposed rep P verbatim (train_e2e_surgical's
return_clean=True captures consume no RNG, so P is bit-identical to the
matrix behind the stored winner numbers XGB 0.5060 / MLP 0.5017 /
LoRA 0.4994) — the sanity gate: if the N=1 battery does not reproduce those
within SANITY_TOL, the script writes the comparison and exits nonzero
WITHOUT running N>1.

New file; no existing experiment code path is modified. Local MPS/CPU.
macOS OpenMP guard: xgboost before torch.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import load_cell, probe_lora, probe_sklearn  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
CELL = dict(dataset="hmda", attr="race", task="loan_decision")
RANK, SIGMA, LAM = 32, 16.0, 10.0     # verified operating point (see docstring)
TRAIN_SEED = 0
PROBE_SEEDS = [0, 1]                  # the main experiments' probe seeds
NS = [1, 2, 4, 8, 16]
DRAW_SEED = 20260723                  # attack-draw RNG (draws 2..16)
SANITY_TOL = 0.01                     # ~seed-noise scale on these AUCs


def battery_per_seed(A, attr, n_attr, device):
    """Tier-1 suite with per-probe-seed AUCs (single-seed calls, mean is exact)."""
    per = {"xgb": [], "mlp": [], "lora": []}
    for s in PROBE_SEEDS:
        x, _ = probe_sklearn("XGB", A, attr, n_attr, [s])
        m, _ = probe_sklearn("MLP", A, attr, n_attr, [s])
        l, _ = probe_lora(A, attr, n_attr, [s], device)
        per["xgb"].append(float(x))
        per["mlp"].append(float(m))
        per["lora"].append(float(l))
        print(f"      [probe seed {s}] XGB={x:.4f} MLP={m:.4f} LoRA={l:.4f}",
              flush=True)
    out = {"per_seed": per}
    for k, v in per.items():
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
    out["tier1_max_mean"] = float(max(out["xgb_mean"], out["mlp_mean"],
                                      out["lora_mean"]))
    return out


def main():
    t0 = time.time()
    device = get_device()
    print(f"[setup] device={device} cell={CELL} r={RANK} sigma={SIGMA:g} "
          f"lam={LAM:g} train_seed={TRAIN_SEED} probe_seeds={PROBE_SEEDS} "
          f"Ns={NS}", flush=True)

    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        CELL["dataset"], CELL["attr"], CELL["task"])
    X_t = torch.from_numpy(X).to(device)

    print("[train] e2e surgical (frozen after this — one model, all N)", flush=True)
    t = time.time()
    P, L, h, Q = train_e2e_surgical(
        X_t, attr, task, n_attr, n_task, rank=RANK, sigma=SIGMA, lam=LAM,
        device=device, seed=TRAIN_SEED, return_clean=True)
    print(f"[train] done in {time.time() - t:.0f}s  P{P.shape} h{h.shape} "
          f"Q{Q.shape}", flush=True)

    # stored single-draw anchors for the sanity gate (full precision, from the
    # two-tier JSON's seed-0 winner row — same P, same probe seeds {0,1})
    tt = json.loads((RESULTS_DIR / "two_tier_certification.json").read_text())
    win = next(c for c in tt["e2e_surgical_winners"]
               if c["cell"] == "easy hmda/race/loan_decision")["rows"][0]
    stored = {k: win[k] for k in ("xgb", "mlp", "lora")}

    # nested attack draws 2..max(N): fresh z per draw, dedicated generator
    g = torch.Generator(device="cpu").manual_seed(DRAW_SEED)
    n_rows, d = h.shape
    h64, Q64 = h.astype(np.float64), Q.astype(np.float64)
    running = P.astype(np.float64).copy()      # sum of draws so far (draw 1 = P)
    n_in_sum = 1

    rows = []
    sanity = None
    for N in NS:
        while n_in_sum < N:
            z = torch.randn(n_rows, RANK, generator=g).numpy().astype(np.float64)
            running += h64 + SIGMA * (z @ Q64.T)
            n_in_sum += 1
        A = P if N == 1 else (running / N).astype(np.float32)
        t = time.time()
        print(f"  [N={N}] sigma_eff={SIGMA / np.sqrt(N):.2f} "
              f"({'exposed P verbatim' if N == 1 else f'average of {N} draws'})",
              flush=True)
        b = battery_per_seed(A, attr, n_attr, device)
        b.update({"N": N, "sigma_eff": float(SIGMA / np.sqrt(N)),
                  "seconds": round(time.time() - t, 1)})
        rows.append(b)
        print(f"    N={N:>2} XGB={b['xgb_mean']:.4f} MLP={b['mlp_mean']:.4f} "
              f"LoRA={b['lora_mean']:.4f} | T1={b['tier1_max_mean']:.4f} "
              f"{'BREACHES' if b['tier1_max_mean'] > BAR else 'holds'} "
              f"({b['seconds']:.0f}s)", flush=True)

        if N == 1:
            deltas = {k: b[f"{k}_mean"] - stored[k] for k in stored}
            ok = all(abs(v) <= SANITY_TOL for v in deltas.values())
            sanity = {"stored_winner_row": stored,
                      "measured_n1": {k: b[f"{k}_mean"] for k in stored},
                      "delta": deltas, "tol": SANITY_TOL, "pass": bool(ok)}
            print("  [sanity N=1 vs stored winner row]", flush=True)
            for k in stored:
                print(f"    {k}: stored={stored[k]:.6f} "
                      f"measured={b[f'{k}_mean']:.6f} delta={deltas[k]:+.6f}",
                      flush=True)
            if not ok:
                out = {"experiment": "averaging_attack", "status": "SANITY_FAIL",
                       "cell": "hmda/race/loan_decision",
                       "config": {"rank": RANK, "sigma": SIGMA, "lam": LAM,
                                  "train_seed": TRAIN_SEED,
                                  "probe_seeds": PROBE_SEEDS,
                                  "draw_seed": DRAW_SEED},
                       "sanity": sanity, "rows": rows,
                       "minutes": round((time.time() - t0) / 60, 1)}
                (RESULTS_DIR / "averaging_attack.json").write_text(
                    json.dumps(out, indent=2))
                print("[STOP] N=1 does not reproduce the stored single-draw "
                      "numbers within tolerance — stopping for review "
                      "(N>1 not run).", flush=True)
                sys.exit(1)
            print("  [sanity] PASS — proceeding to N>1", flush=True)

    print("\n" + "=" * 84)
    print("AVERAGING ATTACK — hmda/race/loan_decision, e2e surgical "
          f"r={RANK} sigma={SIGMA:g} lam={LAM:g} (bar {BAR})")
    print("=" * 84)
    print(f"{'N':>3} {'sigma_eff':>9} {'XGB':>8} {'MLP':>8} {'LoRA':>8} "
          f"{'T1 max':>8}  verdict")
    for b in rows:
        print(f"{b['N']:>3} {b['sigma_eff']:>9.2f} {b['xgb_mean']:>8.4f} "
              f"{b['mlp_mean']:>8.4f} {b['lora_mean']:>8.4f} "
              f"{b['tier1_max_mean']:>8.4f}  "
              f"{'BREACHES' if b['tier1_max_mean'] > BAR else 'holds'}")

    out = {"experiment": "averaging_attack", "status": "COMPLETE",
           "cell": "hmda/race/loan_decision",
           "mechanism": "e2e surgical (isolate-then-noise, Exp-10 arm D winner)",
           "config": {"rank": RANK, "sigma": SIGMA, "lam": LAM,
                      "train_seed": TRAIN_SEED, "probe_seeds": PROBE_SEEDS,
                      "draw_seed": DRAW_SEED, "bar": BAR,
                      "draws_nested": True},
           "prediction_file": "docs/averaging_attack_prediction.md",
           "sanity": sanity, "rows": rows,
           "minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS_DIR / "averaging_attack.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] {RESULTS_DIR / 'averaging_attack.json'} "
          f"({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
