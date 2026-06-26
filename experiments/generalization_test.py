"""TEST B: does the cheap-durability noise-channel result generalize beyond Adult/sex?

On Adult/income_prediction/sex the Gaussian noise channel stopped the ReLU attack
at sigma~2 for ~1 point of task accuracy, because the leaked sex signal is
fragile while income is noise-robust. This script asks whether that favourable
tradeoff holds on cells where the sensitive signal may be as robust as the task
signal:

  * HMDA / underwriting (purpose 0) / attribute = race (5-class), task = loan_decision
  * Diabetes / quality_research (purpose 1) / attribute = age_bucket (10-class),
    task = readmission_outcome

It reuses run_noise_sweep (and therefore the same FreshLoRAAttacker + PCRL
certificate instruments) from noise_channel_test.py verbatim, and the generic
multi-dataset loaders in utils/pcrl_io.py. Runs locally on MPS/CPU; never AWS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.noise_channel_test import run_noise_sweep, BATCH_SIZE, TAU  # noqa: E402
from utils.pcrl_io import (  # noqa: E402
    build_train_loader,
    extract_attr,
    extract_frozen_reps,
    extract_task_label,
    get_device,
    load_encoder,
)

# (dataset, purpose_idx, purpose_name, sensitive_attr, task, results_stem)
CELLS = [
    ("hmda", 0, "underwriting", "race", "loan_decision", "generalization_hmda"),
    ("diabetes", 1, "quality_research", "age_bucket", "readmission_outcome",
     "generalization_diabetes"),
]


def main() -> None:
    device = get_device()
    print(f"[setup] device={device}")
    summary = []
    for dataset, p_idx, p_name, attr, task, stem in CELLS:
        purposes, train_ds, loader = build_train_loader(dataset, batch_size=BATCH_SIZE)
        input_dim = train_ds.info.num_features
        encoder, lora_target = load_encoder(dataset, input_dim, len(purposes))
        H = extract_frozen_reps(encoder, loader, p_idx, device)
        y = extract_attr(loader, attr)
        task_y = extract_task_label(loader, task)
        majority = float(np.bincount(task_y).max() / len(task_y))
        cell_name = f"{dataset} / {p_name} / {attr}"
        print(f"\n[setup] {cell_name}: input_dim={input_dim} lora_target={lora_target!r}")
        rows = run_noise_sweep(
            H, y, task_y, device,
            cell_name=cell_name, results_stem=stem,
            attr_name=attr, task_name=task,
        )
        summary.append((cell_name, rows, majority))

    # Verdict per cell. Utility cost is measured as LIFT OVER MAJORITY retained,
    # not absolute accuracy: these tasks have very high majority baselines, so an
    # absolute drop of ~2pp can mean the entire useful signal is gone.
    print("\n================== TEST B VERDICTS ==================")
    for cell_name, rows, majority in summary:
        clean = rows[0]
        cert_at_rest = clean["baseline_linear_r2"]
        clean_lift = clean["task_acc_mean"] - majority
        rest_note = (f"cert ALREADY breached at rest ({cert_at_rest:.3f}>τ); "
                     if cert_at_rest > TAU else "")
        if clean_lift < 0.01:
            print(f"- {cell_name}: {rest_note}DEGENERATE — the linear task head barely beats "
                  f"majority (lift {clean_lift:+.3f}); noise looks 'free' only because there is "
                  f"no task utility to lose. Not a meaningful tradeoff test.")
            continue
        stopped = next((r for r in rows if not r["breaches_tau"]), None)
        if stopped is None:
            print(f"- {cell_name}: {rest_note}attack NOT STOPPED in swept range "
                  f"(breaches τ at every σ up to {rows[-1]['sigma_rel']}).")
            continue
        stopped_lift = stopped["task_acc_mean"] - majority
        retained = stopped_lift / clean_lift if clean_lift > 0 else 0.0
        verdict = ("CHEAP" if retained >= 0.7 else
                   "MODERATE" if retained >= 0.4 else "EXPENSIVE")
        print(f"- {cell_name}: {rest_note}attack severe (R²={clean['attacked_linear_r2_mean']:.3f} "
              f"at σ=0), stopped only at σ_rel={stopped['sigma_rel']}; task lift over majority "
              f"{clean_lift:+.3f} → {stopped_lift:+.3f} ({retained*100:.0f}% retained) ⇒ {verdict}")


if __name__ == "__main__":
    main()
