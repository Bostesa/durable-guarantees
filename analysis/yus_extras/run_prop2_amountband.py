"""Amount-band correction to the Prop-2 witness table (see
amount_band_recompute_note.md, committed before this ran).

The two rows labelled "middle (HMDA/amount)" in
results/mi_ceiling.json["operating_point_extension"]["points"] were computed
from the hmda/race/loan_decision channel; mi_ceiling.json contains no
loan_amount_band channel. This trains the two missing channels
(hmda/race/loan_amount_band at sigma=12 and sigma=24, train_channel seed 0,
deterministic) and evaluates the SAME appendix-faithful witness used in
run_prop2_witness.py — the construction functions are imported from that
module rather than restated, so the two tables are computed identically.

Recomputation only. No new experimental claim.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

# importing the Task-4 module gives the identical construction + constants
from run_prop2_witness import (  # noqa: E402
    BAR, N_SAMPLE, SEED, bayes_auc, class_stats,
)

from experiments.diagnostic import load_cell  # noqa: E402
from experiments.mi_ceiling import train_channel  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

CELL = ("hmda", "race", "loan_amount_band")
SIGMAS = [(12.0, "T1"), (24.0, "T2")]
# what the mislabelled rows actually reported (loan_decision channel)
MISLABELLED = {12.0: 0.9598947468180754, 24.0: 0.8035629014523376}
# and what the appendix-faithful recompute gave on that wrong channel
WRONG_CHANNEL_FAITHFUL = {12.0: 0.5679, 24.0: 0.5302}


def main():
    t0 = time.time()
    device = get_device()
    rng = np.random.default_rng(SEED)
    ds, at, tk = CELL
    X, attr, task, n_attr, n_task, _, _ = load_cell(ds, at, tk)
    X_t = torch.from_numpy(X).to(device)
    print(f"[setup] device={device} cell={ds}/{at}/{tk} n={len(attr)} "
          f"n_attr={n_attr} n_sample={N_SAMPLE} seed={SEED}\n", flush=True)

    rows = []
    for sigma, tier in SIGMAS:
        h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sigma,
                                device, seed=0)
        stats = class_stats(h, attr, n_attr)
        total_var = float(h.var(0).astype(np.float64).sum())
        marginal = float(norm.cdf(np.sqrt(2.0 * total_var) / sigma))
        faithful, dmin = bayes_auc(stats, sigma, n_attr, rng)
        row = dict(
            label="middle (HMDA/amount)", tier=tier, sigma=sigma,
            channel=f"{ds}/{at}/{tk}", n_attr=n_attr,
            mislabelled_stored_value=MISLABELLED[sigma],
            mislabelled_source="loan_decision channel (mi_ceiling.json)",
            marginal_ceiling_correct_channel=marginal,
            appendix_faithful_auc=faithful,
            appendix_faithful_on_wrong_channel=WRONG_CHANNEL_FAITHFUL[sigma],
            total_var=total_var,
            tr_Sigma_y_mean=float(np.mean([np.trace(s["Sigma"]) for s in stats])),
            min_opposite_sheet_distance=dmin,
            dmin_over_sigma=dmin / sigma,
            exceeds_bar=bool(faithful > BAR))
        rows.append(row)
        print(f"  sigma={sigma:<5g} ({tier})  marginal={marginal:.4f}  "
              f"appendix-faithful={faithful:.4f}  "
              f"{'>0.55' if faithful > BAR else '<=0.55'}  "
              f"dmin/sigma={dmin / sigma:.3f}   "
              f"[mislabelled row said {MISLABELLED[sigma]:.4f}]", flush=True)

    out = dict(
        experiment="prop2_witness_amountband_correction",
        note_file="amount_band_recompute_note.md",
        note_commit="c993f5a",
        purpose=("recomputation correcting a channel mislabel in the "
                 "second-moment ceiling table; not a new experimental claim"),
        cell="/".join(CELL), train_channel_seed=0, n_sample=N_SAMPLE,
        seed=SEED, bar=BAR, rows=rows,
        minutes=round((time.time() - t0) / 60, 1))
    (HERE / "prop2_witness_amountband.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'prop2_witness_amountband.json'} "
          f"({out['minutes']} min)")


if __name__ == "__main__":
    main()
