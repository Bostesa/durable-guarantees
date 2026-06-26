"""PCRL glue: load the audited PCRL Adult model, data, and certificate instruments.

This module is the ONLY place that knows about PCRL internals. It adds the
(read-only) PCRL repo to sys.path and imports/adapts its code. We never modify
the PCRL repo — we mirror exactly the patterns the audit identified:

  * encoder build + checkpoint load (scripts/crosspurp/run_eval_multi.py::load_encoder),
    including the lora_target auto-detection (_detect_lora_target).
  * the Adult train-loader pattern (run_eval_multi.py::build_loaders("adult")).
  * the R^2 certificate instruments themselves
    (pcrl/vision/r2_helper.py::linear_r2 and
     pcrl/evaluation/certificates.py::compute_dominant_axis_r2).

The point of the falsification experiment is to measure with PCRL's OWN
certificate code, so we import those functions rather than reimplement them.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Locate the (read-only) PCRL repo -------------------------------------
PCRL_ROOT = Path(os.environ.get("PCRL_ROOT", "/Users/nathansamson/PCRL")).resolve()
if not PCRL_ROOT.exists():
    raise FileNotFoundError(
        f"PCRL repo not found at {PCRL_ROOT}. Set PCRL_ROOT env var to the repo root."
    )
if str(PCRL_ROOT) not in sys.path:
    sys.path.insert(0, str(PCRL_ROOT))

# --- Import PCRL's own code (do NOT reimplement) ---------------------------
from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.lora import LoRAAdapter, PerPurposeLoRAEncoder  # noqa: E402

# PCRL's certificate instruments — the measuring devices for this experiment.
from pcrl.vision.r2_helper import linear_r2  # noqa: E402,F401  (re-exported)
from pcrl.evaluation.certificates import compute_dominant_axis_r2  # noqa: E402,F401

CHECKPOINT = PCRL_ROOT / "checkpoints" / "v2_adult_s0" / "final.pt"
INCOME_PREDICTION_PURPOSE_IDX = 0

# Adult LoRA config from the audit (matches DATASET_CONFIG["adult"]).
ADULT_LORA_RANK = 8
ADULT_LORA_ALPHA = 16.0


def get_device() -> torch.device:
    """Prefer MPS (Apple Silicon), else CPU. Never CUDA/AWS here."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Encoder build + load (mirrors run_eval_multi.py::load_encoder) --------
def _detect_lora_target(lora_adapters_sd: dict) -> str:
    """Infer lora_target from a checkpoint's lora_adapters state_dict.

    Copied verbatim in behaviour from scripts/crosspurp/run_eval_multi.py so we
    don't hardcode the adapter layout. Keys look like
    ``<purpose_idx>.<linear_idx>.{A.weight,B.weight,bias}``. A linear_idx of 1
    or 2 means all three backbone Linears were adapted ("all_linear");
    otherwise only the final repr_proj was ("repr_proj_only").
    """
    for key in lora_adapters_sd.keys():
        if re.match(r"^\d+\.([12])\.", key) is not None:
            return "all_linear"
    return "repr_proj_only"


def load_adult_encoder(
    ckpt_path: Path = CHECKPOINT,
    input_dim: int = 105,
    n_purposes: int = 3,
) -> torch.nn.Module:
    """Build StandardEncoder + PerPurposeLoRAEncoder and load the s0 checkpoint.

    Uses weights_only=False (audit fact) and the lora_target auto-detection
    rather than hardcoding the adapter layout. Returns an eval()-mode encoder.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lora_target = _detect_lora_target(ckpt["lora_adapters"])

    backbone = StandardEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=64,
        dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone,
        n_purposes=n_purposes,
        rank=ADULT_LORA_RANK,
        alpha=ADULT_LORA_ALPHA,
        dropout=0.0,
        lora_target=lora_target,
    )
    encoder.backbone.load_state_dict(ckpt["backbone"], strict=False)
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])

    # Restore any frozen LEACE projections that were checkpointed.
    enc_buf = ckpt.get("encoder_buffers", {}) or {}
    for p_idx in range(n_purposes):
        P_key, mu_key = f"leace_P_p{p_idx}", f"leace_mu_p{p_idx}"
        if P_key in enc_buf and mu_key in enc_buf:
            encoder.set_leace_projection(p_idx, enc_buf[P_key], enc_buf[mu_key])

    encoder.eval()
    return encoder, lora_target


# --- Data (mirrors run_eval_multi.py::build_loaders("adult"), train split) -
def build_adult_train_loader(batch_size: int = 512):
    """Build the Adult train dataset + loader exactly like build_loaders('adult').

    Returns (purposes, train_ds, train_loader).
    """
    purposes = get_adult_purposes()
    root = str(PCRL_ROOT / "data")
    train_ds = AdultDataset(purposes=purposes, root=root, split="train", download=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pcrl_batch,
        num_workers=0,
    )
    return purposes, train_ds, train_loader


@torch.no_grad()
def extract_frozen_reps(encoder, loader, purpose_idx: int, device) -> np.ndarray:
    """Frozen 64-dim representation BEFORE the task head (what the attacker probes).

    Mirrors run_eval_multi.py::extract_reps: encoder(features, purpose_idx).
    """
    encoder.to(device)
    chunks = []
    for batch in loader:
        h = encoder(batch["features"].to(device), purpose_idx)
        chunks.append(h.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def extract_attr(loader, attr_name: str) -> np.ndarray:
    """Stack a sensitive attribute across the loader (mirrors extract_attr)."""
    out = []
    for batch in loader:
        out.append(batch["sensitive_attrs"][attr_name].numpy())
    return np.concatenate(out, axis=0).astype(np.int64)


def extract_task_label(loader, task_name: str) -> np.ndarray:
    """Stack a task label (e.g. 'income') across the loader."""
    out = []
    for batch in loader:
        out.append(batch["task_labels"][task_name].numpy())
    return np.concatenate(out, axis=0).astype(np.int64)
