"""Experiment 13 (step 1) — extract CelebA representations through PCRL's vision encoder.

The full CelebA run (Experiment 13) needs the frozen per-purpose representations of
every CelebA image under PCRL's trained vision encoder. This script does exactly and
only that long forward pass (~202k images), checkpointed so it resumes if interrupted.

What is extracted, and with what:
  * Model: PCRL's CelebA v2 encoder — `pcrl.models.cnn_encoder.CNNEncoder`
    (FiLM purpose conditioning, conv (32,64,128), repr_dim=128, 5 purposes), loaded
    from `checkpoints/celeba_v2/final.pt` ("encoder" state dict). eval() mode.
  * Data: `pcrl.data.celeba.CelebADataset` — PCRL's own loader (64x64, official
    partitions). For extraction the TRAIN-split transform is replaced by the eval
    transform (no RandomHorizontalFlip) so representations are deterministic.
  * Purposes: the two cells picked by the Experiment-12 coupling scan share
    protected=Young and are both existing PCRL purposes:
      - purpose 0 `smile_detection`            (task Smiling,    predictor 0.522)
      - purpose 3 `attractiveness_prediction`  (task Attractive, predictor 0.734)
    Both purpose representations are computed in the SAME pass over the images
    (image decode dominates the cost, the encoder is cheap).
  * Labels stored alongside: Smiling, Attractive, Young, Male.

Checkpointing: every CHUNK batches, a chunk .npz is written to
`data_cache/celeba/<split>_chunk*.npz` (gitignored — never committed). On restart,
complete chunks are skipped (shuffle=False, so chunks are contiguous index ranges).
When a split completes, chunks are consolidated into `<split>.npz` and removed.

Local MPS/CPU; never AWS.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.pcrl_io import PCRL_ROOT, get_device  # noqa: E402  (adds PCRL to sys.path)

from pcrl.data.celeba import CelebADataset, _get_transform, get_celeba_purposes  # noqa: E402
from pcrl.models.cnn_encoder import CNNEncoder  # noqa: E402

CACHE_DIR = REPO_ROOT / "data_cache" / "celeba"
CKPT = PCRL_ROOT / "checkpoints" / "celeba_v2" / "final.pt"
PURPOSES = {"smile_detection": 0, "attractiveness_prediction": 3}
LABELS = ["Smiling", "Attractive", "Young", "Male"]
SPLITS = ["train", "val", "test"]
BATCH = 256
CHUNK = 100          # batches per checkpoint chunk (256*100 = 25.6k rows)
NUM_WORKERS = 6


def load_encoder(device):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc = CNNEncoder(repr_dim=128, num_purposes=5, purpose_emb_dim=32,
                     conv_channels=(32, 64, 128), dropout=0.3)
    enc.load_state_dict(ck["encoder"])
    enc.eval().to(device)
    return enc


def extract_split(split, enc, device):
    out_path = CACHE_DIR / f"{split}.npz"
    if out_path.exists():
        print(f"[{split}] already complete: {out_path}", flush=True)
        return

    ds = CelebADataset(purposes=get_celeba_purposes(),
                       root=str(PCRL_ROOT / "data" / "celeba"), split=split)
    ds.img_transform = _get_transform("test")   # deterministic (no random flip)
    n = len(ds)
    rows_per_chunk = BATCH * CHUNK
    n_chunks = (n + rows_per_chunk - 1) // rows_per_chunk

    done = sorted(CACHE_DIR.glob(f"{split}_chunk[0-9][0-9][0-9][0-9].npz"))
    done_ids = {int(p.stem.split("chunk")[1]) for p in done}
    print(f"[{split}] {n} images, {n_chunks} chunks, {len(done_ids)} already done",
          flush=True)

    for ci in range(n_chunks):
        if ci in done_ids:
            continue
        lo, hi = ci * rows_per_chunk, min((ci + 1) * rows_per_chunk, n)
        loader = DataLoader(Subset(ds, range(lo, hi)), batch_size=BATCH,
                            shuffle=False, num_workers=NUM_WORKERS)
        reps = {p: [] for p in PURPOSES}
        labs = {l: [] for l in LABELS}
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                x = batch["features"].to(device)
                for pname, pidx in PURPOSES.items():
                    reps[pname].append(enc(x, pidx).cpu().numpy())
                for l in LABELS:
                    src = (batch["task_labels"] if l in batch["task_labels"]
                           else batch["sensitive_attrs"])
                    labs[l].append(src[l].numpy())
        payload = {f"rep_{p}": np.concatenate(v).astype(np.float32)
                   for p, v in reps.items()}
        payload.update({f"label_{l}": np.concatenate(v).astype(np.int64)
                        for l, v in labs.items()})
        tmp = CACHE_DIR / f"{split}_chunk{ci:04d}.tmp.npz"
        np.savez(tmp, **payload)
        tmp.rename(CACHE_DIR / f"{split}_chunk{ci:04d}.npz")
        print(f"[{split}] chunk {ci + 1}/{n_chunks} rows {lo}:{hi} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # consolidate
    chunks = sorted(CACHE_DIR.glob(f"{split}_chunk[0-9][0-9][0-9][0-9].npz"))
    assert len(chunks) == n_chunks, f"{split}: {len(chunks)} chunks != {n_chunks}"
    parts = [np.load(p) for p in chunks]
    merged = {k: np.concatenate([pt[k] for pt in parts]) for k in parts[0].files}
    assert len(next(iter(merged.values()))) == n
    tmp = CACHE_DIR / f"{split}.tmp.npz"
    np.savez(tmp, **merged)
    tmp.rename(out_path)
    for p in chunks:
        p.unlink()
    print(f"[{split}] consolidated -> {out_path} "
          f"({[(k, v.shape) for k, v in merged.items()]})", flush=True)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[setup] device={device} ckpt={CKPT}", flush=True)
    enc = load_encoder(device)
    for split in SPLITS:
        extract_split(split, enc, device)
    print("[done] all splits extracted", flush=True)


if __name__ == "__main__":
    main()
