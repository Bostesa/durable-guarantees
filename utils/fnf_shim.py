"""Environment shim to run the OFFICIAL eth-sri/fnf (2022-era, CUDA-assuming)
unmodified under modern numpy 2.5 / torch 2.12 / Python 3.13 on Apple silicon.

The FNF source tree is NOT edited. Four environment-level aliases, each an
identity operation semantically:

  1. numpy.object / numpy.int / numpy.float / numpy.bool / numpy.str -> the
     builtins of the same name. These attributes WERE exactly those builtins
     (deprecated in numpy 1.20, removed in 1.24). Restored only where numpy no
     longer defines the name, so numpy 2.x's own np.bool / np.str_ are left
     alone. Used by datasets/*.py.
  2. pandas.DataFrame.applymap -> DataFrame.map, which pandas 3.0 renamed it
     to. Same elementwise apply, same semantics.
  3. torch.load(..., weights_only=False) as the default. torch >= 2.6 flipped
     this default to True. FNF saves and reloads whole nn.Module pickles
     (e.g. `adult/made1.pt`) that it wrote itself moments earlier in the same
     pipeline. No third-party checkpoint is ever loaded.
  4. Module-level `device = 'cuda'` globals (train_fnf, *_flow_multi, toy_*)
     rebound to the device given by FNF_DEVICE (default 'cpu'), because this
     machine has no CUDA. Their scripts already expose --device for the same
     purpose. The module globals were simply missed by that flag.

Nothing else is patched: no optimizer, loss, architecture, schedule or metric
is touched. If a run needs more than the four aliases above, that is a GATE
FAILURE to be reported, not a shim to be extended.

Usage:
    import utils.fnf_shim as fnf_shim
    fnf_shim.install(device="cpu")          # before importing any fnf module
    fnf_shim.add_fnf_to_path()              # honours $FNF_ROOT
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Location of the official eth-sri/fnf checkout. Override with $FNF_ROOT.
# The default assumes it sits beside this repository as ../fnf.
FNF_ROOT_DEFAULT = str(Path(__file__).resolve().parents[1].parent / "fnf")
_installed = False


def install(device: str | None = None) -> str:
    """Apply the four aliases. Returns the device string that was installed."""
    global _installed
    dev = device or os.environ.get("FNF_DEVICE", "cpu")

    import numpy as np
    # Restore ONLY the aliases numpy actually removed. numpy 2.x reintroduced
    # `np.bool` as its own boolean scalar (np.bool_), and clobbering that with
    # the builtin breaks numpy.ma internals, so never overwrite what exists.
    for _name, _builtin in (("object", object), ("int", int), ("float", float),
                            ("bool", bool), ("str", str)):
        try:
            getattr(np, _name)
        except AttributeError:
            setattr(np, _name, _builtin)

    # pandas renamed DataFrame.applymap -> DataFrame.map (deprecated 2.1,
    # removed 3.0). Same function, same semantics, elementwise apply.
    import pandas as pd
    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map

    import torch
    if not getattr(torch.load, "_fnf_shimmed", False):
        _orig_load = torch.load

        def _load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)

        _load._fnf_shimmed = True
        torch.load = _load

    os.environ["FNF_DEVICE"] = dev
    _installed = True
    return dev


def add_fnf_to_path(root: str | None = None) -> Path:
    """Put the FNF checkout on sys.path (their code uses top-level imports and
    expects cwd == repo root for its relative data/model paths).

    Resolution order: explicit `root` argument, then $FNF_ROOT, then ../fnf
    beside this repository.
    """
    p = Path(root or os.environ.get("FNF_ROOT", FNF_ROOT_DEFAULT)).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"FNF checkout not found at {p}. Set $FNF_ROOT to the official "
            "eth-sri/fnf clone, or pass root= explicitly.")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def retarget_device(module_names=("train_fnf",), device: str | None = None) -> None:
    """Rebind the module-level `device = 'cuda'` globals after import."""
    dev = device or os.environ.get("FNF_DEVICE", "cpu")
    for name in module_names:
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "device", None) is not None:
            mod.device = dev


if __name__ == "__main__":
    # Launcher for running FNF's own scripts unmodified under the period venv:
    #   python utils/fnf_shim.py train_enc_categorical.py [their args...]
    # cwd must be the FNF repo root (their relative data/model paths need it).
    # Under fnf/.venv (numpy 1.23.5 / pandas 1.5.3) the numpy and pandas
    # aliases are already present, so install() reduces to the torch.load
    # default, needed because their pipeline reloads the MADE prior pickle it
    # wrote itself minutes earlier in the same run.
    import runpy

    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.getcwd())
    runpy.run_path(script, run_name="__main__")
