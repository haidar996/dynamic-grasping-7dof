"""Physics switch for the ekf-mu mismatch evaluation (audit A1; design docs/A1_ekfmu_mismatch_eval_design.md, 2026-11-10).

Environment variable PHYSICS = matched (default) | mismatch, read by step_0i, step_0k, step_1c, step_1d, step_1e.
  matched : the original data (outputs/raw_60hz_h3_f49_sq480_with_arm) and the original output names, so every existing
            result reproduces bit for bit.
  mismatch: data outputs/raw_60hz_h3_f49_sq480_with_arm_mismatch (same p0, v0, sensor seeds, split); every file these
            scripts WRITE gets the suffix _mismatch, so matched outputs are never touched.
"""
import os
from pathlib import Path

PHYSICS = os.environ.get("PHYSICS", "matched")
assert PHYSICS in ("matched", "mismatch"), f"PHYSICS must be 'matched' or 'mismatch', got {PHYSICS!r}"
MISMATCH = PHYSICS == "mismatch"
SUFFIX = "_mismatch" if MISMATCH else ""
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RAW = OUT / f"raw_60hz_h3_f49_sq480_with_arm{SUFFIX}"


def out_path(name, suffix=None):
    """outputs/<stem><suffix>.<ext>; 'ekf_mu_eval_errors.pkl.gz' -> 'ekf_mu_eval_errors_mismatch.pkl.gz'."""
    stem, ext = name.split(".", 1)
    return OUT / f"{stem}{SUFFIX if suffix is None else suffix}.{ext}"
