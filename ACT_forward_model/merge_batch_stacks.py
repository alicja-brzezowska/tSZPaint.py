"""
Merge batched partial stacking outputs into per-HOD hod_NNN.npz files.

After running stacking in batches (--ymap-start/--ymap-end), each HOD index
has files like hod_000_b0.npz, hod_000_b40.npz, hod_000_b80.npz.
This script combines them into hod_000.npz in hod_lhc/, sorted by tSZ params.

Usage:
    python merge_batch_stacks.py
"""

from pathlib import Path
import re
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tszpaint.logging import setup_logging

BATCH_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc_batches")
OUT_DIR   = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc")

HOD_RE = re.compile(r"hod_(\d+)_b\d+\.npz")


def main():
    setup_logging("merge_batch_stacks")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(BATCH_DIR.glob("hod_*_b*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No batch files found in {BATCH_DIR}")

    # Group by HOD index
    by_hod: dict[int, list[Path]] = {}
    for f in all_files:
        m = HOD_RE.match(f.name)
        if m:
            idx = int(m.group(1))
            by_hod.setdefault(idx, []).append(f)

    logger.info(f"Found {len(all_files)} batch files for {len(by_hod)} HOD indices")

    for hod_idx in sorted(by_hod):
        parts = sorted(by_hod[hod_idx])
        all_params, all_profiles, all_errors = [], [], []
        hod_params = param_names = hod_param_names = apertures = None

        for f in parts:
            d = np.load(f, allow_pickle=True)
            all_params.append(d["params"])
            all_profiles.append(d["profiles"])
            all_errors.append(d["errors"])
            if hod_params is None and "hod_params" in d:
                hod_params      = d["hod_params"]
                param_names     = d["param_names"]
                hod_param_names = d["param_names_hod"]
                apertures       = d["apertures_arcmin"]

        params   = np.concatenate(all_params,   axis=0)
        profiles = np.concatenate(all_profiles, axis=0)
        errors   = np.concatenate(all_errors,   axis=0)

        # Sort by tSZ params (consistent ordering)
        order = np.lexsort(params.T[::-1])
        params   = params[order]
        profiles = profiles[order]
        errors   = errors[order]

        out = OUT_DIR / f"hod_{hod_idx:03d}.npz"
        save_kwargs = dict(
            params           = params,
            param_names      = param_names,
            profiles         = profiles,
            errors           = errors,
            apertures_arcmin = apertures,
        )
        if hod_params is not None:
            save_kwargs["hod_params"]      = hod_params
            save_kwargs["param_names_hod"] = hod_param_names

        np.savez(out, **save_kwargs)
        logger.info(f"hod_{hod_idx:03d}: merged {len(parts)} batches → {profiles.shape[0]} profiles → {out.name}")

    logger.info(f"Done. {len(by_hod)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
