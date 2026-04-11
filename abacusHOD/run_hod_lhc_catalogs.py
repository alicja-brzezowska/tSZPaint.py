"""
Generate AbacusHOD LRG catalogs for one HOD LHC parameter point and save to disk.

Called as:  python run_hod_lhc_catalogs.py --hod-idx N   (N = 0 … 199)

Reads 5-param LHC samples from ACT_forward_model/hod_lhc_samples.txt:
    log_Mcut, log_M1, sigma, alpha, kappa

Runs AbacusHOD for each of the 5 redshift snapshots and writes catalogs to:
    hod_mocks/lhc/{N:03d}/lightcone_halos/z{z}/galaxies_rsd/LRGs.dat

which is exactly what run_stacking.py --hod-idx N expects.
"""

import argparse
import copy
import sys
import yaml
import asdf as _asdf
import abacusnbody.hod.abacus_hod as _hod_module
from abacusnbody.hod.abacus_hod import AbacusHOD
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tszpaint.logging import setup_logging

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
CONFIG_PATH  = HERE / "config.yaml"
LHC_PATH     = HERE.parent / "ACT_forward_model" / "hod_lhc_samples.txt"
HOD_MOCKS_ROOT = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/hod_mocks")

Z_SNAPSHOTS = [0.503, 0.542, 0.582, 0.625, 0.671]
PARAM_NAMES = ["log_Mcut", "log_M1", "sigma", "alpha", "kappa"]
# ──────────────────────────────────────────────────────────────────────────────

# Monkey-patch asdf so old LCOrigins header key is transparently aliased
_original_open = _asdf.open
def _patched_open(filename, *args, **kwargs):
    af = _original_open(filename, *args, **kwargs)
    if "header" in af:
        h = af["header"]
        if "LightConeOrigins" not in h and "LCOrigins" in h:
            h["LightConeOrigins"] = h["LCOrigins"]
    return af
_asdf.open = _patched_open
_hod_module.asdf = _asdf


def run_one_snapshot(sim_params: dict, hod_params: dict, z: float, out_dir: Path):
    """Run AbacusHOD for a single redshift and save catalog to out_dir."""
    sp = copy.deepcopy(sim_params)
    sp["z_mock"]    = z
    sp["output_dir"] = str(out_dir)   # AbacusHOD writes to output_dir/sim_name/z{z}/...

    ball = AbacusHOD(sp, hod_params)
    mock = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=True, Nthread=8)
    n_lrg = len(mock["LRG"]["x"])
    logger.info(f"  z={z:.3f}: {n_lrg:,} LRGs → {ball.mock_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hod-idx", type=int, required=True,
                        help="Row index into hod_lhc_samples.txt (0–199)")
    args = parser.parse_args()
    hod_idx = args.hod_idx

    setup_logging(f"hod_lhc_catalog_{hod_idx:03d}")

    # Load LHC sample for this index
    samples = np.loadtxt(LHC_PATH, comments="#")   # (200, 5)
    row = samples[hod_idx]
    logM_cut, logM1, sigma, alpha, kappa = row
    logger.info(
        f"HOD {hod_idx}: logM_cut={logM_cut:.4f}  logM1={logM1:.4f}  "
        f"sigma={sigma:.4f}  alpha={alpha:.4f}  kappa={kappa:.4f}"
    )

    # Check if already done (all 5 snapshots written)
    out_dir = HOD_MOCKS_ROOT / "lhc" / f"{hod_idx:03d}"
    all_done = all(
        (out_dir / "lightcone_halos" / f"z{z:.3f}" / "galaxies_rsd" / "LRGs.dat").exists()
        for z in Z_SNAPSHOTS
    )
    if all_done:
        logger.info(f"HOD {hod_idx}: all snapshots already exist, skipping.")
        return

    # Load config and override HOD params
    config     = yaml.safe_load(open(CONFIG_PATH))
    sim_params = config["sim_params"]
    hod_params = copy.deepcopy(config["HOD_params"])

    hod_params["LRG_params"]["logM_cut"] = logM_cut
    hod_params["LRG_params"]["logM1"]    = logM1
    hod_params["LRG_params"]["sigma"]    = sigma
    hod_params["LRG_params"]["alpha"]    = alpha
    hod_params["LRG_params"]["kappa"]    = kappa

    out_dir.mkdir(parents=True, exist_ok=True)

    for z in Z_SNAPSHOTS:
        catalog_path = out_dir / "lightcone_halos" / f"z{z:.3f}" / "galaxies_rsd" / "LRGs.dat"
        if catalog_path.exists():
            logger.info(f"  z={z:.3f}: already exists, skipping")
            continue
        run_one_snapshot(sim_params, hod_params, z, out_dir)

    logger.info(f"HOD {hod_idx} complete → {out_dir}")


if __name__ == "__main__":
    main()
