"""
For each of the 200 HOD LHC points (logM_cut, logM1, sigma): run AbacusHOD to
generate LRG positions, then stack the fixed tSZ y-map to build training data
for the HOD GP emulator.

  python run_hod_lhc_stacking.py --preconvolve # beam-convolve y-map once, save to disk
  python run_hod_lhc_stacking.py               # run all 200 points
  python run_hod_lhc_stacking.py --idx 0       # run a single index (0-based)
  python run_hod_lhc_stacking.py --resume      # skip already-done rows in output
  python run_hod_lhc_stacking.py --nproc 4     # parallel (each ~11 GB)
"""

import argparse
import copy
import os
import sys
import yaml
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import asdf as _asdf
from astropy.table import Table, vstack
from loguru import logger

# ── local imports ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parent / "ACT_forward_model"))

from tszpaint.logging import setup_logging
from tszpaint.paint.abacus_loader import obtain_healcount_edges
from ACT_data_match import stack_profiles, load_ymap, convolve_beam, cap_filter, APERTURES

# ── paths ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = HERE / "config.yaml"
LHC_PATH      = HERE / "hod_lhc_samples.txt"
OUT_PATH      = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc_stacked.npz")
PER_IDX_DIR   = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc_per_idx")

YMAP_FILE = (
    "/home/ab2927/rds/hpc-work/tSZPaint_data/Step0677-0682/"
    "alpha=0.7824150337281709_beta0=5.739888474752271_"
    "gamma=-0.05024090779728241_log10P0=1.7055429724569904.asdf"
)

HEALCOUNTS_DIR = Path("/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/total/heal-counts")
SUMMED_STEPS = [
    "Step0641-0646", "Step0647-0652", "Step0653-0658",
    "Step0659-0664", "Step0665-0670", "Step0671-0676", "Step0677-0682",
]

HOD_MOCK_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/hod_mocks/lightcone_halos")
LRG_SNAPSHOTS = ["z0.503", "z0.542", "z0.582"]

PARAM_NAMES = ["logM_cut", "logM1", "sigma"]

CONVOLVED_YMAP_PATH = Path(
    "/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/ymap_beam_convolved.npy"
)


_CHI_MIN: float = 0.0
_CHI_MAX: float = 1e9
_YMAP_CONVOLVED: np.ndarray = None
_NSIDE: int = 0



def get_chi_range() -> tuple[float, float]:
    chi_mins, chi_maxs = [], []
    for step in SUMMED_STEPS:
        hc_file = HEALCOUNTS_DIR / f"LightCone0_total_heal-counts_{step}.asdf"
        lo, hi = obtain_healcount_edges(hc_file)
        chi_mins.append(lo)
        chi_maxs.append(hi)
    return min(chi_mins), max(chi_maxs)


def _lrgs_to_angles(table: Table):
    """Return (theta, phi) in radians for LRGs in *table*, filtered by chi."""
    x   = np.asarray(table["x"])
    y   = np.asarray(table["y"])
    z   = np.asarray(table["z"])
    chi = np.sqrt(x**2 + y**2 + z**2)
    mask = (chi >= _CHI_MIN) & (chi <= _CHI_MAX)
    kept = table[mask]
    r     = np.sqrt(np.asarray(kept["x"])**2 + np.asarray(kept["y"])**2 + np.asarray(kept["z"])**2)
    theta = np.arccos(np.asarray(kept["z"]) / r)
    phi   = np.arctan2(np.asarray(kept["y"]), np.asarray(kept["x"])) % (2 * np.pi)
    return theta, phi


def _run_hod_one(idx: int, logM_cut: float, logM1: float, sigma: float) -> tuple:
    """
    Worker: run AbacusHOD for one parameter point, then stack the y-map.
    Returns (idx, [logM_cut, logM1, sigma], profiles, errors).
    """
    import gc
    import abacusnbody.hod.abacus_hod as _hod_module
    from abacusnbody.hod.abacus_hod import AbacusHOD

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

    config = yaml.safe_load(open(CONFIG_PATH))
    sim_params = config["sim_params"]
    hod_params = copy.deepcopy(config["HOD_params"])
    hod_params["LRG_params"]["logM_cut"] = logM_cut
    hod_params["LRG_params"]["logM1"]    = logM1
    hod_params["LRG_params"]["sigma"]    = sigma

    all_theta, all_phi = [], []
    for snap in LRG_SNAPSHOTS:
        sp = copy.deepcopy(sim_params)
        sp["z_mock"] = float(snap[1:])   # e.g. "z0.503" → 0.503
        ball = AbacusHOD(sp, hod_params)
        mock = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=False, Nthread=4)
        lrgs = mock["LRG"]
        x = np.asarray(lrgs["x"])
        y = np.asarray(lrgs["y"])
        z = np.asarray(lrgs["z"])
        chi = np.sqrt(x**2 + y**2 + z**2)
        mask = (chi >= _CHI_MIN) & (chi <= _CHI_MAX)
        if mask.sum() == 0:
            continue
        r     = np.sqrt(x[mask]**2 + y[mask]**2 + z[mask]**2)
        theta = np.arccos(z[mask] / r)
        phi   = np.arctan2(y[mask], x[mask]) % (2 * np.pi)
        all_theta.append(theta)
        all_phi.append(phi)
        del ball, mock, lrgs, x, y, z, chi, mask, r, theta, phi
        gc.collect()

    if not all_theta:
        raise RuntimeError(f"idx={idx}: no LRGs survived chi filter")

    theta_all = np.concatenate(all_theta)
    phi_all   = np.concatenate(all_phi)
    logger.info(f"  idx={idx}: {len(theta_all):,} LRGs in chi range")

    cap_values = cap_filter(_YMAP_CONVOLVED, _NSIDE, theta_all, phi_all)
    profiles = cap_values.mean(axis=0)
    errors   = cap_values.std(axis=0) / np.sqrt(len(theta_all))
    gc.collect()
    return idx, [logM_cut, logM1, sigma], profiles, errors


def _worker_init(omp_threads: int, chi_min: float, chi_max: float,
                 ymap_path: str, nside: int):
    global _CHI_MIN, _CHI_MAX, _YMAP_CONVOLVED, _NSIDE
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(omp_threads)
    _CHI_MIN = chi_min
    _CHI_MAX = chi_max
    _YMAP_CONVOLVED = np.load(ymap_path)
    _NSIDE = nside



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",          type=int, default=None, help="Run single LHC index (writes per-idx file)")
    parser.add_argument("--resume",       action="store_true",    help="Skip already-saved rows")
    parser.add_argument("--nproc",        type=int, default=1,    help="Parallel workers")
    parser.add_argument("--preconvolve",  action="store_true",    help="Beam-convolve y-map and save to disk, then exit")
    parser.add_argument("--merge",        action="store_true",    help="Merge all per-idx files into hod_lhc_stacked.npz")
    args = parser.parse_args()

    setup_logging("hod_lhc_stack")

    global _CHI_MIN, _CHI_MAX, _YMAP_CONVOLVED, _NSIDE
    _CHI_MIN, _CHI_MAX = get_chi_range()
    logger.info(f"chi range: [{_CHI_MIN:.1f}, {_CHI_MAX:.1f}] Mpc/h")


    if args.merge:
        files = sorted(PER_IDX_DIR.glob("idx_*.npz"))
        if not files:
            raise FileNotFoundError(f"No per-idx files found in {PER_IDX_DIR}")
        all_params, all_profiles, all_errors, lhc_indices = [], [], [], []
        for f in files:
            d = np.load(f)
            all_params.append(d["params"])
            all_profiles.append(d["profiles"])
            all_errors.append(d["errors"])
            lhc_indices.append(int(d["lhc_index"]))
        order = np.argsort(lhc_indices)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            OUT_PATH,
            params           = np.array(all_params)[order],
            param_names      = PARAM_NAMES,
            profiles         = np.array(all_profiles)[order],
            errors           = np.array(all_errors)[order],
            apertures_arcmin = APERTURES,
            lhc_indices      = np.array(lhc_indices)[order],
        )
        logger.info(f"Merged {len(files)} files → {OUT_PATH}")
        return


    if args.preconvolve:
        logger.info(f"Pre-convolving y-map: {YMAP_FILE}")
        ymap_raw, nside = load_ymap(YMAP_FILE)
        ymap_conv = convolve_beam(ymap_raw)
        CONVOLVED_YMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(CONVOLVED_YMAP_PATH, ymap_conv)
        logger.info(f"Saved beam-convolved map ({nside=}) → {CONVOLVED_YMAP_PATH}")
        return


    if not CONVOLVED_YMAP_PATH.exists():
        raise FileNotFoundError(
            f"Beam-convolved map not found: {CONVOLVED_YMAP_PATH}\n"
            "Run with --preconvolve first."
        )
    logger.info(f"Loading cached beam-convolved map from {CONVOLVED_YMAP_PATH}")
    import healpy as hp
    _YMAP_CONVOLVED = np.load(CONVOLVED_YMAP_PATH)
    _NSIDE = hp.get_nside(_YMAP_CONVOLVED)
    logger.info(f"Loaded convolved map, nside={_NSIDE}")

    samples = np.loadtxt(LHC_PATH)   # (125, 3): logM_cut, logM1, sigma
    logger.info(f"Loaded {len(samples)} LHC points from {LHC_PATH}")

    done_results = {}   # idx → (params, profiles, errors)
    if args.resume and OUT_PATH.exists():
        d = np.load(OUT_PATH, allow_pickle=True)
        saved_params = d["params"]          # (K, 3)
        saved_profiles = d["profiles"]      # (K, N_ap)
        saved_errors   = d["errors"]        # (K, N_ap)
        saved_idx      = d["lhc_indices"]   # (K,)
        for i, idx in enumerate(saved_idx):
            done_results[int(idx)] = (
                saved_params[i].tolist(), saved_profiles[i], saved_errors[i]
            )
        logger.info(f"Resuming: {len(done_results)} rows already done")

    if args.idx is not None:
        todo = [args.idx]
    else:
        todo = [i for i in range(len(samples)) if i not in done_results]
    logger.info(f"Running {len(todo)} point(s) with {args.nproc} worker(s)")

    if not todo:
        logger.info("Nothing to do.")
        return

    new_results = {}   
    omp_per_worker = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", args.nproc)) // args.nproc)
    logger.info(f"OMP threads per worker: {omp_per_worker}")

    with ProcessPoolExecutor(
        max_workers=args.nproc,
        initializer=_worker_init,
        initargs=(omp_per_worker, _CHI_MIN, _CHI_MAX,
                  str(CONVOLVED_YMAP_PATH), _NSIDE),
    ) as ex:
        futures = {
            ex.submit(_run_hod_one, i, *samples[i]): i
            for i in todo
        }
        for k, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                idx, params, profiles, errors = fut.result()
                new_results[idx] = (params, profiles, errors)
                logger.info(f"[{k}/{len(todo)}] idx={idx} done")
            except Exception as e:
                logger.error(f"[{k}/{len(todo)}] idx={i} FAILED: {e}")


    if args.idx is not None:
        idx = list(new_results.keys())[0]
        params, profiles, errors = new_results[idx]
        PER_IDX_DIR.mkdir(parents=True, exist_ok=True)
        out = PER_IDX_DIR / f"idx_{idx:04d}.npz"
        np.savez(out, params=params, profiles=profiles, errors=errors,
                 apertures_arcmin=APERTURES, lhc_index=idx)
        logger.info(f"Saved idx={idx} → {out}")
    else:
        all_results = {**done_results, **new_results}
        sorted_idx  = sorted(all_results)
        all_params   = np.array([all_results[i][0] for i in sorted_idx])
        all_profiles = np.array([all_results[i][1] for i in sorted_idx])
        all_errors   = np.array([all_results[i][2] for i in sorted_idx])
        lhc_indices  = np.array(sorted_idx)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            OUT_PATH,
            params           = all_params,
            param_names      = PARAM_NAMES,
            profiles         = all_profiles,
            errors           = all_errors,
            apertures_arcmin = APERTURES,
            lhc_indices      = lhc_indices,
        )
        logger.info(f"Saved {len(sorted_idx)} profiles → {OUT_PATH}")


if __name__ == "__main__":
    main()
