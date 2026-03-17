
import os
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger
from astropy.table import Table, vstack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tszpaint.logging import setup_logging
from tszpaint.paint.abacus_loader import obtain_healcount_edges
from ACT_data_match import stack_profiles, APERTURES

YMAP_ROOT  = Path("/home/ab2927/rds/hpc-work/tSZPaint_data")
OUT_ROOT   = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles")
YMAP_DIR   = YMAP_ROOT / "Step0677-0682"

HEALCOUNTS_DIR = Path("/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/total/heal-counts")
SUMMED_STEPS = [
    "Step0641-0646",
    "Step0647-0652",
    "Step0653-0658",
    "Step0659-0664",
    "Step0665-0670",
    "Step0671-0676",
    "Step0677-0682",
]

LRG_SNAPSHOTS = [
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.503/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.542/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.582/galaxies_rsd/LRGs.dat",
]

PARAM_NAMES = ["alpha", "beta0", "gamma", "log10P0"]

PARAM_RE = re.compile(
    r"alpha=(?P<alpha>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_beta0=(?P<beta0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_gamma=(?P<gamma>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_log10P0=(?P<log10P0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
)

# Set at startup in main(), read by worker processes via fork-inherited globals
_THETA_LRG = None
_PHI_LRG   = None
_APPLY_BEAM = True


def parse_params(filename: str) -> dict:
    m = PARAM_RE.search(filename)
    if m is None:
        raise ValueError(f"Could not parse params from: {filename}")
    return {k: float(v) for k, v in m.groupdict().items()}


def get_chi_range() -> tuple[float, float]:
    chi_mins, chi_maxs = [], []
    for step in SUMMED_STEPS:
        hc_file = HEALCOUNTS_DIR / f"LightCone0_total_heal-counts_{step}.asdf"
        lo, hi = obtain_healcount_edges(hc_file)
        chi_mins.append(lo)
        chi_maxs.append(hi)
    return min(chi_mins), max(chi_maxs)


def load_and_filter_lrgs(chi_min: float, chi_max: float):
    tables = [Table.read(f, format="ascii.ecsv") for f in LRG_SNAPSHOTS]
    combined = vstack(tables)
    x = np.asarray(combined["x"])
    y = np.asarray(combined["y"])
    z = np.asarray(combined["z"])
    chi = np.sqrt(x**2 + y**2 + z**2)
    mask = (chi >= chi_min) & (chi <= chi_max)
    kept = combined[mask]
    logger.info(f"LRGs: {len(combined):,} total → {mask.sum():,} kept  (chi=[{chi_min:.1f}, {chi_max:.1f}] Mpc/h)")
    r     = np.sqrt(np.asarray(kept["x"])**2 + np.asarray(kept["y"])**2 + np.asarray(kept["z"])**2)
    theta = np.arccos(np.asarray(kept["z"]) / r)
    phi   = np.arctan2(np.asarray(kept["y"]), np.asarray(kept["x"])) % (2 * np.pi)
    return theta, phi


def _worker_init(omp_threads: int):
    """Limit OMP threads per worker so beam convolution doesn't OOM."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(omp_threads)


def _process_one(fname: str):
    """Worker: stack one y-map. Uses fork-inherited _THETA_LRG/_PHI_LRG."""
    import gc
    params = parse_params(fname)
    t0 = perf_counter()
    y_stacked, y_err = stack_profiles(
        str(YMAP_DIR / fname), _THETA_LRG, _PHI_LRG, apply_beam=_APPLY_BEAM
    )
    elapsed = perf_counter() - t0
    gc.collect()   # ensure the 3 GB y-map is released before next job
    return fname, [params[k] for k in PARAM_NAMES], y_stacked, y_err, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param",   default=None)
    parser.add_argument("--output",  default="all_steps_stacked.npz")
    parser.add_argument("--no-beam", action="store_true")
    parser.add_argument("--nproc",   type=int, default=1,
                        help="Number of parallel workers (each needs ~11 GB)")
    args = parser.parse_args()

    setup_logging("cap_stack")

    chi_min, chi_max = get_chi_range()
    logger.info(f"Summed y-map chi range: [{chi_min:.1f}, {chi_max:.1f}] Mpc/h")

    # Load LRGs once; fork-inherited by all workers at no extra memory cost
    global _THETA_LRG, _PHI_LRG, _APPLY_BEAM
    _THETA_LRG, _PHI_LRG = load_and_filter_lrgs(chi_min, chi_max)
    _APPLY_BEAM = not args.no_beam

    all_fnames = sorted(f.name for f in YMAP_DIR.glob("*.asdf"))
    if args.param:
        all_fnames = [f for f in all_fnames if args.param in f]
        if not all_fnames:
            raise ValueError(f"No file matching --param '{args.param}'")

    out_file = OUT_ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing {len(all_fnames)} maps with {args.nproc} worker(s) → {out_file}")

    results = {}   # fname → (params, y_stacked, y_err)

    omp_per_worker = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", args.nproc)) // args.nproc)
    logger.info(f"OMP threads per worker: {omp_per_worker}")
    with ProcessPoolExecutor(max_workers=args.nproc,
                             initializer=_worker_init,
                             initargs=(omp_per_worker,)) as ex:
        futures = {ex.submit(_process_one, fname): fname for fname in all_fnames}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                fname, params, y_stacked, y_err, elapsed = fut.result()
                results[fname] = (params, y_stacked, y_err)
                logger.info(f"[{i}/{len(all_fnames)}] {fname}  ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"[{i}/{len(all_fnames)}] FAILED {fname}: {e}")

    # Reconstruct in sorted order
    all_params, profiles, errors = [], [], []
    for fname in sorted(results):
        params, y_stacked, y_err = results[fname]
        all_params.append(params)
        profiles.append(y_stacked)
        errors.append(y_err)

    np.savez(
        out_file,
        params=np.array(all_params),    # (N, 4)
        param_names=PARAM_NAMES,
        profiles=np.array(profiles),     # (N, N_ap)
        errors=np.array(errors),         # (N, N_ap)
        apertures_arcmin=APERTURES,
    )
    print(f"Saved {len(profiles)} profiles → {out_file}")


if __name__ == "__main__":
    main()
