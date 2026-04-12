"""
For each of the 200 HOD LHC points: run AbacusHOD once to generate LRG positions,
then stack ALL 125 pressure-profile y-maps with that same catalog.

The galaxy catalog is generated once per HOD index and fork-inherited by map workers,
so AbacusHOD (the expensive part) runs exactly once per task.

Each SLURM array task (--idx N) produces hod_NNN.npz containing:
    pressure_params    : (125, 4)  pressure params [alpha, beta0, gamma, log10P0]
    param_names_pressure: (4,)
    profiles_cap       : (125, N_ap) CAP stacked profiles
    errors_cap         : (125, N_ap)
    apertures_cap      : (N_ap,)
    hod_params         : (5,)  HOD params [logM_cut, logM1, sigma, alpha_hod, kappa]
    param_names_hod    : (5,)

Requires preconvolved maps in YMAP_DIR_CONV (run preconvolve_maps.py first).

Usage:
    python run_hod_lhc_stacking.py --idx 0          # one HOD point, all 125 maps
    python run_hod_lhc_stacking.py --idx 0 --nproc 8
    python run_hod_lhc_stacking.py --merge           # merge per-idx files
"""

import argparse
import copy
import gc
import os
import re
import sys
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import asdf as _asdf
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parent / "ACT_forward_model"))

from tszpaint.logging import setup_logging
from tszpaint.paint.abacus_loader import obtain_healcount_edges
from tszpaint.cosmology.mass_conversion import H_ABACUS, MAX_M200M_H
from ACT_data_match import cap_filter, APERTURES_CAP

# ── paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH    = HERE / "config.yaml"
LHC_PATH       = HERE.parent / "ACT_forward_model" / "hod_lhc_samples.txt"
YMAP_DIR_CONV  = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/preconvolved_Step0677-0682")
OUT_DIR        = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc")
HEALCOUNTS_DIR = Path("/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/total/heal-counts")

SUMMED_STEPS = [
    "Step0641-0646", "Step0647-0652", "Step0653-0658",
    "Step0659-0664", "Step0665-0670", "Step0671-0676", "Step0677-0682",
]
LRG_SNAPSHOTS = [0.503, 0.542, 0.582, 0.625, 0.671]
MAX_LRG       = 4_000_000
NPROC_MAPS    = 8   # parallel workers for map stacking (one worker per map)

PARAM_NAMES_PRESSURE = ["alpha", "beta0", "gamma", "log10P0"]
PARAM_NAMES_HOD      = ["log_Mcut", "log_M1", "sigma", "alpha_hod", "kappa"]

PARAM_RE = re.compile(
    r"alpha=(?P<alpha>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_beta0=(?P<beta0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_gamma=(?P<gamma>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_log10P0=(?P<log10P0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
)

# Fork-inherited by map workers
_THETA_LRG:  np.ndarray = None
_PHI_LRG:    np.ndarray = None
_NPROC_MAPS: int        = NPROC_MAPS


def parse_pressure_params(fname: str) -> list[float]:
    m = PARAM_RE.search(fname)
    if m is None:
        raise ValueError(f"Could not parse pressure params from: {fname}")
    return [float(m.group(k)) for k in PARAM_NAMES_PRESSURE]


def get_chi_range() -> tuple[float, float]:
    chi_mins, chi_maxs = [], []
    for step in SUMMED_STEPS:
        hc_file = HEALCOUNTS_DIR / f"LightCone0_total_heal-counts_{step}.asdf"
        lo, hi = obtain_healcount_edges(hc_file)
        chi_mins.append(lo)
        chi_maxs.append(hi)
    return min(chi_mins), max(chi_maxs)


def _patch_asdf():
    import abacusnbody.hod.abacus_hod as _hod_module
    _orig = _asdf.open
    def _patched(fn, *a, **kw):
        af = _orig(fn, *a, **kw)
        if "header" in af:
            h = af["header"]
            if "LightConeOrigins" not in h and "LCOrigins" in h:
                h["LightConeOrigins"] = h["LCOrigins"]
        return af
    _asdf.open = _patched
    _hod_module.asdf = _asdf


def generate_lrgs(hod_idx: int, chi_min: float, chi_max: float,
                  logM_cut: float, logM1: float, sigma: float,
                  alpha_hod: float, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """Run AbacusHOD across all 5 snapshots, apply chi/mass filter and subsample."""
    from abacusnbody.hod.abacus_hod import AbacusHOD
    _patch_asdf()

    config     = yaml.safe_load(open(CONFIG_PATH))
    sim_params = config["sim_params"]
    hod_params = copy.deepcopy(config["HOD_params"])
    hod_params["LRG_params"].update(dict(
        logM_cut=logM_cut, logM1=logM1, sigma=sigma, alpha=alpha_hod, kappa=kappa,
    ))

    xs, ys, zs, ms = [], [], [], []
    for z_snap in LRG_SNAPSHOTS:
        sp = copy.deepcopy(sim_params)
        sp["z_mock"]    = z_snap
        sp["output_dir"] = "/tmp"  # unused — write_to_disk=False
        ball = AbacusHOD(sp, hod_params)
        mock = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=False, Nthread=8)
        lrg  = mock["LRG"]
        xs.append(np.asarray(lrg["x"],    dtype=np.float64))
        ys.append(np.asarray(lrg["y"],    dtype=np.float64))
        zs.append(np.asarray(lrg["z"],    dtype=np.float64))
        ms.append(np.asarray(lrg["mass"], dtype=np.float64))
        logger.info(f"  z={z_snap:.3f}: {len(xs[-1]):,} LRGs")
        del ball, mock, lrg
        gc.collect()

    x   = np.concatenate(xs)
    y   = np.concatenate(ys)
    z_c = np.concatenate(zs)
    m   = np.concatenate(ms)

    chi  = np.sqrt(x**2 + y**2 + z_c**2)
    mask = (chi >= chi_min) & (chi <= chi_max) & (m <= MAX_M200M_H)
    x, y, z_c = x[mask], y[mask], z_c[mask]
    logger.info(
        f"  {mask.sum():,} LRGs after chi/mass filter "
        f"(chi=[{chi_min:.1f},{chi_max:.1f}], M200m <= {MAX_M200M_H/H_ABACUS:.2e} M_sun)"
    )
    del m, chi, mask
    gc.collect()

    if len(x) == 0:
        raise RuntimeError(f"hod_idx={hod_idx}: no LRGs survived chi/mass filter")

    if len(x) > MAX_LRG:
        rng = np.random.default_rng(seed=hod_idx)
        sub = rng.choice(len(x), MAX_LRG, replace=False)
        x, y, z_c = x[sub], y[sub], z_c[sub]
        logger.info(f"  subsampled to {MAX_LRG:,} LRGs (seed={hod_idx})")

    r     = np.sqrt(x**2 + y**2 + z_c**2)
    theta = np.arccos(z_c / r)
    phi   = np.arctan2(y, x) % (2 * np.pi)
    return theta, phi


def _worker_init(omp_threads: int):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(omp_threads)


def _process_one_map(npy_path: str) -> tuple[str, list[float], np.ndarray, np.ndarray, float]:
    """Worker: load one preconvolved map, apply cap_filter with fork-inherited catalog."""
    import healpy as hp
    t0    = perf_counter()
    ymap  = np.load(npy_path).astype(np.float64)
    nside = hp.get_nside(ymap)
    # Divide available threads among concurrent map workers
    n_threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) // _NPROC_MAPS)
    vals      = cap_filter(ymap, nside, _THETA_LRG, _PHI_LRG, n_workers=n_threads)
    profile   = vals.mean(axis=0)
    err       = vals.std(axis=0) / np.sqrt(len(_THETA_LRG))
    params    = parse_pressure_params(Path(npy_path).stem)
    elapsed   = perf_counter() - t0
    gc.collect()
    return npy_path, params, profile, err, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",   type=int, default=None,
                        help="HOD LHC index 0-199")
    parser.add_argument("--nproc", type=int, default=NPROC_MAPS,
                        help=f"Parallel map workers (default {NPROC_MAPS})")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all per-idx hod_NNN.npz into hod_lhc_stacked.npz")
    args = parser.parse_args()

    setup_logging(f"hod_lhc_{args.idx:03d}" if args.idx is not None else "hod_lhc_merge")

    # ── merge ──────────────────────────────────────────────────────────────────
    if args.merge:
        files = sorted(OUT_DIR.glob("hod_*.npz"))
        if not files:
            raise FileNotFoundError(f"No hod_*.npz in {OUT_DIR}")
        all_pressure, all_profiles, all_errors, all_hod, lhc_indices = [], [], [], [], []
        apertures = pnames_p = pnames_h = None
        for f in files:
            d = np.load(f, allow_pickle=True)
            all_pressure.append(d["pressure_params"])
            all_profiles.append(d["profiles_cap"])
            all_errors.append(d["errors_cap"])
            all_hod.append(d["hod_params"])
            lhc_indices.append(int(d["hod_idx"]))
            if apertures is None:
                apertures = d["apertures_cap"]
                pnames_p  = d["param_names_pressure"]
                pnames_h  = d["param_names_hod"]
        order = np.argsort(lhc_indices)
        out = OUT_DIR / "hod_lhc_stacked.npz"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(out,
            pressure_params      = np.array(all_pressure)[order],
            profiles_cap         = np.array(all_profiles)[order],
            errors_cap           = np.array(all_errors)[order],
            hod_params           = np.array(all_hod)[order],
            param_names_pressure = pnames_p,
            param_names_hod      = pnames_h,
            apertures_cap        = apertures,
            lhc_indices          = np.array(lhc_indices)[order],
        )
        logger.info(f"Merged {len(files)} files → {out}")
        return

    # ── single HOD index ───────────────────────────────────────────────────────
    if args.idx is None:
        parser.error("Provide --idx N or --merge")

    hod_idx = args.idx
    nproc   = args.nproc

    if not YMAP_DIR_CONV.exists():
        raise FileNotFoundError(
            f"Preconvolved map directory not found: {YMAP_DIR_CONV}\n"
            "Run preconvolve_maps.py first."
        )
    map_files = sorted(YMAP_DIR_CONV.glob("*.npy"))
    if not map_files:
        raise FileNotFoundError(f"No .npy maps in {YMAP_DIR_CONV}")
    logger.info(f"Found {len(map_files)} preconvolved maps")

    chi_min, chi_max = get_chi_range()
    logger.info(f"chi range: [{chi_min:.1f}, {chi_max:.1f}] Mpc/h")

    samples = np.loadtxt(LHC_PATH)  # (200, 5)
    row = samples[hod_idx]
    logM_cut, logM1, sigma, alpha_hod, kappa = row
    logger.info(
        f"HOD {hod_idx}: logM_cut={logM_cut:.4f}  logM1={logM1:.4f}  "
        f"sigma={sigma:.4f}  alpha={alpha_hod:.4f}  kappa={kappa:.4f}"
    )

    # Generate catalog once — set as globals so map workers fork-inherit them
    global _THETA_LRG, _PHI_LRG, _NPROC_MAPS
    _THETA_LRG, _PHI_LRG = generate_lrgs(
        hod_idx, chi_min, chi_max, logM_cut, logM1, sigma, alpha_hod, kappa,
    )
    _NPROC_MAPS = nproc
    logger.info(f"Catalog ready: {len(_THETA_LRG):,} LRGs → stacking {len(map_files)} maps ({nproc} workers)")

    omp_per_worker = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", nproc)) // nproc)
    results = {}
    with ProcessPoolExecutor(
        max_workers=nproc,
        initializer=_worker_init,
        initargs=(omp_per_worker,),
    ) as ex:
        futures = {ex.submit(_process_one_map, str(f)): f for f in map_files}
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                npy_path, params, profile, err, elapsed = fut.result()
                results[npy_path] = (params, profile, err)
                logger.info(f"[{i}/{len(map_files)}] {Path(npy_path).name}  ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"[{i}/{len(map_files)}] FAILED {f.name}: {e}")

    all_pressure, all_profiles, all_errors = [], [], []
    for f in sorted(results):
        params, profile, err = results[f]
        all_pressure.append(params)
        all_profiles.append(profile)
        all_errors.append(err)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"hod_{hod_idx:03d}.npz"
    np.savez(out,
        pressure_params      = np.array(all_pressure),
        param_names_pressure = PARAM_NAMES_PRESSURE,
        profiles_cap         = np.array(all_profiles),
        errors_cap           = np.array(all_errors),
        apertures_cap        = APERTURES_CAP,
        hod_params           = row,
        param_names_hod      = PARAM_NAMES_HOD,
        hod_idx              = hod_idx,
    )
    logger.info(f"Saved {len(all_profiles)} profiles → {out}")


if __name__ == "__main__":
    main()
