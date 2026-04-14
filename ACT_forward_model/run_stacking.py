import argparse
import copy
import gc
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from astropy.table import Table
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ACT_data_match import APERTURES_CAP, APERTURES_RR, stack_profiles

from tszpaint.cosmology.mass_conversion import H_ABACUS, MAX_M200M_H
from tszpaint.logging import setup_logging
from tszpaint.paint.abacus_loader import obtain_healcount_edges

# ── run configuration ────────────────────────────────────────────────────────
NPROC = 8  # parallel map workers (fiducial single-HOD run)
FILTER_TYPES = ["cap", "ring_ring"]
APPLY_BEAM = True
# ─────────────────────────────────────────────────────────────────────────────

YMAP_ROOT = Path("/home/ab2927/rds/hpc-work/tSZPaint_data")
OUT_ROOT = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles")
YMAP_DIR_RAW = YMAP_ROOT / "Step0677-0682"
YMAP_DIR_CONV = YMAP_ROOT / "preconvolved_Step0677-0682"

HEALCOUNTS_DIR = Path(
    "/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/total/heal-counts"
)
SUMMED_STEPS = [
    "Step0617-0622",
    "Step0623-0627",
    "Step0628-0634",
    "Step0635-0640",
    "Step0641-0646",
    "Step0647-0652",
    "Step0653-0658",
    "Step0659-0664",
    "Step0665-0670",
    "Step0671-0676",
    "Step0677-0682",
]

# Fiducial HOD catalog (used when --hod-idx is not given)
FIDUCIAL_LRG_SNAPSHOTS = [
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.503/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.542/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.582/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.625/galaxies_rsd/LRGs.dat",
    YMAP_ROOT / "hod_mocks/lightcone_halos/z0.671/galaxies_rsd/LRGs.dat",
]

# HOD LHC grid catalog template (used when --hod-idx N is given)
HOD_SNAPSHOT_TEMPLATES = [
    str(
        YMAP_ROOT
        / "hod_mocks/lhc/{idx:03d}/lightcone_halos/z0.503/galaxies_rsd/LRGs.dat"
    ),
    str(
        YMAP_ROOT
        / "hod_mocks/lhc/{idx:03d}/lightcone_halos/z0.542/galaxies_rsd/LRGs.dat"
    ),
    str(
        YMAP_ROOT
        / "hod_mocks/lhc/{idx:03d}/lightcone_halos/z0.582/galaxies_rsd/LRGs.dat"
    ),
    str(
        YMAP_ROOT
        / "hod_mocks/lhc/{idx:03d}/lightcone_halos/z0.625/galaxies_rsd/LRGs.dat"
    ),
    str(
        YMAP_ROOT
        / "hod_mocks/lhc/{idx:03d}/lightcone_halos/z0.671/galaxies_rsd/LRGs.dat"
    ),
]

PARAM_NAMES = ["alpha", "beta0", "gamma", "log10P0"]
HOD_PARAM_NAMES = ["log_Mcut", "log_M1", "sigma", "alpha_hod", "kappa"]
HOD_LHC_FILE = Path(__file__).parent / "hod_lhc_samples.txt"

PARAM_RE = re.compile(
    r"alpha=(?P<alpha>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_beta0=(?P<beta0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_gamma=(?P<gamma>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
    r"_log10P0=(?P<log10P0>[+-]?\d+\.?\d*(?:e[+-]?\d+)?)"
)

_THETA_LRG = None
_PHI_LRG = None


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


def load_and_filter_lrgs(snapshots: list, chi_min: float, chi_max: float):
    tables = [Table.read(str(f), format="ascii.ecsv") for f in snapshots]
    x = np.concatenate([np.asarray(t["x"]) for t in tables])
    y = np.concatenate([np.asarray(t["y"]) for t in tables])
    z = np.concatenate([np.asarray(t["z"]) for t in tables])
    m = np.concatenate([np.asarray(t["mass"]) for t in tables])
    chi = np.sqrt(x**2 + y**2 + z**2)
    mask = (chi >= chi_min) & (chi <= chi_max) & (m <= MAX_M200M_H)
    x, y, z = x[mask], y[mask], z[mask]
    logger.info(
        f"LRGs: {len(m):,} total → {mask.sum():,} kept  "
        f"(chi=[{chi_min:.1f},{chi_max:.1f}] Mpc/h, M200m <= {MAX_M200M_H / H_ABACUS:.2e} M_sun)"
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return theta, phi


HOD_CONFIG_PATH = Path(__file__).resolve().parents[1] / "abacusHOD" / "config.yaml"
HOD_Z_SNAPSHOTS = [0.503, 0.542, 0.582, 0.625, 0.671]


def _patch_asdf():
    """Monkey-patch asdf so old LCOrigins header key is transparently aliased."""
    import abacusnbody.hod.abacus_hod as _hod_module
    import asdf as _asdf

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


def generate_lrgs_in_memory(hod_idx: int, chi_min: float, chi_max: float):
    """Run AbacusHOD in-memory for hod_idx"""
    from abacusnbody.hod.abacus_hod import AbacusHOD

    _patch_asdf()

    config = yaml.safe_load(open(HOD_CONFIG_PATH))
    sim_params = config["sim_params"]
    hod_params = copy.deepcopy(config["HOD_params"])

    row = np.loadtxt(HOD_LHC_FILE, comments="#")[hod_idx]
    logM_cut, logM1, sigma, alpha_hod, kappa = row
    hod_params["LRG_params"].update(
        dict(
            logM_cut=logM_cut,
            logM1=logM1,
            sigma=sigma,
            alpha=alpha_hod,
            kappa=kappa,
        )
    )
    logger.info(
        f"HOD {hod_idx}: logM_cut={logM_cut:.4f}  logM1={logM1:.4f}  "
        f"sigma={sigma:.4f}  alpha={alpha_hod:.4f}  kappa={kappa:.4f}"
    )

    xs, ys, zs, ms = [], [], [], []
    for z in HOD_Z_SNAPSHOTS:
        sp = copy.deepcopy(sim_params)
        sp["z_mock"] = z
        sp["output_dir"] = "/tmp"  # unused — write_to_disk=False
        ball = AbacusHOD(sp, hod_params)
        mock = ball.run_hod(ball.tracers, want_rsd=True, write_to_disk=False, Nthread=8)
        lrg = mock["LRG"]
        xs.append(np.asarray(lrg["x"], dtype=np.float64))
        ys.append(np.asarray(lrg["y"], dtype=np.float64))
        zs.append(np.asarray(lrg["z"], dtype=np.float64))
        ms.append(np.asarray(lrg["mass"], dtype=np.float64))
        logger.info(f"  z={z:.3f}: {len(xs[-1]):,} LRGs")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z_c = np.concatenate(zs)
    m = np.concatenate(ms)

    chi = np.sqrt(x**2 + y**2 + z_c**2)
    mask = (chi >= chi_min) & (chi <= chi_max) & (m <= MAX_M200M_H)
    x, y, z_c = x[mask], y[mask], z_c[mask]
    logger.info(
        f"LRGs: {len(m):,} total → {mask.sum():,} kept  "
        f"(chi=[{chi_min:.1f},{chi_max:.1f}] Mpc/h, M200m <= {MAX_M200M_H / H_ABACUS:.2e} M_sun)"
    )
    r = np.sqrt(x**2 + y**2 + z_c**2)
    theta = np.arccos(z_c / r)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return theta, phi


def _worker_init(omp_threads: int):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(omp_threads)


def _process_one(
    fname: str, ymap_dir: Path, apply_beam: bool, n_workers: int, filter_type: str
):
    """Worker: stack one y-map."""
    params = parse_params(fname)
    t0 = perf_counter()
    y_stacked, y_err = stack_profiles(
        str(ymap_dir / fname),
        _THETA_LRG,
        _PHI_LRG,
        apply_beam=apply_beam,
        n_workers=n_workers,
        filter_type=filter_type,
    )
    elapsed = perf_counter() - t0
    gc.collect()
    return fname, [params[k] for k in PARAM_NAMES], y_stacked, y_err, elapsed


def run_filter(
    filter_type: str,
    hod_idx,
    nproc: int,
    ymap_start,
    ymap_end,
    hod_params,
    ymap_dir: Path,
    apply_beam: bool,
):
    batched = ymap_start is not None or ymap_end is not None

    if hod_idx is None:
        output_name = f"fiducial_{filter_type}.npz"
        out_dir = OUT_ROOT
        log_name = f"{filter_type}_stack"
    else:
        output_name = f"hod_{hod_idx:03d}_{filter_type}.npz"
        out_dir = OUT_ROOT / "hod_lhc"
        log_name = f"hod_stack_{hod_idx:03d}_{filter_type}"

    setup_logging(log_name)
    logger.info(
        f"Starting {filter_type} stack"
        + (f" (HOD index {hod_idx})" if hod_idx is not None else "")
    )

    ext = ".npy" if not apply_beam else ".asdf"
    all_fnames = sorted(f.name for f in ymap_dir.glob(f"*{ext}"))
    if batched:
        all_fnames = all_fnames[ymap_start:ymap_end]
        logger.info(
            f"Batched run: y-maps [{ymap_start}:{ymap_end}] → {len(all_fnames)} maps"
        )

    out_file = out_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing {len(all_fnames)} maps ({nproc} workers) → {out_file}")

    results = {}
    n_workers = max(
        1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) // nproc
    )
    omp_per_worker = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", nproc)) // nproc)
    with ProcessPoolExecutor(
        max_workers=nproc, initializer=_worker_init, initargs=(omp_per_worker,)
    ) as ex:
        futures = {
            ex.submit(
                _process_one, fname, ymap_dir, apply_beam, n_workers, filter_type
            ): fname
            for fname in all_fnames
        }
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                fname, params, y_stacked, y_err, elapsed = fut.result()
                results[fname] = (params, y_stacked, y_err)
                logger.info(f"[{i}/{len(all_fnames)}] {fname}  ({elapsed:.1f}s)")
            except Exception as e:
                logger.error(f"[{i}/{len(all_fnames)}] FAILED {fname}: {e}")

    all_params, profiles, errors = [], [], []
    for fname in sorted(results):
        params, y_stacked, y_err = results[fname]
        all_params.append(params)
        profiles.append(y_stacked)
        errors.append(y_err)

    save_kwargs = dict(
        params=np.array(all_params),
        param_names=PARAM_NAMES,
        profiles=np.array(profiles),
        errors=np.array(errors),
        apertures_arcmin=APERTURES_CAP if filter_type == "cap" else APERTURES_RR,
    )
    if hod_params is not None:
        save_kwargs["hod_params"] = hod_params
        save_kwargs["param_names_hod"] = HOD_PARAM_NAMES
    np.savez(out_file, **save_kwargs)
    logger.info(f"Saved {len(profiles)} profiles → {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hod-idx",
        type=int,
        default=None,
        help="HOD LHC index (0-199). Omit for fiducial single-HOD run.",
    )
    parser.add_argument(
        "--nproc", type=int, default=None, help="Override NPROC map workers."
    )
    parser.add_argument(
        "--ymap-start",
        type=int,
        default=None,
        help="Start index into sorted y-map list (for batched runs).",
    )
    parser.add_argument(
        "--ymap-end",
        type=int,
        default=None,
        help="End index into sorted y-map list (exclusive, for batched runs).",
    )
    args = parser.parse_args()

    hod_idx = args.hod_idx
    nproc = args.nproc or NPROC
    ymap_start = args.ymap_start
    ymap_end = args.ymap_end

    # ── resolve run-specific config ──────────────────────────────────────────
    # fiducial run
    if hod_idx is None:
        ymap_dir = YMAP_DIR_RAW
        apply_beam = APPLY_BEAM
        snapshots = FIDUCIAL_LRG_SNAPSHOTS
        hod_params = None
    else:
        # hod-grid run
        ymap_dir = YMAP_DIR_CONV
        apply_beam = False  # beam already applied by preconvolve_maps.py
        snapshots = None  # unused — generate_lrgs_in_memory is called instead
        hod_params = np.loadtxt(HOD_LHC_FILE, comments="#")[
            hod_idx
        ]  # (5,) saved to output
    # ─────────────────────────────────────────────────────────────────────────

    chi_min, chi_max = get_chi_range()
    logger.info(f"y-map chi range: [{chi_min:.1f}, {chi_max:.1f}] Mpc/h")

    global _THETA_LRG, _PHI_LRG
    if hod_idx is not None:
        _THETA_LRG, _PHI_LRG = generate_lrgs_in_memory(hod_idx, chi_min, chi_max)
    else:
        _THETA_LRG, _PHI_LRG = load_and_filter_lrgs(snapshots, chi_min, chi_max)

    MAX_LRG = 4_000_000
    if hod_idx is not None and len(_THETA_LRG) > MAX_LRG:
        rng = np.random.default_rng(seed=hod_idx)
        sub = rng.choice(len(_THETA_LRG), MAX_LRG, replace=False)
        _THETA_LRG = _THETA_LRG[sub]
        _PHI_LRG = _PHI_LRG[sub]
        logger.info(f"Subsampled to {MAX_LRG:,} galaxies (seed={hod_idx})")

    for filter_type in FILTER_TYPES:
        run_filter(
            filter_type=filter_type,
            hod_idx=hod_idx,
            nproc=nproc,
            ymap_start=ymap_start,
            ymap_end=ymap_end,
            hod_params=hod_params,
            ymap_dir=ymap_dir,
            apply_beam=apply_beam,
        )


if __name__ == "__main__":
    main()
