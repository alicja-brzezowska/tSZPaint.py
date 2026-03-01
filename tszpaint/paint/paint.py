import os
from pathlib import Path
import numba

import asdf
import healpy as hp
import numpy as np
from loguru import logger

from tszpaint.config import (
    INTERPOLATORS_PATH,
)
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import compute_theta_200, get_angular_size_from_comoving
from tszpaint.logging import array_size, memory_usage, time_calls, trace_calls
from tszpaint.paint.abacus_loader import SimulationData, load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.paint.visualize import PlotConfig, Visualizer
from tszpaint.paint.weights import compute_weights
from tszpaint.y_profile.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile.y_profile import (
    create_battaglia_profile,
)

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"


def load_interpolator(path: Path = JAX_PATH):
    return BattagliaLogInterpolator.from_pickle(path)


def y_vs_r(
    interpolator: BattagliaLogInterpolator,
    M_halos: np.ndarray,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    y_values: np.ndarray,
    r_search: np.ndarray,
    z: float,
    n_halos: int,
    seed: int,
    logM_center: float | None,
    logM_halfwidth: float,
):
    profile_n_bins = 20
    rng = np.random.default_rng(seed)
    if logM_center is None:
        candidate_halos = np.arange(len(M_halos))
    else:
        logM = np.log10(M_halos)
        in_bin = np.abs(logM - logM_center) <= logM_halfwidth
        candidate_halos = np.flatnonzero(in_bin)

    n_total = len(candidate_halos)
    if n_total == 0:
        return {
            "x_centers": np.array([]),
            "y_mean": np.array([]),
            "y_err": np.array([]),
            "counts": np.array([]),
            "n_sample": 0,
            "x_ref": np.array([]),
            "y_battaglia": np.array([]),
            "mass_ref": np.nan,
            "logM_center": logM_center,
        }

    n_sample = min(n_halos, n_total)
    sample_halos = rng.choice(candidate_halos, size=n_sample, replace=False)

    theta_200 = compute_theta_200(MODEL, M_halos, z)
    ratio = r_search[sample_halos] / theta_200[sample_halos]
    x_max = np.nanmax(ratio[np.isfinite(ratio)])
    x_min = max(1e-4, x_max / 1e3)
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), profile_n_bins + 1)
    x_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    sum_y = np.zeros(profile_n_bins, dtype=np.float64)
    sum_y2 = np.zeros(profile_n_bins, dtype=np.float64)
    counts = np.zeros(profile_n_bins, dtype=np.float64)

    for h in sample_halos:
        start = halo_starts[h]
        count = halo_counts[h]
        if count == 0:
            continue
        d = distances[start : start + count]
        y = y_values[start : start + count]
        keep = d <= r_search[h]
        if not np.any(keep):
            continue
        d = d[keep]
        y = y[keep]
        x = d / theta_200[h]
        bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
        bin_ids = np.minimum(bin_ids, profile_n_bins - 1)

        sum_y += np.bincount(bin_ids, weights=y, minlength=profile_n_bins)
        sum_y2 += np.bincount(bin_ids, weights=y**2, minlength=profile_n_bins)
        counts += np.bincount(bin_ids, minlength=profile_n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        y_mean = sum_y / counts
        y_var = sum_y2 / counts - y_mean**2
        y_std = np.sqrt(np.maximum(y_var, 0.0))
        y_err = y_std / np.sqrt(counts)

    mass_ref = np.median(M_halos[sample_halos])
    theta_ref = compute_theta_200(MODEL, np.array([mass_ref]), z)[0]
    x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    theta_values = np.maximum(x_ref * theta_ref, 1e-40)
    log_theta = np.log(theta_values)
    log_M = np.full_like(log_theta, np.log10(mass_ref), dtype=np.float64)
    z_values = np.full_like(log_theta, z, dtype=np.float32)
    y_battaglia = np.asarray(interpolator.eval_for_logs(log_theta, z_values, log_M))

    return {
        "x_centers": x_centers,
        "y_mean": y_mean,
        "y_err": y_err,
        "counts": counts,
        "n_sample": n_sample,
        "x_ref": x_ref,
        "y_battaglia": y_battaglia,
        "mass_ref": mass_ref,
        "logM_center": logM_center,
    }


@memory_usage
@array_size
@time_calls
@trace_calls
def paint_y(
    config: PainterConfig,
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    radius: np.ndarray,
    z: float,
    nside: int,
    use_weights: bool = True,
):
    logger.info(
        "CPU threads: "
        f"SLURM_CPUS_PER_TASK={os.getenv('SLURM_CPUS_PER_TASK')}, "
        f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}, "
        f"NUMBA_NUM_THREADS={os.getenv('NUMBA_NUM_THREADS')}, "
        f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS')}, "
        f"OPENBLAS_NUM_THREADS={os.getenv('OPENBLAS_NUM_THREADS')}, "
        f"NUMEXPR_NUM_THREADS={os.getenv('NUMEXPR_NUM_THREADS')}"
    )
    logger.info(f"Numba threads in use: {numba.get_num_threads()}")

    logger.info(f"Starting paint: {len(M_halos)} halos, nside={nside}")
    logger.info(
        f"  particle_counts: {particle_counts.nbytes / 1e6:.1f}MB, dtype={particle_counts.dtype}"
    )

    # if tree:
    # tree, pix_xyz, pix_indices = build_tree(config)

    # pix_in_halos, distances, halo_starts, halo_counts, halo_indices = query_tree(
    #    config = config,
    #    halo_xyz = halo_xyz,
    #    r_90 = r_90,
    #    particle_tree = tree,
    #    particle_xyz = pix_xyz,
    # )

    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)

    r_90_phys = get_angular_size_from_comoving(MODEL, radius, z)
    r_search = r_90_phys * config.search_radius
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = (
        find_pixels_in_halos(nside, halo_xyz, r_search, n_workers=8)
    )

    logger.info(
        f"r_90 stats: min={r_90_phys.min():.3e}, median={np.median(r_90_phys):.3e}, max={r_90_phys.max():.3e}"
    )
    logger.info(
        f"r_search stats: min={r_search.min():.3e}, median={np.median(r_search):.3e}, max={r_search.max():.3e}"
    )
    logger.info(f"pixel-halo pairs: {len(pix_in_halos):,}")
    logger.info(f"distances bytes: {distances.nbytes / 1e9:.2f} GB")
    logger.info(f"weights expected bytes (float64): {len(distances) * 8 / 1e9:.2f} GB")

    if use_weights:
        weights = compute_weights(
            config,
            pixel_indices=pix_in_halos,
            distances=distances,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            r_90=r_90_phys,
            particle_counts=particle_counts,
            method="vectorized",
        )
    else:
        weights = np.ones(len(pix_in_halos), dtype=np.float64)

    log_M = np.log10(M_halos)
    log_distances = np.log(distances + 1e-40)

    logger.info(
        f"log_distances stats: min={log_distances.min():.3e}, "
        f"median={np.median(log_distances):.3e}, max={log_distances.max():.3e}"
    )

    # Create halo index array to map each pixel to its halo's mass
    log_M_values = log_M[halo_indices]
    z_values = np.full_like(distances, z, dtype=np.float32)

    y_values = interpolator.eval_for_logs(log_distances, z_values, log_M_values)

    y_values *= weights

    y_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    np.add.at(y_map, pix_in_halos, y_values)

    y_per_halo = np.bincount(halo_indices, weights=y_values, minlength=len(M_halos))

    # for plotting y vs r/r200
    profile_n_halos = 1000
    profile_seed = 123
    profile_logM_centers = [12.7, 13.0, 13.7, 14.0, 14.7, 15.0]
    profile_logM_halfwidth = 0.2

    radial_profiles = [
        y_vs_r(
            interpolator=interpolator,
            M_halos=M_halos,
            distances=distances,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            y_values=y_values,
            r_search=r_search,
            z=z,
            n_halos=profile_n_halos,
            seed=profile_seed,
            logM_center=center,
            logM_halfwidth=profile_logM_halfwidth,
        )
        for center in profile_logM_centers
    ]
    return y_map, y_per_halo, radial_profiles


def paint_y_wrapper(
    config: PainterConfig,
    data: SimulationData,
    interpolator: BattagliaLogInterpolator,
    use_weights: bool = True,
):
    """
    Paint y-map wrapper
    """
    return paint_y(
        config,
        data.theta,
        data.phi,
        data.m_halos,
        data.particle_counts,
        interpolator,
        data.radii_halos,
        data.redshift,
        config.nside,
        use_weights,
    )


def display_map_statistics(y_map: np.ndarray):
    logger.info("\nMap statistics:")
    logger.info(f"  Min: {y_map.min():.3e}")
    logger.info(f"  Max: {y_map.max():.3e}")
    logger.info(f"  Mean: {y_map.mean():.3e}")
    logger.info(f"  Non-zero pixels: {np.sum(y_map > 0)}/{len(y_map)}")


def paint_abacus(
    config: PainterConfig,
    halo_dir: Path,
    healcounts_file_1: Path,
    output_file: str,
    interpolator_path: Path = JAX_PATH,
    use_weights: bool = True,
):
    """
    Paint the y-compton map using Abacus halo catalogs and heal-counts.
    """
    data = load_abacus_for_painting(
        halo_dir=halo_dir,
        healcounts_file_1=healcounts_file_1,
        nside=config.nside,
    )
    paint_and_visualize(
        config,
        data,
        halo_dir,
        healcounts_file_1,
        output_file,
        interpolator_path,
        use_weights,
    )


def paint_and_visualize(
    config: PainterConfig,
    data: SimulationData,
    halo_dir: Path | None = None,
    healcounts_file_1: Path | None = None,
    output_file: str | None = None,
    interpolator_path: Path = JAX_PATH,
    use_weights: bool = True,
):
    interpolator = load_interpolator(interpolator_path)
    y_map, y_per_halo, radial_profile = paint_y_wrapper(
        config,
        data,
        interpolator=interpolator,
        use_weights=use_weights,
    )
    display_map_statistics(y_map)
    output_stub = None
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        halo_index = np.arange(len(data.m_halos), dtype=np.int64)
        af = asdf.AsdfFile(
            {
                "header": {
                    "nside": config.nside,
                    "nest": True,
                    "redshift": data.redshift,
                    "search_radius_multiplier": config.search_radius,
                    "bin_width": config.weight_bin_width,
                    "healpix_file": str(healcounts_file_1)
                    if healcounts_file_1 is not None
                    else None,
                    "halo_catalog_file": str(halo_dir)
                    if halo_dir is not None
                    else None,
                },
                "data": {
                    "y_map": y_map,
                    "halo_index": halo_index,
                    "theta_halo": data.theta,
                    "phi_halo": data.phi,
                    "y_per_halo": y_per_halo,
                    "halo_M": data.m_halos,
                    "r98_halo": data.radii_halos,
                    "radial_profile": radial_profile,
                },
            }
        )
        af.write_to(output_path)

        logger.info(f"Saved to {output_path}")
        output_stub = str(output_path.with_suffix(""))

    vis = Visualizer(config.nside, output_stub)
    vis.plot_ra_dec(y_map, PlotConfig.standard(), sim_data=data)
    vis.plot_Y_vs_M(data, y_per_halo)
    vis.plot_Y_vs_R200(radial_profile)
    # vis.visualize_y_map(y_map)
