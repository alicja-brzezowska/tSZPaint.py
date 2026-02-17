from pathlib import Path
import os

import healpy as hp
import numpy as np
from loguru import logger

from tszpaint.config import (
    INTERPOLATORS_PATH,
)
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import get_angular_size_from_comoving
from tszpaint.logging import trace_calls, memory_usage, array_size, time_calls
from tszpaint.paint.abacus_loader import SimulationData, load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.paint.visualize import PlotConfig, Visualizer
from tszpaint.paint.weights import compute_weights
from tszpaint.y_profile.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile.y_profile import (
    create_battaglia_profile,
)
from tszpaint.paint.tree import build_tree, query_tree

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"


def load_interpolator(path: Path = JAX_PATH):
    return BattagliaLogInterpolator.from_pickle(path)


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
    try:
        import numba  # local import to avoid hard dependency at module import time

        logger.info(f"Numba threads in use: {numba.get_num_threads()}")
    except Exception:
        logger.info("Numba threads in use: unavailable")
    logger.info(f"Starting vectorized paint: {len(M_halos)} halos, nside={nside}")
    logger.info(
        f"  particle_counts: {particle_counts.nbytes / 1e6:.1f}MB, dtype={particle_counts.dtype}"
    )

    #tree, pix_xyz, pix_indices = build_tree(config)
    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)

    theta_200 = get_angular_size_from_comoving(MODEL, radius, z) * config.search_radius 
    #pix_in_halos, distances, halo_starts, halo_counts, halo_indices = query_tree(
    #    config = config, 
    #    halo_xyz = halo_xyz,
    #    theta_200 = theta_200,
    #    particle_tree = tree,
    #    particle_xyz = pix_xyz,
    #)
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = (
        find_pixels_in_halos(nside, halo_xyz, theta_200, n_workers = 8)
    )

    logger.info(f"theta_200 stats: min={theta_200.min():.3e}, median={np.median(theta_200):.3e}, max={theta_200.max():.3e}")
    logger.info(f"pixel-halo pairs: {len(pix_in_halos):,}")
    logger.info(f"distances bytes: {distances.nbytes/1e9:.2f} GB")
    logger.info(f"weights expected bytes (float64): {len(distances)*8/1e9:.2f} GB")

    if use_weights:
        weights = compute_weights(
            config,
            pixel_indices=pix_in_halos,
            distances=distances,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            theta_200=theta_200,
            particle_counts=particle_counts,
        )
    else:
        weights = np.ones(len(pix_in_halos), dtype=np.float64)

    log_M = np.log10(M_halos)
    log_distances = np.log(distances + 1e-40)

    # Create halo index array to map each pixel to its halo's mass
    log_M_values = log_M[halo_indices]
    z_values = np.full_like(distances, z, dtype=np.float32)

    y_values = interpolator.eval_for_logs(log_distances, z_values, log_M_values)

    y_values *= weights

    y_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    np.add.at(y_map, pix_in_halos, y_values)

    y_per_halo = np.bincount(
        halo_indices, weights=y_values, minlength=len(M_halos)
    )

    return y_map, y_per_halo


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
    paint_and_visualize(config, data, output_file, interpolator_path, use_weights)


def paint_and_visualize(
    config: PainterConfig,
    data: SimulationData,
    output_file: str | None = None,
    interpolator_path: Path = JAX_PATH,
    use_weights: bool = True,
):
    interpolator = load_interpolator(interpolator_path)
    y_map, y_per_halo = paint_y_wrapper(
        config,
        data,
        interpolator=interpolator,
        use_weights=use_weights,
    )
    display_map_statistics(y_map)
    if output_file:
        hp.write_map(output_file, y_map, overwrite=True, nest=True)
        logger.info(f"Saved to {output_file}")

    vis = Visualizer(config.nside, output_file)
    vis.plot_ra_dec(y_map, PlotConfig.standard(), sim_data=data)
    vis.plot_Y_vs_M(data, y_per_halo)
    vis.visualize_y_map(y_map)
