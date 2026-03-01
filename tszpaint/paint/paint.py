import os
from pathlib import Path

import asdf
import healpy as hp
import numba
import numpy as np
from loguru import logger

from tszpaint.config import (
    INTERPOLATORS_PATH,
)
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import get_angular_size_from_comoving
from tszpaint.logging import array_size, memory_usage, time_calls, trace_calls
from tszpaint.paint.abacus_loader import SimulationData, load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.paint.visualize import PlotConfig, Visualizer
from tszpaint.paint.weights import compute_weights
from tszpaint.scripts.radial_profile import (
    RadialProfileBuilder,
    RadialProfileBuilderConfig,
)
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


@memory_usage
@array_size
@time_calls
@trace_calls
def paint_y(
    config: PainterConfig,
    data: SimulationData,
    interpolator: BattagliaLogInterpolator,
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

    logger.info(f"Starting paint: {len(data.m_halos)} halos, nside={config.nside}")
    logger.info(
        f"  particle_counts: {data.particle_counts.nbytes / 1e6:.1f}MB, dtype={data.particle_counts.dtype}"
    )

    halo_xyz = convert_rad_to_cart(data.theta, data.phi)

    r_90_phys = get_angular_size_from_comoving(MODEL, data.radii_halos, data.redshift)
    r_search = r_90_phys * config.search_radius
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = (
        find_pixels_in_halos(config.nside, halo_xyz, r_search, n_workers=8)
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
            particle_counts=data.particle_counts,
            method="vectorized",
        )
    else:
        weights = np.ones(len(pix_in_halos), dtype=np.float64)

    log_M = np.log10(data.m_halos)
    log_distances = np.log(distances + 1e-40)

    logger.info(
        f"""log_distances stats: min={log_distances.min():.3e}, \n
        median={np.median(log_distances):.3e}, max={log_distances.max():.3e}"""
    )

    # Create halo index array to map each pixel to its halo's mass
    log_M_values = log_M[halo_indices]
    z_values = np.full_like(distances, z, dtype=np.float32)

    y_values = interpolator.eval_for_logs(log_distances, z_values, log_M_values)  # pyright: ignore[reportUnknownVariableType, reportCallIssue]

    y_values *= weights

    y_map = np.zeros(hp.nside2npix(config.nside), dtype=np.float32)
    np.add.at(y_map, pix_in_halos, y_values)

    y_per_halo = np.bincount(
        halo_indices, weights=y_values, minlength=len(data.m_halos)
    )

    # for plotting y vs r/r200
    profile_n_halos = 1000
    profile_seed = 123
    profile_logM_centers = [12.7, 13.0, 13.7, 14.0, 14.7, 15.0]
    profile_logM_halfwidth = 0.2

    radial_cfg = RadialProfileBuilderConfig(
        r_search=r_search,
        num_halos=profile_n_halos,
        seed=profile_seed,
        log_m_centers=profile_logM_centers,
        log_m_halfwidth=profile_logM_halfwidth,
    )
    radial_profiles = RadialProfileBuilder(
        radial_cfg,
        data,
        halo_starts,
        halo_counts,
        distances,
        interpolator,
        MODEL,
    ).build(y_map)

    return y_map, y_per_halo, radial_profiles


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
    y_map, y_per_halo, radial_profile = paint_y(
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
