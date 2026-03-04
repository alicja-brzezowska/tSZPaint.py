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
# from tszpaint.paint.visualize import PlotConfig, Visualizer
from tszpaint.paint.weights import compute_weights
# from tszpaint.scripts.radial_profile import (
#     RadialProfileBuilder,
#     RadialProfileBuilderConfig,
# )
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


def get_real_space_from_eigenvals(
    eigenvalues: np.ndarray,
    r98: np.ndarray,
) -> np.ndarray:
    """Convert inertia eigenvalues to real-space semi-axis lengths a>=b>=c.

    Assumes eigenvalues are sorted largest->smallest (inertia ordering).
    Uses volume-preserving normalization abc = r98^3.
    """
    eps = np.finfo(np.float32).tiny
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float32), eps)
    r98 = np.asarray(r98, dtype=np.float32)

    ratio_ba = np.sqrt(eigenvalues[:, 2] / eigenvalues[:, 1])
    ratio_ca = np.sqrt(eigenvalues[:, 2] / eigenvalues[:, 0])

    a = r98 / np.cbrt(ratio_ba * ratio_ca)
    b = ratio_ba * a
    c = ratio_ca * a
    return np.stack([a, b, c], axis=1)




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

    if config.halo_geometry == "triaxial":
        semi_axes_comoving = get_real_space_from_eigenvals(
            data.eigenvalues,
            data.radii_halos,
        )
    else:
        spherical_r_comoving = np.asarray(data.radii_halos, dtype=np.float32)
        semi_axes_comoving = np.repeat(spherical_r_comoving[:, None], 3, axis=1)

    semi_axes_angular = get_angular_size_from_comoving(
        MODEL, semi_axes_comoving, data.redshift
    )
    semi_axes_angular *= config.search_radius

    pix_in_halos, zeta, halo_starts, halo_counts, halo_indices, projected_r_eff_2d = (
        find_pixels_in_halos(
            config.nside,
            halo_xyz,
            semi_axes_angular,
            data.eigenvectors,
            n_workers=8,
            geometry=config.halo_geometry,
        )
    )

    r_search = projected_r_eff_2d
    r_90_equiv = r_search / config.search_radius

    logger.info(
        f"r_search stats: min={r_search.min():.3e}, median={np.median(r_search):.3e}, max={r_search.max():.3e}"
    )
    logger.info(f"pixel-halo pairs: {len(pix_in_halos):,}")
    logger.info(f"zeta bytes: {zeta.nbytes / 1e9:.2f} GB")
    logger.info(f"weights expected bytes (float64): {len(zeta) * 8 / 1e9:.2f} GB")

    if use_weights:
        weights = compute_weights(
            config,
            pixel_indices=pix_in_halos,
            distances=zeta,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            r_90=r_90_equiv,
            particle_counts=data.particle_counts,
            method="vectorized",
        )
    else:
        weights = np.ones(len(pix_in_halos), dtype=np.float32)

    log_M = np.log10(data.m_halos)
    log_distances = np.log(zeta + 1e-40)

    if len(log_distances) > 0:
        logger.info(
            f"""log_distances stats: min={log_distances.min():.3e}, \n
        median={np.median(log_distances):.3e}, max={log_distances.max():.3e}"""
        )

    # Create halo index array to map each pixel to its halo's mass
    log_M_values = log_M[halo_indices]
    z_values = np.full_like(zeta, data.redshift, dtype=np.float32)

    y_values = interpolator.eval_for_logs(log_distances, z_values, log_M_values)  # pyright: ignore[reportUnknownVariableType, reportCallIssue]

    y_values *= weights

    y_map = np.zeros(hp.nside2npix(config.nside), dtype=np.float32)
    np.add.at(y_map, pix_in_halos, y_values)

    # y_per_halo = np.bincount(
    #     halo_indices, weights=y_values, minlength=len(data.m_halos)
    # ).astype(np.float32)

    # # for plotting y vs r/r200
    # profile_n_halos = 1000
    # profile_seed = 123
    # profile_logM_centers = [12, 12.5, 13, 13.5, 14, 14.5]
    # profile_logM_halfwidth = 0.15

    # radial_cfg = RadialProfileBuilderConfig(
    #     r_search=r_search,
    #     num_halos=profile_n_halos,
    #     seed=profile_seed,
    #     log_m_centers=profile_logM_centers,
    #     log_m_halfwidth=profile_logM_halfwidth,
    # )
    # builder = RadialProfileBuilder(
    #     radial_cfg,
    #     data,
    #     pix_in_halos,
    #     halo_starts,
    #     halo_counts,
    #     zeta,
    #     interpolator,
    #     MODEL,
    #     y_values=y_values,
    # )
    # radial_profiles = builder.build(y_map)
    # radial_profiles_isolated = builder.build_isolated()

    return y_map  # , y_per_halo, radial_profiles, radial_profiles_isolated


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
    y_map = paint_y(
        config,
        data,
        interpolator=interpolator,
        use_weights=use_weights,
    )
    display_map_statistics(y_map)
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() or output_path.is_symlink():
            logger.warning(f"Output file exists, overwriting: {output_path}")
            output_path.unlink()
        af = asdf.AsdfFile(
            {
                "header": {
                    "nside": config.nside,
                    "nest": True,
                    "redshift": data.redshift,
                    "search_radius_multiplier": config.search_radius,
                    "bin_width": config.weight_bin_width,
                    "halo_geometry": config.halo_geometry,
                    "healpix_file": str(healcounts_file_1)
                    if healcounts_file_1 is not None
                    else None,
                    "halo_catalog_file": str(halo_dir)
                    if halo_dir is not None
                    else None,
                },
                "data": {
                    "y_map": y_map,
                    # "halo_index": np.arange(len(data.m_halos), dtype=np.int64),
                    # "theta_halo": data.theta,
                    # "phi_halo": data.phi,
                    # "y_per_halo": y_per_halo,
                    # "halo_M": data.m_halos,
                    # "r98_halo": data.radii_halos,
                    # "radial_profile": [rp.as_dict() for rp in radial_profile],
                    # "radial_profile_isolated": [rp.as_dict() for rp in radial_profile_isolated],
                },
            }
        )
        af.write_to(output_path)
        logger.info(f"Saved to {output_path}")

    # output_stub = str(Path(output_file).with_suffix("")) if output_file else None
    # vis = Visualizer(config.nside, output_stub)
    # vis.plot_ra_dec(y_map, PlotConfig.standard(), sim_data=data, filename_suffix="clean_tight", zoom_scale=12.0, show_halo_centers=False)
    # vis.plot_ra_dec(y_map, PlotConfig.standard(), sim_data=data, filename_suffix="clean_tight_x2", zoom_scale=24.0, show_halo_centers=False)
    # vis.plot_Y_vs_M(data, y_per_halo)
    # vis.plot_Y_vs_R200(radial_profile_isolated)
    # vis.plot_Y_vs_R200(radial_profile, suffix="y_vs_r200_after_painting")
    # vis.visualize_y_map(y_map)
