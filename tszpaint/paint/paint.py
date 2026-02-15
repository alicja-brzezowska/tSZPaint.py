from pathlib import Path

import healpy as hp
import numpy as np
from loguru import logger

from tszpaint.config import (
    HALO_CATALOGS_PATH,
    HEALCOUNTS_PATH,
    INTERPOLATORS_PATH,
)
from tszpaint.converters import convert_rad_to_cart
from tszpaint.paint.abacus_loader import SimulationData, load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.tree import build_tree, query_tree
from tszpaint.paint.visualize import Visualizer
from tszpaint.paint.weights import compute_weights
from tszpaint.y_profile.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile.y_profile import (
    Battaglia16ThermalSZProfile,
    angular_size,
    compute_R_delta,
    create_battaglia_profile,
)

# HEALPix
NSIDE = 8192
Z = 0.5  # FOR MOCK DATA
N = 2  # Multiple of theta_200 to search
N_BINS = 20  # NOTE: THINK how many bins!

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"


def compute_theta_200(
    model: Battaglia16ThermalSZProfile,
    M_halos: np.ndarray,
    Z: float = 0.5,
    delta: int = 200,
):
    """Compute θ_200 (angular radius) for each halo."""
    R_200 = compute_R_delta(model, M_halos, Z, delta=delta)
    return angular_size(model, R_200, Z)


def load_interpolator(path: Path = JAX_PATH):
    return BattagliaLogInterpolator.from_pickle(path)


def paint_y(
    config: PainterConfig,
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    use_weights: bool = True,
):
    logger.info(f"Starting vectorized paint: {len(M_halos)} halos, nside={nside}")
    logger.info(
        f"  particle_counts: {particle_counts.nbytes / 1e6:.1f}MB, dtype={particle_counts.dtype}"
    )

    # Build and query tree
    tree, pix_xyz, pix_indices = build_tree(config)
    npix = len(pix_indices)

    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)
    theta_200 = compute_theta_200(MODEL, M_halos, Z=z, delta=200)

    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = query_tree(
        config=config,
        halo_xyz=halo_xyz,
        theta_200=theta_200,
        particle_tree=tree,
        particle_xyz=pix_xyz,
    )

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
    z_values = np.full_like(distances, z, dtype=float)

    y_values = interpolator.eval_for_logs(log_distances, z_values, log_M_values)

    y_values_with_weight = y_values * weights

    y_map = np.zeros(npix, dtype=float)
    np.add.at(y_map, pix_in_halos, y_values_with_weight)

    y_per_halo = np.bincount(
        halo_indices, weights=y_values_with_weight, minlength=len(M_halos)
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
        data.redshift,
        config.nside,
        use_weights,
    )


def paint_abacus(
    config: PainterConfig,
    halo_dir: Path,
    healcounts_file_1: Path,
    healcounts_file_2: Path,
    healcounts_file_3: Path,
    output_file: str = "y_map_abacus.fits",
    nside: int = NSIDE,
    interpolator_path: Path = JAX_PATH,
    use_weights: bool = True,
):
    """
    Paint the y-compton map using Abacus halo catalogs and heal-counts.
    """
    data = load_abacus_for_painting(
        halo_dir=halo_dir,
        healcounts_file_1=healcounts_file_1,
        healcounts_file_2=healcounts_file_2,
        healcounts_file_3=healcounts_file_3,
        nside=nside,
    )

    interpolator = load_interpolator(interpolator_path)

    logger.info("Painting y-map ...")
    y_map, y_per_halo = paint_y_wrapper(
        config,
        data,
        interpolator=interpolator,
        use_weights=use_weights,
    )

    logger.info("\nMap statistics:")
    logger.info(f"  Min: {y_map.min():.3e}")
    logger.info(f"  Max: {y_map.max():.3e}")
    logger.info(f"  Mean: {y_map.mean():.3e}")
    logger.info(f"  Non-zero pixels: {np.sum(y_map > 0)}/{len(y_map)}")

    if output_file:
        hp.write_map(output_file, y_map, overwrite=True, nest=True)
        logger.info(f"Saved to {output_file}")

        vis = Visualizer(data, y_map, y_per_halo, config.nside)
        vis.plot_ra_dec(output_file.replace(".fits", "_zoom_radec.png"))
        vis.plot_Y_vs_M(output_file.replace(".fits", "_y_vs_m.png"))

    return y_map


def main():
    halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
    healcounts_file1 = (
        HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"
    )
    healcounts_file2 = (
        HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0677-0682.asdf"
    )
    healcounts_file3 = (
        HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0665-0670.asdf"
    )
    output_file = "y_map_abacus.fits"

    logger.info("Painting Abacus tSZ map...")
    logger.info(f"Halo directory: {halo_dir}")
    logger.info(f"Healcounts file 1: {healcounts_file1}")
    logger.info(f"Healcounts file 2: {healcounts_file2}")
    logger.info(f"Healcounts file 3: {healcounts_file3}")
    logger.info(f"Output file: {output_file}")
    config = PainterConfig(NSIDE, N, N_BINS)

    paint_abacus(
        config,
        halo_dir=halo_dir,
        healcounts_file_1=healcounts_file1,
        healcounts_file_2=healcounts_file2,
        healcounts_file_3=healcounts_file3,
        output_file=output_file,
        nside=NSIDE,
    )


if __name__ == "__main__":
    main()
