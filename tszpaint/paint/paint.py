import healpy as hp
import matplotlib.pyplot as plt
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
from tszpaint.paint.visualize import plot_ra_dec, plot_Y_vs_M
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


def load_interpolator(path=JAX_PATH):
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

    Y_per_halo = np.bincount(
        halo_indices, weights=y_values_with_weight, minlength=len(M_halos)
    )

    return y_map, Y_per_halo, M_halos


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
    halo_dir,
    healcounts_file_1,
    healcounts_file_2,
    healcounts_file_3,
    output_file="y_map_abacus.fits",
    nside=NSIDE,
    interpolator_path=JAX_PATH,
    use_weights=True,
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

    print("Painting y-map ...")
    y_map, Y_per_halo, M_halos_out = paint_y_wrapper(
        config,
        data,
        interpolator=interpolator,
        use_weights=use_weights,
    )

    print("\nMap statistics:")
    print(f"  Min: {y_map.min():.3e}")
    print(f"  Max: {y_map.max():.3e}")
    print(f"  Mean: {y_map.mean():.3e}")
    print(f"  Non-zero pixels: {np.sum(y_map > 0)}/{len(y_map)}")

    if output_file:
        hp.write_map(output_file, y_map, overwrite=True, nest=True)
        print(f"Saved to {output_file}")

        plot_ra_dec(
            y_map,
            nside,
            data.theta,
            data.phi,
            output_file.replace(".fits", "_zoom_radec.png"),
        )

    if Y_per_halo is not None:
        plot_Y_vs_M(
            M_halos_out,
            Y_per_halo,
            output_file.replace(".fits", "_Y_vs_M.png")
            if output_file
            else "Y_vs_M.png",
        )

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

    print("Painting Abacus tSZ map...")
    print(f"Halo directory: {halo_dir}")
    print(f"Healcounts file 1: {healcounts_file1}")
    print(f"Healcounts file 2: {healcounts_file2}")
    print(f"Healcounts file 3: {healcounts_file3}")
    print(f"Output file: {output_file}")
    config = PainterConfig(NSIDE, N, N_BINS)

    # interpolator = load_interpolator(JAX_PATH)
    # redshift = 0.625
    # nside = 2048
    # method = "vectorized"
    # use_weights = True
    # halo_theta, halo_phi, M_halos = create_mock_halo_catalogs(NPIX=hp.nside2npix(nside), m=np.arange(hp.nside2npix(nside)))
    # _, _, particle_counts = create_mock_particle_data(NPIX=hp.nside2npix(nside), m=np.arange(hp.nside2npix(nside)))

    #    y_map_mock = paint_y_mock_data(
    #        halo_theta=halo_theta,
    #        halo_phi=halo_phi,
    #        M_halos=M_halos,
    #        particle_counts=particle_counts,
    #        interpolator=interpolator,
    #        z=Z,
    #        nside=nside,
    #        method=method,
    #        use_weights=use_weights,
    #        verbose=True,
    #    )

    #    hp.mollview(y_map_mock, title="tSZ y-map on mock data (z = 0.625)", unit="y", norm="log", min=1e-12)
    #    hp.graticule()
    #    plt.savefig("y_map_mock.png", dpi=200, bbox_inches="tight")
    #    print("Saved visualization to y_map_abacus_mock.png")

    y_map = paint_abacus(
        config,
        halo_dir=str(halo_dir),
        healcounts_file_1=str(healcounts_file1),
        healcounts_file_2=str(healcounts_file2),
        healcounts_file_3=str(healcounts_file3),
        output_file=output_file,
        nside=NSIDE,
    )

    hp.mollview(
        y_map,
        title="tSZ y-map on real data (z = 0.542)",
        unit="y",
        norm="log",
        min=1e-12,
        nest=True,
    )
    hp.graticule()
    plt.savefig("y_map_abacus_real.png", dpi=200, bbox_inches="tight")
    print("Saved visualization to y_map_abacus_real.png")


if __name__ == "__main__":
    main()
