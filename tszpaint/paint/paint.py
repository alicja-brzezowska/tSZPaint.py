import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numba import jit, prange

from tszpaint.config import (
    HALO_CATALOGS_PATH,
    HEALCOUNTS_PATH,
    INTERPOLATORS_PATH,
)
from tszpaint.converters import convert_rad_to_cart
from tszpaint.decorators import timer
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.tree import build_tree, query_tree
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
nbins = 20  # NOTE: THINK how many bins!

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


@jit(nopython=True, parallel=True, cache=True)
def weights_mechanism(
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    init_weights: np.ndarray,  # weights based on particle counts
):
    """compute normalized weights for each particle contribution within halos."""

    EMPTY_FRACTION_THRESHOLD = (
        2.0 / 3.0
    )  # fall back to isotropic if >2/3 of bin is empty

    N_halos = len(theta_200)
    weights = np.ones_like(distances, dtype=np.float64)
    bin_edges = np.linspace(0.0, float(N), nbins + 1)

    for h in prange(N_halos):
        start = halo_starts[h]
        count = halo_counts[h]

        if count == 0:
            continue

        d = distances[start : start + count]
        w = init_weights[start : start + count]

        # Normalize the angular distance
        x = d / theta_200[h]

        # For each particle in halo, find which bin it belongs to
        bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
        bin_ids = np.minimum(bin_ids, nbins - 1)

        # Count total pixels and non-empty pixels per bin
        bin_pixel_tot = np.zeros(nbins, dtype=np.float64)
        bin_weight_tot = np.zeros(nbins, dtype=np.float64)
        bin_nonempty = np.zeros(nbins, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            bin_pixel_tot[b] += 1.0
            bin_weight_tot[b] += w[i]
            if w[i] > 1e-12:
                bin_nonempty[b] += 1.0

        normalization_per_bin = np.ones(nbins, dtype=np.float64)
        for b in range(nbins):
            if bin_pixel_tot[b] > 0.0:
                empty_frac = 1.0 - bin_nonempty[b] / bin_pixel_tot[b]
                if empty_frac > EMPTY_FRACTION_THRESHOLD:
                    # Too sparse: use uniform weights (isotropic profile)
                    normalization_per_bin[b] = 1.0
                elif bin_weight_tot[b] > 0.0:
                    normalization_per_bin[b] = bin_pixel_tot[b] / bin_weight_tot[b]

        # Apply: for sparse bins, w*1.0 still uses init_weights, so override to uniform
        result = np.ones(count, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            empty_frac = (
                1.0 - bin_nonempty[b] / bin_pixel_tot[b]
                if bin_pixel_tot[b] > 0.0
                else 1.0
            )
            if empty_frac > EMPTY_FRACTION_THRESHOLD:
                result[i] = 1.0  # uniform weight → isotropic profile
            else:
                result[i] = w[i] * normalization_per_bin[b]

        weights[start : start + count] = result

    return weights


@timer
def compute_weights(
    pixel_indices: np.ndarray,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    particle_counts: np.ndarray,
):
    """
    Calculate the proportional weights for pixels based on particle counts.
    """
    counts = particle_counts[pixel_indices]
    init_weights = np.power(counts + 1e-10, 5.0 / 3.0)  # propto N_particles^(5/3)

    weights = weights_mechanism(
        distances,
        halo_starts,
        halo_counts,
        theta_200,
        init_weights,
    )

    return weights


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
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    use_weights: bool = True,
    verbose: bool = False,
):
    """
    Paint y-map wrapper
    """
    return paint_y(
        config,
        halo_theta,
        halo_phi,
        M_halos,
        particle_counts,
        interpolator,
        z,
        nside,
        use_weights,
    )


def plot_zoom(y_map, nside, halo_theta=None, halo_phi=None, outpng="y_map_zoom.png"):
    """Zoom to brightest pixel."""
    ipix = int(np.nanargmax(y_map))
    theta, phi = hp.pix2ang(nside, ipix, nest=True)
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)

    pix_arcmin = hp.nside2resol(nside, arcmin=True)
    reso = pix_arcmin / 6.0
    xsize = 4000

    plt.figure(figsize=(10, 10))
    hp.gnomview(
        np.log10(y_map + 10**-20),
        rot=[lon, lat],
        xsize=xsize,
        reso=reso,
        nest=True,
        title=f"Zoomed (nside={nside})",
        unit="log10(y)",
        hold=True,
    )
    hp.graticule()

    # Plot halo centers to check consistency with healpix pixels;
    if halo_theta is not None and halo_phi is not None:
        halo_lon = np.degrees(halo_phi)
        halo_lat = 90.0 - np.degrees(halo_theta)
        hp.projscatter(
            halo_lon,
            halo_lat,
            lonlat=True,
            marker="x",
            color="red",
            s=20,
            alpha=0.7,
            label="Halo centers",
        )
        plt.legend()

    plt.savefig(outpng, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")


def plot_ra_dec(
    y_map, nside, halo_theta=None, halo_phi=None, outpng="y_map_zoom_radec.png"
):
    """Zoom to specific RA/Dec."""
    ra_deg = 140.609
    dec_deg = -0.047

    pix_arcmin = hp.nside2resol(nside, arcmin=True)
    reso = pix_arcmin / 6.0
    xsize = 4000

    plt.figure(figsize=(10, 10))
    hp.gnomview(
        np.log10(y_map + 10**-20),
        rot=[ra_deg, dec_deg],
        xsize=xsize,
        reso=reso,
        nest=True,
        title=f"Painting output; merged 3 files (RA={ra_deg:.3f} Dec={dec_deg:.3f})",
        unit="log10(y)",
        hold=True,
    )
    hp.graticule()

    if halo_theta is not None and halo_phi is not None:
        halo_lon = np.degrees(halo_phi)
        halo_lat = 90.0 - np.degrees(halo_theta)
        hp.projscatter(
            halo_lon,
            halo_lat,
            lonlat=True,
            marker="x",
            color="red",
            s=20,
            alpha=0.7,
            label="Halo centers",
        )
        plt.legend()

    plt.savefig(outpng, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")


def plot_Y_vs_M(M_halos, Y_per_halo, outpng="Y_vs_M.png", nbins_plot=80):
    """Binned log-log plot of integrated Y vs halo mass with weighted linear fit."""
    mask = (
        (M_halos > 0)
        & (Y_per_halo > 0)
        & np.isfinite(M_halos)
        & np.isfinite(Y_per_halo)
    )
    logM = np.log10(M_halos[mask])
    logY = np.log10(Y_per_halo[mask])

    bins = np.linspace(logM.min(), logM.max(), nbins_plot + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    y_mean = np.full(nbins_plot, np.nan)
    y_err = np.full(nbins_plot, np.nan)

    for i in range(nbins_plot):
        in_bin = (logM >= bins[i]) & (logM < bins[i + 1])
        n = in_bin.sum()
        if n < 5:
            continue
        vals = logY[in_bin]
        y_mean[i] = np.mean(vals)
        y_err[i] = np.std(vals, ddof=1) / np.sqrt(n)

    good = np.isfinite(y_mean) & np.isfinite(y_err) & (y_err > 0)

    # Weighted linear fit: logY = intercept + slope * logM
    # np.polyfit w = 1/sigma for WLS (it squares internally)
    coeffs, cov = np.polyfit(
        centers[good],
        y_mean[good],
        1,
        w=1.0 / y_err[good],
        cov=True,
    )
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0, 0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        centers[good],
        y_mean[good],
        yerr=y_err[good],
        fmt="o",
        ms=4,
        capsize=2,
        label=r"Binned mean $\pm\,\sigma/\sqrt{N}$",
    )

    xx = np.linspace(centers[good].min(), centers[good].max(), 200)
    ax.plot(
        xx,
        intercept + slope * xx,
        "-",
        lw=2,
        label=f"Fit: slope = {slope:.3f} $\\pm$ {slope_err:.3f}",
    )

    ax.set_xlabel(r"$\log_{10}\,M_{200c}\;[M_\odot]$")
    ax.set_ylabel(r"$\log_{10}\,Y_{200}\;[\mathrm{sr}]$")
    ax.set_title("Integrated Compton-$y$ vs Halo Mass")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")
    print(f"  Slope = {slope:.3f} +/- {slope_err:.3f}")
    print(f"  Intercept = {intercept:.3f}")


def paint_abacus(
    config: PainterConfig,
    halo_dir,
    healcounts_file_1,
    healcounts_file_2,
    healcounts_file_3,
    output_file="y_map_abacus.fits",
    nside=NSIDE,
    interpolator_path=JAX_PATH,
    method="vectorized",
    use_weights=True,
):
    """
    Paint the y-compton map using Abacus halo catalogs and heal-counts.
    """
    halo_theta, halo_phi, M_halos, particle_counts, redshift = load_abacus_for_painting(
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
        halo_theta=halo_theta,
        halo_phi=halo_phi,
        M_halos=M_halos,
        particle_counts=particle_counts,
        interpolator=interpolator,
        z=redshift,
        nside=nside,
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
            halo_theta,
            halo_phi,
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
        halo_dir=str(halo_dir),
        healcounts_file_1=str(healcounts_file1),
        healcounts_file_2=str(healcounts_file2),
        healcounts_file_3=str(healcounts_file3),
        output_file=output_file,
        method="vectorized",
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
