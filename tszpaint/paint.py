import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from numba import jit, prange
import asdf
import matplotlib.pyplot as plt

from tszpaint.y_profile import (
    Battaglia16ThermalSZProfile,
    create_battaglia_profile,
    compute_R_delta,
    angular_size,
)
from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.config import DATA_PATH, ABACUS_DATA_PATH, INTERPOLATORS_PATH, HALO_CATALOGS_DATA_PATH, HEALCOUNTS_DATA_PATH
from tszpaint.abacus_loader import load_abacus_for_painting

# HEALPix
NSIDE = 8192

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"
HALO_CATALOGS_PATH = HALO_CATALOGS_DATA_PATH / "halo_info_000.asdf"  
HEALCOUNTS_PATH = HEALCOUNTS_DATA_PATH / "LightCone0_halo_heal-counts_Step0628-0634.asdf"

Z = 0.5
PAINT_METHOD = "vectorized"


def reaf_asdf_healcounts():
    """Read particle counts from AbacusSummit ASDF file."""
    halo_file = asdf.open(HEALCOUNTS_PATH)
    m = halo_file["PartCounts/PartCounts_000"]
    return m


def create_mock_particle_data(NPIX, m):
    """Create mock data for testing, mimicking Abacussummit data structure."""
    rng = np.random.default_rng(seed=28)
    baseline = 1_000_000_000
    contrast = rng.lognormal(mean=0.0, sigma=2.0, size=len(m))
    lam = baseline * contrast
    particle_counts = rng.poisson(lam=lam).astype(np.int64)
    theta, phi = hp.pix2ang(NSIDE, m)
    return theta, phi, particle_counts


def create_mock_halo_catalogs(NPIX, m):
    """Create halo-catalog mock data for testing."""
    N_halos = 20000
    rng = np.random.default_rng(123)
    halo_theta = np.pi * rng.random(N_halos)
    halo_phi = 2 * np.pi * rng.random(N_halos)
    logM = rng.uniform(15.5, 16.5, size=N_halos)
    M_halos = 10.0**logM
    return halo_theta, halo_phi, M_halos


def convert_rad_to_cart(theta, phi):
    """Given radial coordinates, convert to cartesian."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.column_stack([x, y, z])
    return xyz


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


def build_tree(nside=NSIDE):
    """Build a 3D KDTree of HEALPix pixels."""
    npix = hp.nside2npix(nside)
    pix_indices = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix_indices)
    pix_xyz = convert_rad_to_cart(theta, phi)
    tree = cKDTree(pix_xyz)
    return tree, pix_xyz, pix_indices


def query_tree(
    halo_xyz: np.ndarray,
    theta_200: np.ndarray,
    particle_tree: cKDTree,
    particle_xyz: np.ndarray,
    N: int = 4,
):
    """
    Query the tree out to N times theta_200 to find which pixels belong to which halo.
    """
    N_halos = halo_xyz.shape[0]
    search_angles = N * theta_200
    search_radii = 2.0 * np.sin(0.5 * search_angles)

    pix_in_halos = particle_tree.query_ball_point(x=halo_xyz, r=search_radii)

    halo_counts = np.fromiter(
        (len(p) for p in pix_in_halos), dtype=np.int64, count=N_halos
    )
    halo_starts = np.zeros(N_halos, dtype=np.int64)
    if N_halos > 1:
        halo_starts[1:] = np.cumsum(halo_counts[:-1])

    total = int(halo_counts.sum())
    if total == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            halo_starts,
            halo_counts,
            np.empty(0, dtype=np.int64),
        )

    pixel_indices_flat = np.concatenate(
        [np.asarray(p, dtype=np.int64) for p in pix_in_halos]
    )
    halo_indices = np.repeat(np.arange(N_halos, dtype=np.int64), halo_counts)

    pixel_xyz = particle_xyz[pixel_indices_flat]
    halo_xyz_repeated = halo_xyz[halo_indices]
    cosangs = np.clip(np.sum(pixel_xyz * halo_xyz_repeated, axis=1), -1.0, 1.0)
    distances_flat = np.arccos(cosangs)

    return pixel_indices_flat, distances_flat, halo_starts, halo_counts, halo_indices


@jit(nopython=True, parallel=True, cache=True)
def _compute_weights_numba(
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    raw_weights: np.ndarray,
    N: float,
    nbins: int,
):

    N_halos = len(theta_200)
    weights = np.ones_like(distances, dtype=np.float64)
    bin_edges = np.linspace(0.0, N, nbins + 1)

    for h in prange(N_halos):
        start = halo_starts[h]
        count = halo_counts[h]

        if count == 0:
            continue

        # Views into this halo's data
        theta_h = distances[start : start + count]
        w_h = raw_weights[start : start + count].copy()

        # Radial coordinate normalized by theta_200
        x_h = theta_h / theta_200[h]

        # Bin assignment
        bin_ids = np.searchsorted(bin_edges[1:], x_h, side="left")
        bin_ids = np.minimum(bin_ids, nbins - 1)

        # Histogram: count and sum per bin (scatter-add, must be sequential)
        bin_counts_h = np.zeros(nbins, dtype=np.float64)
        bin_sums_h = np.zeros(nbins, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            bin_counts_h[b] += 1.0
            bin_sums_h[b] += w_h[i]

        # Normalization factors per bin (loop over nbins, typically 20)
        norm_factors = np.ones(nbins, dtype=np.float64)
        for b in range(nbins):
            if bin_sums_h[b] > 0.0:
                norm_factors[b] = bin_counts_h[b] / bin_sums_h[b]

        # Apply normalization via gather (vectorized in Numba)
        weights[start : start + count] = w_h * norm_factors[bin_ids]

    return weights


def weigh_particle_contr(
    pixel_indices: np.ndarray,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    particle_counts: np.ndarray,
    N: float = 4.0,
    nbins: int = 20,
):
    """
    Compute weights for particle contributions within halos.
    """
    counts = particle_counts[pixel_indices]
    raw_weights = np.power(counts + 1e-10, 5.0 / 3.0)

    weights = _compute_weights_numba(
        distances, halo_starts, halo_counts, theta_200, raw_weights, N, nbins
    )

    return weights


def paint_y_vectorized(
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    N: float = 4.0,
    nbins: int = 20,
    use_weights: bool = True,
):

    # Build KDTree
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)

    # Convert halo positions to cartesian
    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)
    theta_200 = compute_theta_200(MODEL, M_halos, Z=z, delta=200)

    # Query tree 
    pixel_indices_flat, distances_flat, halo_starts, halo_counts, _ = query_tree(
        halo_xyz=halo_xyz,
        theta_200=theta_200,
        particle_tree=tree,
        particle_xyz=pix_xyz,
        N=N,
    )

    # TEST: Switchable compute weights
    if use_weights:
        weights = weigh_particle_contr(
            pixel_indices=pixel_indices_flat,
            distances=distances_flat,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            theta_200=theta_200,
            particle_counts=particle_counts,
            N=N,
            nbins=nbins,
        )
    else:
        weights = np.ones(len(pixel_indices_flat), dtype=np.float64)

    # Prepare mass and theta arrays for interpolation
    log_M_halos = np.log10(M_halos)
    log_theta_all = np.log(distances_flat + 1e-40)

    # Create halo index array to map each pixel to its halo's mass
    halo_indices_for_pixels = np.repeat(np.arange(len(M_halos)), halo_counts)
    log_M_all = log_M_halos[halo_indices_for_pixels]
    z_all = np.full_like(distances_flat, z, dtype=float)

    # Single interpolation call for all points
    y_iso_all = interpolator.eval_for_logs(log_theta_all, z_all, log_M_all)

    # Apply weights
    y_weighted_all = y_iso_all * weights

    # Accumulate into map
    y_map = np.zeros(npix, dtype=float)
    np.add.at(y_map, pixel_indices_flat, y_weighted_all)

    return y_map


def paint_y_chunked(
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    N: float = 4.0,
    nbins: int = 20,
    chunk_size: int = 50,
    use_weights: bool = True,
):
    """
    Paint y-map in chunks to reduce memory usage.
    1. Build KDTree of pixels.
    2. For each chunk of halos:
       a. Query tree for pixels within N×θ_200.
       b. Compute weights.
       c. Interpolate y-values.
       d. Accumulate into y-map.
    3. Return final y-map.
    """
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)
    y_map = np.zeros(npix, dtype=float)

    N_halos = len(M_halos)
    n_chunks = (N_halos + chunk_size - 1) // chunk_size

    print(f"Processing {N_halos} halos in {n_chunks} chunks of {chunk_size}...")

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, N_halos)

        # Process chunk
        halo_xyz_chunk = convert_rad_to_cart(
            halo_theta[start_idx:end_idx], halo_phi[start_idx:end_idx]
        )
        M_chunk = M_halos[start_idx:end_idx]
        theta_200_chunk = compute_theta_200(MODEL, M_chunk, Z=z, delta=200)

        # Query and paint chunk
        (
            pixel_indices_flat,
            distances_flat,
            halo_starts,
            halo_counts,
            _,
        ) = query_tree(
            halo_xyz=halo_xyz_chunk,
            theta_200=theta_200_chunk,
            particle_tree=tree,
            particle_xyz=pix_xyz,
            N=N,
        )

        if len(pixel_indices_flat) == 0:
            continue

        # TEST: Switchable Compute weights 
        if use_weights:
            weights = weigh_particle_contr(
                pixel_indices=pixel_indices_flat,
                distances=distances_flat,
                halo_starts=halo_starts,
                halo_counts=halo_counts,
                theta_200=theta_200_chunk,
                particle_counts=particle_counts,
                N=N,
                nbins=nbins,
            )
        else:
            weights = np.ones(len(pixel_indices_flat), dtype=np.float64)

        # Interpolate
        log_M_chunk = np.log10(M_chunk)
        halo_indices_chunk = np.repeat(np.arange(len(M_chunk)), halo_counts)
        
        log_theta = np.log(distances_flat + 1e-40)
        log_M = log_M_chunk[halo_indices_chunk]
        z_arr = np.full_like(distances_flat, z, dtype=float)

        y_iso = interpolator.eval_for_logs(log_theta, z_arr, log_M)
        y_weighted = y_iso * weights

        # Accumulate
        np.add.at(y_map, pixel_indices_flat, y_weighted)

    return y_map


def paint_y(
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    N: float = 4.0,
    nbins: int = 20,
    chunk_size: int = 100,
    method: str = "vectorized",
    use_weights: bool = True,
):
    """
    Paint tSZ y-parameter map.

    Parameters
    method : str
        "vectorized" (default, faster) or "chunked" (memory-efficient).
    use_weights : bool
        If True (default), apply particle-count weighting for substructure.
        If False, use uniform weights (XGPaint-like behavior).

    Returns
    y_map : np.ndarray
        HEALPix map with y-parameter values
    """
    if method == "chunked":
        return paint_y_chunked(
            halo_theta, halo_phi, M_halos, particle_counts,
            interpolator, z, nside, N, nbins, chunk_size, use_weights
        )

    return paint_y_vectorized(
        halo_theta, halo_phi, M_halos, particle_counts,
        interpolator, z, nside, N, nbins, use_weights
    )


def paint_abacus(
    halo_dir,
    healcounts_file,
    output_file="y_map_abacus.fits",
    nside=NSIDE,
    interpolator_path=JAX_PATH,
    method="vectorized",
    use_weights=True,
):
    """
    Paint tSZ y-map from real Abacus simulation data.

    Parameters
    ----------
    halo_dir : str or Path
        Directory containing header.asdf and halo_info.asdf.
    healcounts_file : str or Path
        Path to the healcounts ASDF file.
    output_file : str
        Output FITS file path.
    nside : int
        HEALPix resolution.
    interpolator_path : Path
        Path to the Battaglia interpolator pickle.
    method : str
        "vectorized" or "chunked".
    use_weights : bool
        Whether to use particle-count weighting.

    Returns
    -------
    y_map : np.ndarray
        The painted y-parameter HEALPix map.
    """
    print(f"Loading Abacus data from {halo_dir}...")
    halo_theta, halo_phi, M_halos, particle_counts, redshift = load_abacus_for_painting(
        halo_dir=halo_dir,
        healcounts_file=healcounts_file,
        nside=nside,
    )

    print(f"Loaded {len(M_halos)} halos at z={redshift:.3f}")
    print(f"Mass range: {M_halos.min():.2e} - {M_halos.max():.2e} Msun")

    interpolator = load_interpolator(interpolator_path)

    print(f"Painting y-map with method='{method}', use_weights={use_weights}...")
    y_map = paint_y(
        halo_theta=halo_theta,
        halo_phi=halo_phi,
        M_halos=M_halos,
        particle_counts=particle_counts,
        interpolator=interpolator,
        z=redshift,
        nside=nside,
        method=method,
        use_weights=use_weights,
    )

    print(f"\nMap statistics:")
    print(f"  Min: {y_map.min():.3e}")
    print(f"  Max: {y_map.max():.3e}")
    print(f"  Mean: {y_map.mean():.3e}")
    print(f"  Non-zero pixels: {np.sum(y_map > 0)}/{len(y_map)}")

    if output_file:
        hp.write_map(output_file, y_map, overwrite=True)
        print(f"Saved to {output_file}")

    return y_map


def main():
    # Run actual Abacus painting
    halo_dir = ABACUS_DATA_PATH / "halos" / "z0.625"
    healcounts_file = HEALCOUNTS_DATA_PATH / "LightCone0_halo_heal-counts_Step0628-0634.asdf"
    output_file = "y_map_abacus.fits"
    
    print(f"Painting Abacus tSZ map...")
    print(f"Halo directory: {halo_dir}")
    print(f"Healcounts file: {healcounts_file}")
    print(f"Output file: {output_file}")
    
    y_map = paint_abacus(
        halo_dir=str(halo_dir),
        healcounts_file=str(healcounts_file),
        output_file=output_file,
        nside=NSIDE,
    )
    
    # Also create a visualization
    hp.mollview(y_map, title="Abacus tSZ y-map", unit="y", norm="log", min=1e-12)
    hp.graticule()
    plt.savefig("y_map_abacus.png", dpi=200, bbox_inches="tight")
    print("Saved visualization to y_map_abacus.png")


if __name__ == "__main__":
    main()