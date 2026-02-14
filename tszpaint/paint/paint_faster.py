import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from tszpaint.config import DATA_PATH, INTERPOLATORS_PATH
from tszpaint.converters import convert_rad_to_cart
from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile import (
    Battaglia16ThermalSZProfile,
    angular_size,
    compute_R_delta,
    create_battaglia_profile,
)

NSIDE = 1024

MOCK_CATALOGUE_NUM_HALOS = 20_000

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"
HALO_CATALOGS_PATH = DATA_PATH / "file_name"
Z = 0.5
PAINT_METHOD = "vectorized"

RNG_SEED = 17


def generate_mock_particle_counts(
    n_pixels: int,
    seed: int = 17,
    baseline_density: float = 1e9,
    overdensity_sigma: float = 2.0,
):
    """Create mock data for testing, mimicking Abacussummit data structure."""
    rng = np.random.default_rng(seed=seed)
    contrast = rng.lognormal(mean=0.0, sigma=overdensity_sigma, size=n_pixels)
    lam = baseline_density * contrast
    particle_counts = rng.poisson(lam=lam).astype(int)
    return particle_counts


def create_mock_halo_catalogs(n_halos: int):
    """Create halo-catalog mock data for testing."""
    rng = np.random.default_rng(RNG_SEED)
    halo_theta = np.pi * rng.random(n_halos)
    halo_phi = 2 * np.pi * rng.random(n_halos)
    logM = rng.uniform(15.5, 16.5, size=n_halos)
    m_halos = 10.0**logM
    return halo_theta, halo_phi, m_halos


def query_tree(
    halo_xyz: np.ndarray,
    theta_200: np.ndarray,
    particle_tree: cKDTree,
    particle_xyz: np.ndarray,
    N: float = 4.0,
):
    """
    Query tree for pixels within N×theta_200 of each halo.

    Returns data in CSR (Compressed Sparse Row) format for GPU efficiency:
    - Flat arrays for all (halo, pixel) pairs
    - halo_starts/counts for indexing into flat arrays
    """
    num_halos = halo_xyz.shape[0]

    # Search radii (convert angular to 3D Euclidean distance)
    search_angles = N * theta_200
    search_radii = 2.0 * np.sin(0.5 * search_angles)

    # Query tree (returns list of variable-length arrays)
    pix_in_halos = particle_tree.query_ball_point(x=halo_xyz, r=search_radii)

    # Compute CSR structure
    halo_counts = np.array([len(p) for p in pix_in_halos], dtype=np.int64)
    total_pairs = halo_counts.sum()

    if total_pairs == 0:
        return _empty_result(num_halos)

    halo_starts = np.zeros(num_halos, dtype=np.int64)
    halo_starts[1:] = np.cumsum(halo_counts[:-1])

    # Pre-allocate flat arrays (avoid concatenation overhead)
    pixel_indices_flat = np.empty(total_pairs, dtype=np.int64)
    halo_indices_flat = np.empty(total_pairs, dtype=np.int64)

    # Fill flat arrays
    for i, pixels in enumerate(pix_in_halos):
        start = halo_starts[i]
        count = halo_counts[i]
        pixel_indices_flat[start : start + count] = pixels
        halo_indices_flat[start : start + count] = i

    # Compute angular distances (vectorized)
    distances_flat = _compute_angular_distances(
        particle_xyz, pixel_indices_flat, halo_xyz, halo_indices_flat
    )

    return (
        pixel_indices_flat,
        distances_flat,
        halo_starts,
        halo_counts,
        halo_indices_flat,
    )


def _compute_angular_distances(
    particle_xyz: np.ndarray,
    pixel_indices: np.ndarray,
    halo_xyz: np.ndarray,
    halo_indices: np.ndarray,
) -> np.ndarray:
    """Compute angular distances between particles and their assigned halos."""
    # Gather particle and halo positions
    pixel_xyz = particle_xyz[pixel_indices]  # (n_pairs, 3)
    halo_xyz_repeated = halo_xyz[halo_indices]  # (n_pairs, 3)

    # Dot product (both vectors are unit vectors on sphere)
    cosangs = np.einsum("ij,ij->i", pixel_xyz, halo_xyz_repeated)
    cosangs = np.clip(cosangs, -1.0, 1.0)

    return np.arccos(cosangs)


def _empty_result(N_halos: int):
    """Return empty arrays when no pixels found."""
    return (
        np.empty(0, dtype=np.int64),  # pixel_indices_flat
        np.empty(0, dtype=np.float64),  # distances_flat
        np.zeros(N_halos, dtype=np.int64),  # halo_starts
        np.zeros(N_halos, dtype=np.int64),  # halo_counts
        np.empty(0, dtype=np.int64),  # halo_indices_flat
    )


def compute_weights(
    distances: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    raw_weights: np.ndarray,
    N: float,
    nbins: int,
) -> np.ndarray:
    """
    Vectorized computation of normalized weights for radial bins around halos.

    For each halo, normalizes particle weights within radial bins so that
    each bin contributes equally (prevents overdense regions from dominating).
    """
    # Create halo ID for each particle
    halo_ids = np.repeat(np.arange(len(theta_200)), halo_counts)

    # Compute normalized radial coordinate for all particles
    x = distances / theta_200[halo_ids]  # Distance in units of theta_200

    # Assign particles to radial bins
    bin_edges = np.linspace(0.0, N, nbins + 1)
    bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
    bin_ids = np.minimum(bin_ids, nbins - 1)

    # Create composite key: (halo_id, bin_id) for grouping
    composite_key = halo_ids * nbins + bin_ids

    # Compute bin statistics using bincount
    unique_keys, inverse_indices = np.unique(composite_key, return_inverse=True)

    bin_counts = np.bincount(inverse_indices, minlength=len(unique_keys))
    bin_sums = np.bincount(
        inverse_indices, weights=raw_weights, minlength=len(unique_keys)
    )

    # Compute normalization: count / sum for each bin
    # This makes each bin contribute equally
    normalization = np.where(bin_sums > 0, bin_counts / bin_sums, 1.0)

    # Apply normalization to each particle
    weights = raw_weights * normalization[inverse_indices]

    return weights


def compute_theta_200(
    model: Battaglia16ThermalSZProfile,
    M_halos: np.ndarray,
    Z: float = 0.5,
    delta: int = 200,
):
    """Compute θ_200 (angular radius) for each halo."""
    R_200 = compute_R_delta(model, M_halos, Z, delta=delta)
    return angular_size(model, R_200, Z)


def build_tree(nside: int = NSIDE):
    """Build a 3D KDTree of HEALPix pixels."""
    npix = hp.nside2npix(nside)
    pix_indices = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix_indices)
    pix_xyz = convert_rad_to_cart(theta, phi)
    tree = cKDTree(pix_xyz)
    return tree, pix_xyz, pix_indices


def paint_y_vectorized(
    halo_thetas: np.ndarray,
    halo_phis: np.ndarray,
    halo_masses: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    N: float = 4.0,
    nbins: int = 20,
    use_weights: bool = True,
) -> np.ndarray:
    # Build spatial index of all pixels
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)

    # Convert halo positions and compute characteristic radii
    halo_xyz = convert_rad_to_cart(halo_thetas, halo_phis)
    theta_200 = compute_theta_200(MODEL, halo_masses, Z=z, delta=200)

    # Find all (halo, pixel) pairs within N×theta_200
    pixel_indices_flat, distances_flat, halo_starts, halo_counts, _ = query_tree(
        halo_xyz=halo_xyz,
        theta_200=theta_200,
        particle_tree=tree,
        particle_xyz=pix_xyz,
        N=N,
    )

    if len(pixel_indices_flat) == 0:
        return np.zeros(npix, dtype=float)

    # Compute particle contribution weights
    counts = particle_counts[pixel_indices_flat]
    raw_weights = np.power(counts + 1e-10, 5.0 / 3.0)  # Pressure scaling

    if use_weights:
        weights = compute_weights(
            distances=distances_flat,
            halo_counts=halo_counts,
            theta_200=theta_200,
            raw_weights=raw_weights,
            N=N,
            nbins=nbins,
        )
    else:
        weights = raw_weights

    # Map halo properties to each (halo, pixel) pair
    halo_indices = np.repeat(np.arange(len(halo_masses)), halo_counts)
    log_M = np.log10(halo_masses[halo_indices])
    log_theta = np.log(distances_flat + 1e-40)  # Small offset for numerical stability
    z_arr = np.full_like(distances_flat, z, dtype=float)

    # Interpolate y-parameter from pressure profile
    y_iso = interpolator.eval_for_logs(log_theta, z_arr, log_M)

    # Apply weights and accumulate onto map
    y_weighted = y_iso * weights
    y_map = np.zeros(npix, dtype=float)
    np.add.at(y_map, pixel_indices_flat, y_weighted)

    return y_map


def main():
    n_pixels = hp.nside2npix(NSIDE)
    halo_thetas, halo_phis, halo_masses = create_mock_halo_catalogs(
        MOCK_CATALOGUE_NUM_HALOS
    )
    particle_counts = generate_mock_particle_counts(n_pixels)
