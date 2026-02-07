import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from numba import jit, prange
import asdf
import matplotlib.pyplot as plt
import math
import time
import psutil
import os

from tszpaint.y_profile import (
    Battaglia16ThermalSZProfile,
    create_battaglia_profile,
    compute_R_delta,
    angular_size,
)
from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.config import DATA_PATH, ABACUS_DATA_PATH, INTERPOLATORS_PATH, HALO_CATALOGS_PATH, HEALCOUNTS_PATH
from tszpaint.abacus_loader import load_abacus_for_painting

# HEALPix
NSIDE = 1024
Z = 0.5 # FOR MOCK DATA

MODEL = create_battaglia_profile()
PYTHON_PATH = INTERPOLATORS_PATH / "y_values_python.pkl"
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
JULIA_PATH = INTERPOLATORS_PATH / "battaglia_interpolation.jld2"
HALO_CATALOGS_FILE_PATH = HALO_CATALOGS_PATH / "lightcone_halo_info_000.asdf"  
HEALCOUNTS_FILE_PATH = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0635-0640.asdf"

PAINT_METHOD = "vectorized"

# NOTE: TIMING
def _mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def _log(msg, t0=None):
    mem = _mem_mb()
    if t0 is not None:
        print(f"  {msg}: {time.perf_counter() - t0:.3f}s | mem: {mem:.0f} MB")
    else:
        print(f"  {msg} | mem: {mem:.0f} MB")
    return time.perf_counter()


def create_mock_particle_data(NPIX, m):
    """Create mock particle count datasets for tests."""
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
    """
    Build a 3D KDTree of HEALPix pixels.
    Need to convert to cartesian coordinates; as no angular KDTree in scipy.
    """
    npix = hp.nside2npix(nside)
    pix_indices = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix_indices)
    pix_xyz = convert_rad_to_cart(theta, phi)
    tree = cKDTree(pix_xyz)
    return tree, pix_xyz, pix_indices # NOTE: do i need pix_xyz?


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
    # list of arrays; each array contains pixel indices for that halo (in HealPix map)

    # define and return halo_starts and halo_counts for efficient weighting
    halo_counts = np.array([len(p) for p in pix_in_halos], dtype=np.int64) 
    halo_starts = np.zeros(N_halos, dtype=np.int64)
    halo_starts[1:] = np.cumsum(halo_counts[:-1]) # cumulative sum: exclude last; as first index is 0 

    pix_in_halos = np.concatenate(
        [np.asarray(p, dtype=np.int64) for p in pix_in_halos]
    ) #flatten the array of arrays 

    # create a halo index array mapping each pixel to its halo:
    halo_indices = np.repeat(np.arange(N_halos, dtype=np.int64), halo_counts) 

    # exclude pixels not in any halo:
    particle_xyz = particle_xyz[pix_in_halos]
    halo_xyz = halo_xyz[halo_indices]

    # obtain the angular distances between particles and the halo centers:
    chord_distances = np.linalg.norm(particle_xyz - halo_xyz, axis=1)       # NOTE: maybe could use directly the query?
    distances = 2.0 * np.arcsin(np.clip(chord_distances / 2.0, -1.0, 1.0))
    
    return pix_in_halos, distances, halo_starts, halo_counts, halo_indices 


@jit(nopython=True, parallel=True, cache=True)


def compute_weights(
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    init_weights: np.ndarray, # weights based on particle counts
    N: float,
    nbins: int,
):
    """compute normalized weights for each particle contribution within halos."""

    N_halos = len(theta_200)
    weights = np.ones_like(distances, dtype=np.float64)
    bin_edges = np.linspace(0.0, N, nbins + 1) 

    for h in prange(N_halos):
        start = halo_starts[h]
        count = halo_counts[h]

        if count == 0: # NOTE: check if this is actually an issue
            continue

        d = distances[start : start + count]
        w = init_weights[start : start + count] # NOTE: .copy() if crashes

        # Normalize the angular distance
        x = d / theta_200[h]

        # For each particle in halo, find which bin it belongs to
        bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
        bin_ids = np.minimum(bin_ids, nbins - 1)

        # Total pixel counts and weight sums for each bin for normalization
        # NOTE: think how to vectorize this better
        bin_pixel_tot = np.zeros(nbins, dtype=np.float64)
        bin_weight_tot = np.zeros(nbins, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            bin_pixel_tot[b] += 1.0
            bin_weight_tot[b] += w[i]


        normalization_per_bin = np.ones(nbins, dtype=np.float64)
        for b in range(nbins):
            if bin_weight_tot[b] > 0.0:
                normalization_per_bin[b] = bin_pixel_tot[b] / bin_weight_tot[b]

        weights[start : start + count] = w * normalization_per_bin[bin_ids]

    return weights


def compute_initial_weights(
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
    Calculate the proportional weights for pixels based on particle counts. 
    """
    counts = particle_counts[pixel_indices]
    init_weights = np.power(counts + 1e-10, 5.0 / 3.0) # propto N_particles^(5/3)

    weights = compute_weights(
        distances, halo_starts, halo_counts, theta_200, init_weights, N, nbins
    )

    return weights


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
    use_weights: bool = True,
    verbose: bool = False,
):
    if verbose:
        t0 = _log(f"Starting vectorized paint: {len(M_halos)} halos, nside={nside}")
        print(f"  particle_counts: {particle_counts.nbytes/1e6:.1f}MB, dtype={particle_counts.dtype}")

    # Build and query tree
    if verbose: t1 = time.perf_counter()
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)
    if verbose:
        t1 = _log(f"build_tree (npix={npix}, pix_xyz={pix_xyz.nbytes/1e6:.1f}MB)", t1)

    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)
    theta_200 = compute_theta_200(MODEL, M_halos, Z=z, delta=200)

    if verbose: t1 = time.perf_counter()
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = query_tree(
        halo_xyz=halo_xyz,
        theta_200=theta_200,
        particle_tree=tree,
        particle_xyz=pix_xyz,
        N=N,
    )
    if verbose:
        t1 = _log(f"query_tree ({len(pix_in_halos):,} pixel-halo pairs)", t1)

    if use_weights:
        if verbose: t1 = time.perf_counter()
        weights = compute_initial_weights(
            pixel_indices=pix_in_halos,
            distances=distances,
            halo_starts=halo_starts,
            halo_counts=halo_counts,
            theta_200=theta_200,
            particle_counts=particle_counts,
            N=N,
            nbins=nbins,
        )
        if verbose:
            t1 = _log("compute_weights", t1)
    else:
        weights = np.ones(len(pix_in_halos), dtype=np.float64)

    if verbose: t1 = time.perf_counter()
    log_M = np.log10(M_halos)
    log_distances = np.log(distances + 1e-40)

    # Create halo index array to map each pixel to its halo's mass
    log_M_values = log_M[halo_indices]
    z_values = np.full_like(distances, z, dtype=float)
    if verbose:
        t1 = _log("prepare arrays", t1)

    if verbose: t1 = time.perf_counter()
    y_values = interpolator.eval_for_logs(log_distances, z_values, log_M_values)
    if verbose:
        t1 = _log("interpolation", t1)

    if verbose: t1 = time.perf_counter()
    y_values_with_weight = y_values * weights

    y_map = np.zeros(npix, dtype=float)
    np.add.at(y_map, pix_in_halos, y_values_with_weight)
    if verbose:
        t1 = _log("accumulate", t1)
        _log("Done", t0)

    return y_map


# For large datasets, try painting in chunks
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
    chunk_size: int = 100,
    use_weights: bool = True,
    verbose: bool = False,  # TIMING
):
    if verbose:
        t0 = _log(f"Starting: {len(M_halos)} halos, nside={nside}")
        print(f"  particle_counts: {particle_counts.nbytes/1e6:.1f}MB, dtype={particle_counts.dtype}")

    if verbose: t1 = time.perf_counter()
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)
    y_map = np.zeros(npix, dtype=float)
    if verbose:
        t1 = _log(f"build_tree (npix={npix}, pix_xyz={pix_xyz.nbytes/1e6:.1f}MB)", t1)
        print(f"  y_map: {y_map.nbytes/1e6:.1f}MB")

    N_halos = len(M_halos)
    n_chunks = math.ceil(N_halos / chunk_size)

    total_query = total_weight = total_interp = 0.0
    total_pixels = 0

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, N_halos)

        halo_xyz_chunk = convert_rad_to_cart(
            halo_theta[start_idx:end_idx], halo_phi[start_idx:end_idx]
        )
        M_chunk = M_halos[start_idx:end_idx]
        theta_200_chunk = compute_theta_200(MODEL, M_chunk, Z=z, delta=200)

        if verbose: tq = time.perf_counter()
        (
            pix_in_halos,
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
        if verbose: total_query += time.perf_counter() - tq

        if len(pix_in_halos) == 0:
            continue

        total_pixels += len(pix_in_halos)

        if use_weights:
            if verbose: tw = time.perf_counter()
            weights = compute_initial_weights(
                pixel_indices=pix_in_halos,
                distances=distances_flat,
                halo_starts=halo_starts,
                halo_counts=halo_counts,
                theta_200=theta_200_chunk,
                particle_counts=particle_counts,
                N=N,
                nbins=nbins,
            )
            if verbose: total_weight += time.perf_counter() - tw
        else:
            weights = np.ones(len(pix_in_halos), dtype=np.float64)

        if verbose: ti = time.perf_counter()
        log_M_chunk = np.log10(M_chunk)
        halo_indices_chunk = np.repeat(np.arange(len(M_chunk)), halo_counts)

        log_theta = np.log(distances_flat + 1e-40)
        log_M = log_M_chunk[halo_indices_chunk]
        z_arr = np.full_like(distances_flat, z, dtype=float)
        if verbose: t_prep = time.perf_counter() - ti

        if verbose: t_interp_start = time.perf_counter()
        y_iso = interpolator.eval_for_logs(log_theta, z_arr, log_M) # interp: get y values from interpolator
        if verbose: t_interp_only = time.perf_counter() - t_interp_start

        y_weighted = y_iso * weights

        if verbose: t_accum_start = time.perf_counter() # accum: add the y values to the map
        np.add.at(y_map, pix_in_halos, y_weighted)
        if verbose:
            t_accum = time.perf_counter() - t_accum_start
            total_interp += t_prep + t_interp_only + t_accum
            # Log first chunk breakdown for insight
            if chunk_idx == 0:
                print(f"  [chunk 0 breakdown] prep={t_prep:.3f}s, interp={t_interp_only:.3f}s, accum={t_accum:.3f}s")

    if verbose:
        print(f"  --- Timing ({n_chunks} chunks, {total_pixels:,} pixel-halo pairs) ---")
        print(f"  query_tree:        {total_query:.3f}s")
        print(f"  compute_weights:   {total_weight:.3f}s")
        print(f"  interp+accumulate: {total_interp:.3f}s")
        _log("Done", t0)

    return y_map


def paint_y_mock_data(
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    N: float = 4.0,
    nbins: int = 20,
    chunk_size: int = 1000,
    method: str = "vectorized",
    use_weights: bool = True,
    verbose: bool = False,
):
    """
    Paint y-map from mock halo and particle count data.
    """
    if method == "chunked":
        return paint_y_chunked(
            halo_theta, halo_phi, M_halos, particle_counts,
            interpolator, z, nside, N, nbins, chunk_size, use_weights, verbose
        )

    return paint_y(
        halo_theta, halo_phi, M_halos, particle_counts,
        interpolator, z, nside, N, nbins, use_weights, verbose
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
    Paint the y-compton map using Abacus halo catalogs and heal-counts.
    """
    halo_theta, halo_phi, M_halos, particle_counts, redshift = load_abacus_for_painting(
        halo_dir=halo_dir,
        healcounts_file=healcounts_file,
        nside=nside,
    )

    interpolator = load_interpolator(interpolator_path)

    print(f"Painting y-map ...")
    y_map = paint_y_mock_data(
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
    halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
    healcounts_file = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"
    output_file = "y_map_abacus.fits"
    
    print(f"Painting Abacus tSZ map...")
    print(f"Halo directory: {halo_dir}")
    print(f"Healcounts file: {healcounts_file}")
    print(f"Output file: {output_file}")

    interpolator = load_interpolator(JAX_PATH)
    redshift = 0.625
    nside = 2048
    method = "vectorized"
    use_weights = True
    halo_theta, halo_phi, M_halos = create_mock_halo_catalogs(NPIX=hp.nside2npix(nside), m=np.arange(hp.nside2npix(nside)))
    _, _, particle_counts = create_mock_particle_data(NPIX=hp.nside2npix(nside), m=np.arange(hp.nside2npix(nside)))


    y_map_mock = paint_y_mock_data(
        halo_theta=halo_theta,
        halo_phi=halo_phi,
        M_halos=M_halos,
        particle_counts=particle_counts,
        interpolator=interpolator,
        z=Z,
        nside=nside,
        method=method,
        use_weights=use_weights,
        verbose=True,
    ) 
    
    y_map = paint_abacus(
        halo_dir=str(halo_dir),
        healcounts_file=str(healcounts_file),
        output_file=output_file,
        method="vectorized", 
        nside=NSIDE,
    )

    hp.mollview(y_map_mock, title="tSZ y-map on mock data (z = 0.625)", unit="y", norm="log", min=1e-12)
    hp.graticule()
    plt.savefig("y_map_mock.png", dpi=200, bbox_inches="tight")
    print("Saved visualization to y_map_abacus_mock.png")


if __name__ == "__main__":
    main()