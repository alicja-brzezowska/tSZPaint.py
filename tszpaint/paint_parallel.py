"""
Parallel painting functions for HPC environments.

This module provides paint functions optimized for multi-core CPUs,
using the parallel interpolator and avoiding chunking overhead.

Usage:
    # Set thread count BEFORE importing
    import os
    os.environ['NUMBA_NUM_THREADS'] = '64'

    from tszpaint.paint_parallel import paint_y_parallel
    from tszpaint.interpolator_parallel import ParallelBattagliaInterpolator

    interpolator = ParallelBattagliaInterpolator.from_pickle(path)
    y_map = paint_y_parallel(halo_theta, halo_phi, M_halos, particle_counts, interpolator, ...)
"""

import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from numba import jit, prange
import time
import psutil
import os

from tszpaint.y_profile import (
    Battaglia16ThermalSZProfile,
    create_battaglia_profile,
    compute_R_delta,
    angular_size,
)


MODEL = create_battaglia_profile()


def _mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _log(msg, t0=None):
    mem = _mem_mb()
    if t0 is not None:
        print(f"  {msg}: {time.perf_counter() - t0:.3f}s | mem: {mem:.0f} MB")
    else:
        print(f"  {msg} | mem: {mem:.0f} MB")
    return time.perf_counter()


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


def build_tree(nside):
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
    """Query the tree to find which pixels belong to which halo."""
    N_halos = halo_xyz.shape[0]

    search_angles = N * theta_200
    search_radii = 2.0 * np.sin(0.5 * search_angles)

    pix_in_halos = particle_tree.query_ball_point(x=halo_xyz, r=search_radii)

    halo_counts = np.array([len(p) for p in pix_in_halos], dtype=np.int64)
    halo_starts = np.zeros(N_halos, dtype=np.int64)
    halo_starts[1:] = np.cumsum(halo_counts[:-1])

    pix_in_halos = np.concatenate(
        [np.asarray(p, dtype=np.int64) for p in pix_in_halos]
    )

    halo_indices = np.repeat(np.arange(N_halos, dtype=np.int64), halo_counts)

    particle_xyz_selected = particle_xyz[pix_in_halos]
    halo_xyz_selected = halo_xyz[halo_indices]

    chord_distances = np.linalg.norm(particle_xyz_selected - halo_xyz_selected, axis=1)
    distances = 2.0 * np.arcsin(np.clip(chord_distances / 2.0, -1.0, 1.0))

    return pix_in_halos, distances, halo_starts, halo_counts, halo_indices


@jit(nopython=True, parallel=True, cache=True)
def compute_weights(
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    theta_200: np.ndarray,
    init_weights: np.ndarray,
    N: float,
    nbins: int,
):
    """Compute normalized weights for each particle contribution within halos."""
    N_halos = len(theta_200)
    weights = np.ones_like(distances, dtype=np.float64)
    bin_edges = np.linspace(0.0, N, nbins + 1)

    for h in prange(N_halos):
        start = halo_starts[h]
        count = halo_counts[h]

        if count == 0:
            continue

        d = distances[start : start + count]
        w = init_weights[start : start + count]

        x = d / theta_200[h]

        bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
        bin_ids = np.minimum(bin_ids, nbins - 1)

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
    """Calculate proportional weights for pixels based on particle counts."""
    counts = particle_counts[pixel_indices]
    init_weights = np.power(counts + 1e-10, 5.0 / 3.0)

    weights = compute_weights(
        distances, halo_starts, halo_counts, theta_200, init_weights, N, nbins
    )

    return weights


def paint_y_parallel(
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator,  # ParallelBattagliaInterpolator
    z: float = 0.5,
    nside: int = 1024,
    N: float = 4.0,
    nbins: int = 20,
    use_weights: bool = True,
    verbose: bool = False,
):
    """
    Paint y-map using parallel interpolation (no chunking).

    This processes ALL halos at once, using the parallel interpolator
    to evaluate all ~27M pixel-halo pairs simultaneously across all cores.

    Parameters
    ----------
    interpolator : ParallelBattagliaInterpolator
        The parallel interpolator instance
    verbose : bool
        Print timing information

    Returns
    -------
    y_map : np.ndarray
        The Compton-y map
    """
    if verbose:
        t0 = _log(f"Starting parallel paint: {len(M_halos)} halos, nside={nside}")
        print(f"  particle_counts: {particle_counts.nbytes/1e6:.1f}MB")

    # Warmup the interpolator (JIT compile if needed)
    if verbose:
        t1 = time.perf_counter()
    interpolator.warmup()
    if verbose:
        t1 = _log("JIT warmup", t1)

    # Build tree
    if verbose:
        t1 = time.perf_counter()
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)
    if verbose:
        t1 = _log(f"build_tree (npix={npix})", t1)

    # Convert halo positions and compute angular sizes
    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)
    theta_200 = compute_theta_200(MODEL, M_halos, Z=z, delta=200)

    # Query tree for ALL halos at once
    if verbose:
        t1 = time.perf_counter()
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = query_tree(
        halo_xyz=halo_xyz,
        theta_200=theta_200,
        particle_tree=tree,
        particle_xyz=pix_xyz,
        N=N,
    )
    if verbose:
        t1 = _log(f"query_tree ({len(pix_in_halos):,} pixel-halo pairs)", t1)

    # Compute weights
    if use_weights:
        if verbose:
            t1 = time.perf_counter()
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

    # Prepare interpolation inputs (ALL at once, no chunking)
    if verbose:
        t1 = time.perf_counter()
    log_M = np.log10(M_halos)
    log_theta = np.log(distances + 1e-40)
    log_M_values = log_M[halo_indices]
    z_values = np.full_like(distances, z, dtype=np.float64)
    if verbose:
        t1 = _log("prepare arrays", t1)

    # PARALLEL INTERPOLATION - the main speedup
    if verbose:
        t1 = time.perf_counter()
    y_values = interpolator.eval_for_logs(log_theta, z_values, log_M_values)
    if verbose:
        t1 = _log("parallel interpolation", t1)

    # Apply weights and accumulate
    if verbose:
        t1 = time.perf_counter()
    y_weighted = y_values * weights
    y_map = np.zeros(npix, dtype=np.float64)
    np.add.at(y_map, pix_in_halos, y_weighted)
    if verbose:
        t1 = _log("accumulate", t1)
        _log("Done", t0)

    return y_map
