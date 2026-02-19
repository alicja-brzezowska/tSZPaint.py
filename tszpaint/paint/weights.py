from typing import Literal

import numpy as np
from numba import jit, prange

from tszpaint.logging import array_size, memory_usage, time_calls, trace_calls
from tszpaint.paint.config import PainterConfig


@jit(nopython=True, parallel=True)
def _fast_init_weights(
    particle_counts: np.ndarray, pixel_indices: np.ndarray
) -> np.ndarray:
    """JIT-compiled index lookup and power operation for init_weights."""
    init_weights = np.empty(len(pixel_indices), dtype=np.float64)
    for i in prange(len(pixel_indices)):
        idx = pixel_indices[i]
        count = particle_counts[idx]
        init_weights[i] = np.power(count + 1e-10, 5.0 / 3.0)
    return init_weights


@memory_usage
@array_size
@time_calls
@trace_calls
@jit(nopython=True, parallel=True)
def weights_mechanism(
    search_radius: float,
    n_bins: int,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    r_90: np.ndarray,
    init_weights: np.ndarray,
):
    """Compute normalized weights for each particle contribution within halos."""

    N_halos = len(r_90)
    weights = np.ones_like(distances, dtype=np.float64)
    bin_edges = np.linspace(0.0, search_radius, n_bins + 1)

    for h in prange(N_halos):
        start = halo_starts[h]
        count = halo_counts[h]

        if count == 0:
            continue

        d = distances[start : start + count]
        w = init_weights[start : start + count]

        # Normalize the angular distance
        x = d / r_90[h]

        # For each particle in halo, find which bin it belongs to
        bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
        bin_ids = np.minimum(bin_ids, n_bins - 1)

        # Count total pixels and non-empty pixels per bin
        bin_pixel_tot = np.zeros(n_bins, dtype=np.float64)
        bin_weight_tot = np.zeros(n_bins, dtype=np.float64)
        bin_nonempty = np.zeros(n_bins, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            bin_pixel_tot[b] += 1.0
            bin_weight_tot[b] += w[i]
            if w[i] > 1e-12:
                bin_nonempty[b] += 1.0

        normalization_per_bin = np.ones(n_bins, dtype=np.float64)
        for b in range(n_bins):
            if bin_pixel_tot[b] > 0.0:
                normalization_per_bin[b] = bin_pixel_tot[b] / bin_weight_tot[b]

        # Apply: for sparse bins, w*1.0 still uses init_weights, so override to uniform
        result = np.ones(count, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            result[i] = w[i] * normalization_per_bin[b]

        weights[start : start + count] = result

    return weights


@memory_usage
@array_size
@time_calls
@trace_calls
def compute_weights(
    config: PainterConfig,
    pixel_indices: np.ndarray,
    distances: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    r_90: np.ndarray,
    particle_counts: np.ndarray,
    method: Literal["normal", "vectorized"] = "vectorized",
):
    """
    Calculate the proportional weights for pixels based on particle counts.
    """
    # Use JIT-compiled function for fast parallel index lookup and power operation
    init_weights = _fast_init_weights(particle_counts, pixel_indices)

    if method == "normal":
        return weights_mechanism(
            config.search_radius,
            config.n_bins,
            distances,
            halo_starts,
            halo_counts,
            r_90,
            init_weights,
        )
    else:
        return weights_mechanism_vec(
            config, distances, halo_counts, r_90, init_weights
        )


@time_calls
def weights_mechanism_vec(
    config: PainterConfig,
    distances: np.ndarray,
    halo_counts: np.ndarray,
    r_90: np.ndarray,
    raw_weights: np.ndarray,
) -> np.ndarray:
    """Vectorized computation of normalized weights for radial bins around halos."""
    # Create halo ID for each particle
    halo_ids = np.repeat(np.arange(len(r_90)), halo_counts)

    # Compute normalized radial coordinate for all particles
    x = distances / r_90[halo_ids]  # Distance in units of r_90

    # Assign particles to radial bins
    bin_edges = np.linspace(0.0, config.search_radius, config.n_bins + 1)
    bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
    bin_ids = np.minimum(bin_ids, config.n_bins - 1)

    # Create composite key: (halo_id, bin_id) for grouping
    composite_key = halo_ids * config.n_bins + bin_ids

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
