from unittest import result
import numpy as np
from numba import prange, jit

from tszpaint.logging import time_calls, trace_calls, memory_usage, array_size
from tszpaint.paint.config import PainterConfig


@jit(nopython=True, parallel=True)
def _fast_init_weights(particle_counts: np.ndarray, pixel_indices: np.ndarray) -> np.ndarray:
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
    theta_200: np.ndarray,
    init_weights: np.ndarray,
):
    """compute normalized weights for each particle contribution within halos."""

    N_halos = len(theta_200)
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
        x = d / theta_200[h]

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
    theta_200: np.ndarray,
    particle_counts: np.ndarray,
):
    """
    Calculate the proportional weights for pixels based on particle counts.
    """
    # Use JIT-compiled function for fast parallel index lookup and power operation
    init_weights = _fast_init_weights(particle_counts, pixel_indices)

    # Extract config values for JIT function
    weights = weights_mechanism(
        config.search_radius,
        config.n_bins,
        distances,
        halo_starts,
        halo_counts,
        theta_200,
        init_weights,
    )

    return weights
