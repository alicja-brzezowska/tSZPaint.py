import numpy as np
from numba import prange

from tszpaint.logging import time_calls, trace_calls
from tszpaint.paint.config import PainterConfig


def weights_mechanism(
    config: PainterConfig,
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
    bin_edges = np.linspace(0.0, config.search_radius, config.n_bins + 1)

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
        bin_ids = np.minimum(bin_ids, config.n_bins - 1)

        # Count total pixels and non-empty pixels per bin
        bin_pixel_tot = np.zeros(config.n_bins, dtype=np.float64)
        bin_weight_tot = np.zeros(config.n_bins, dtype=np.float64)
        bin_nonempty = np.zeros(config.n_bins, dtype=np.float64)
        for i in range(count):
            b = bin_ids[i]
            bin_pixel_tot[b] += 1.0
            bin_weight_tot[b] += w[i]
            if w[i] > 1e-12:
                bin_nonempty[b] += 1.0

        normalization_per_bin = np.ones(config.n_bins, dtype=np.float64)
        for b in range(config.n_bins):
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
    counts = particle_counts[pixel_indices]
    init_weights = np.power(counts + 1e-10, 5.0 / 3.0)  # propto N_particles^(5/3)

    weights = weights_mechanism(
        config,
        distances,
        halo_starts,
        halo_counts,
        theta_200,
        init_weights,
    )

    return weights
