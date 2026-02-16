import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import healpy as hp
import numpy as np

from tszpaint.logging import time_calls, timer, trace_calls


@time_calls
@trace_calls
def find_pixels_in_halos(
    nside: int,
    halo_xyz: np.ndarray,
    search_radii: np.ndarray,
    nest: bool = True,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        nside: HEALPix resolution parameter (npix = 12 x nside^2)
        halo_xyz: Halo positions as unit cartesian vectors, shape (N_halos, 3)
        search_radii: Search radius for each halo in radians, shape (N_halos,)
        nest: Use nested pixel ordering (recommended for spatial locality)

    Returns:
        pixel_indices: Flattened array of all pixel indices across all halos
        distances: Angular distance (radians) from each pixel to its halo center
        halo_starts: Starting index in flattened arrays for each halo
        halo_counts: Number of pixels found for each halo
        halo_indices: Halo index for each pixel (for vectorized operations)

    """

    if n_workers is None:
        n_workers = os.cpu_count()

    n_halos = len(halo_xyz)

    def query_single_halo(i: int):
        """Query pixels for a single halo."""
        pixels = hp.query_disc(
            nside=nside,
            vec=halo_xyz[i],
            radius=search_radii[i],
            nest=nest,
            inclusive=False,
        )

        if len(pixels) == 0:
            return i, np.array([], dtype=np.int64), np.array([])

        pix_vecs = hp.pix2vec(nside, pixels, nest=nest)
        distances = hp.rotator.angdist(halo_xyz[i], pix_vecs)

        return i, pixels, distances

    with timer(f"Parallel queries ({n_workers} workers)"):
        results = [None] * n_halos

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(query_single_halo, i): i for i in range(n_halos)}

            for future in as_completed(futures):
                i, pixels, distances = future.result()
                results[i] = (pixels, distances)

    # Flatten results
    with timer("Flattening parallel results"):
        all_pixels = []
        all_distances = []
        halo_counts = np.zeros(n_halos, dtype=np.int64)

        for i, (pixels, distances) in enumerate(results):
            if len(pixels) > 0:
                all_pixels.append(pixels)
                all_distances.append(distances)
                halo_counts[i] = len(pixels)

        pixel_indices = (
            np.concatenate(all_pixels) if all_pixels else np.array([], dtype=np.int64)
        )
        distances = np.concatenate(all_distances) if all_distances else np.array([])

        halo_starts = np.zeros(n_halos, dtype=np.int64)
        halo_starts[1:] = np.cumsum(halo_counts[:-1])

        halo_indices = np.repeat(np.arange(n_halos, dtype=np.int64), halo_counts)

    return pixel_indices, distances, halo_starts, halo_counts, halo_indices
