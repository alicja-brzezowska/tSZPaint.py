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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all HEALPix pixels within search radius of each halo.

    Args:
        nside: HEALPix resolution parameter (npix = 12 × nside²)
        halo_xyz: Halo positions as unit cartesian vectors, shape (N_halos, 3)
        search_radii: Search radius for each halo in radians, shape (N_halos,)
        nest: Use nested pixel ordering (recommended for spatial locality)

    Returns:
        pixel_indices: Flattened array of all pixel indices across all halos
        distances: Angular distance (radians) from each pixel to its halo center
        halo_starts: Starting index in flattened arrays for each halo
        halo_counts: Number of pixels found for each halo
        halo_indices: Halo index for each pixel (for vectorized operations)

    Example:
        >>> pixels, dists, starts, counts, halo_ids = find_pixels_in_halos(
        ...     nside=512,
        ...     halo_xyz=halo_positions,
        ...     search_radii=3.0 * halo_theta_200,
        ... )
        >>> # Get pixels for halo i:
        >>> halo_i_pixels = pixels[starts[i]:starts[i]+counts[i]]
        >>> halo_i_dists = dists[starts[i]:starts[i]+counts[i]]
    """
    n_halos = len(halo_xyz)

    # Collect results per halo
    all_pixels = []
    all_distances = []
    halo_counts = np.zeros(n_halos, dtype=np.int64)

    with timer("Querying HEALPix discs"):
        for i in range(n_halos):
            # Find all pixels within angular radius
            pixels = hp.query_disc(
                nside=nside,
                vec=halo_xyz[i],
                radius=search_radii[i],
                nest=nest,
                inclusive=False,
            )

            if len(pixels) == 0:
                continue

            # Get pixel center vectors and compute angular distances
            pix_vecs = hp.pix2vec(nside, pixels, nest=nest)
            distances = hp.rotator.angdist(halo_xyz[i], pix_vecs)

            all_pixels.append(pixels)
            all_distances.append(distances)
            halo_counts[i] = len(pixels)

    # Flatten results into contiguous arrays
    with timer("Flattening results"):
        pixel_indices = (
            np.concatenate(all_pixels) if all_pixels else np.array([], dtype=np.int64)
        )
        distances = np.concatenate(all_distances) if all_distances else np.array([])

        # Compute offsets for each halo in the flattened arrays
        halo_starts = np.zeros(n_halos, dtype=np.int64)
        halo_starts[1:] = np.cumsum(halo_counts[:-1])

        # Create mapping from pixel → halo
        halo_indices = np.repeat(np.arange(n_halos, dtype=np.int64), halo_counts)

    return pixel_indices, distances, halo_starts, halo_counts, halo_indices


# Parallel version for large catalogs
@time_calls
@trace_calls
def find_pixels_in_halos_parallel(
    nside: int,
    halo_xyz: np.ndarray,
    search_radii: np.ndarray,
    nest: bool = True,
    n_workers: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel version of find_pixels_in_halos.

    Use when n_halos > ~10,000 for better performance.

    Args:
        nside: HEALPix resolution parameter
        halo_xyz: Halo positions as unit cartesian vectors, shape (N_halos, 3)
        search_radii: Search radius for each halo in radians, shape (N_halos,)
        nest: Use nested pixel ordering
        n_workers: Number of parallel workers (default: cpu_count)

    Returns:
        Same as find_pixels_in_halos()
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
