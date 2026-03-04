import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import healpy as hp
import numpy as np

from tszpaint.logging import time_calls, timer, trace_calls, memory_usage, array_size


@memory_usage
@array_size
@time_calls
@trace_calls
def find_pixels_in_halos(
    nside: int,
    halo_xyz: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    nest: bool = True,
    n_workers: int | None = None,
    geometry: Literal["triaxial", "spherical"] = "triaxial",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find halo pixels and return effective projected radius per halo-pixel pair.
    """

    if n_workers is None:
        n_workers = os.cpu_count()

    n_halos = len(halo_xyz)

    def tangent_basis(n0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For the vector n0, define a tangent cartesian basis (t1,t2)"""
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(n0, ref)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # define two orthonormal vectors t1 and t2 to n0 
        t1 = np.cross(ref, n0)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n0, t1)
        t2 /= np.linalg.norm(t2)
        return t1, t2

    def query_single_halo(i: int):
        evals = eigenvalues[i]
        evecs = eigenvectors[i]
        n0 = halo_xyz[i]

        if geometry == "spherical":
            r_eff_2d = float(np.maximum(evals[0], np.finfo(np.float64).tiny))
            pixels = hp.query_disc(
                nside=nside,
                vec=n0,
                radius=r_eff_2d,
                nest=nest,
                inclusive=True,
            )

            if len(pixels) == 0:
                return i, np.array([], dtype=np.int64), np.array([]), r_eff_2d

            x, y, z = hp.pix2vec(nside, pixels, nest=nest)
            pixel_xyz = np.stack([x, y, z], axis=1)
            cosang = np.clip(pixel_xyz @ n0, -1.0, 1.0)
            distances = np.arccos(cosang)

            return i, pixels, distances, r_eff_2d

        # normalization 
        e_a = evecs[2] / np.linalg.norm(evecs[2])
        e_b = evecs[1] / np.linalg.norm(evecs[1])
        e_c = evecs[0] / np.linalg.norm(evecs[0])

        # matrix Q of the ellipsoid in 3D space: x^T Q x = 1 
        inv_a2 = 1.0 / (evals[0] * evals[0])
        inv_b2 = 1.0 / (evals[1] * evals[1])
        inv_c2 = 1.0 / (evals[2] * evals[2])
        Q = (
            np.outer(e_a, e_a) * inv_a2
            + np.outer(e_b, e_b) * inv_b2
            + np.outer(e_c, e_c) * inv_c2
        )

        # project the ellipsoid onto the plane perpendicular to n0
        t1, t2 = tangent_basis(n0)
        S11 = t1 @ Q @ t1
        S12 = t1 @ Q @ t2
        S22 = t2 @ Q @ t2

        # 2D ellipse matrix; solve for the eigenvalues 
        S = np.array([[S11, S12], [S12, S22]], dtype=np.float64)
        s_eigs = np.linalg.eigvalsh(S)
        a_proj = 1.0 / np.sqrt(max(s_eigs[0], np.finfo(np.float64).tiny))
        c_proj = 1.0 / np.sqrt(max(s_eigs[1], np.finfo(np.float64).tiny))
        # effective radius
        r_eff_2d = np.sqrt(a_proj * c_proj) # A = pi * a_proj * c_proj

        # initial query to the major projected axis
        query_radius = a_proj
        pixels = hp.query_disc(
            nside=nside,
            vec=n0,
            radius=query_radius,
            nest=nest,
            inclusive=True,
        )

        if len(pixels) == 0:
            return i, np.array([], dtype=np.int64), np.array([]), r_eff_2d

        x, y, z = hp.pix2vec(nside, pixels, nest=nest)
        pixel_xyz = np.stack([x, y, z], axis=1)
        u = (pixel_xyz @ t1) / np.maximum(pixel_xyz @ n0, 1e-12)
        v = (pixel_xyz @ t2) / np.maximum(pixel_xyz @ n0, 1e-12)

        val = S11 * u * u + 2.0 * S12 * u * v + S22 * v * v
        # only keep pixels within the ellipse 
        mask = val <= 1.0

        pixels = pixels[mask]
        if len(pixels) == 0:
            return i, np.array([], dtype=np.int64), np.array([]), r_eff_2d

        val = np.maximum(val[mask], 0.0)
        distances = r_eff_2d * np.sqrt(val)

        return i, pixels, distances, r_eff_2d

    with timer(f"Parallel queries ({n_workers} workers)"):
        results = [None] * n_halos

        halo_effective_radius = np.zeros(n_halos, dtype=np.float64)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(query_single_halo, i): i for i in range(n_halos)}

            for future in as_completed(futures):
                i, pixels, distances, r_eff_2d = future.result()
                results[i] = (pixels, distances)
                halo_effective_radius[i] = r_eff_2d

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

    return (
        pixel_indices,
        distances,
        halo_starts,
        halo_counts,
        halo_indices,
        halo_effective_radius,
    )

