"""
Parallel interpolator for HPC environments with many CPU cores.

This module provides a Numba-accelerated parallel trilinear interpolator
that can utilize all available CPU cores for fast batch interpolation.

Usage:
    # Set thread count BEFORE importing (or in your job script)
    import os
    os.environ['NUMBA_NUM_THREADS'] = '64'  # or your core count

    from tszpaint.interpolator_parallel import ParallelBattagliaInterpolator

    # Load from your existing pickle file
    interp = ParallelBattagliaInterpolator.from_pickle(path)

    # Use parallel evaluation (same interface as original)
    y_values = interp.eval_for_logs(log_theta, z, log_M)
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import jit, prange


# =============================================================================
# Numba-accelerated parallel trilinear interpolation
# =============================================================================

@jit(nopython=True, cache=True)
def _find_index_and_frac(val, grid_min, grid_max, n_points):
    """Find grid index and fractional position for a single value."""
    t = (val - grid_min) / (grid_max - grid_min) * (n_points - 1)
    t = max(0.0, min(t, n_points - 1.0))
    i = int(t)
    if i >= n_points - 1:
        i = n_points - 2
        f = 1.0
    else:
        f = t - i
    return i, f


@jit(nopython=True, parallel=True, cache=True)
def trilinear_interp_parallel(
    log_theta: np.ndarray,
    z: np.ndarray,
    log_M: np.ndarray,
    grid_log_theta_min: float,
    grid_log_theta_max: float,
    n_theta: int,
    grid_z_min: float,
    grid_z_max: float,
    n_z: int,
    grid_log_M_min: float,
    grid_log_M_max: float,
    n_M: int,
    log_prof_y: np.ndarray,
) -> np.ndarray:
    """
    Parallel trilinear interpolation on a regular 3D grid.

    Set NUMBA_NUM_THREADS environment variable to control thread count.
    For HPC with 64 cores: export NUMBA_NUM_THREADS=64
    """
    n = len(log_theta)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        i0, di = _find_index_and_frac(log_theta[i], grid_log_theta_min, grid_log_theta_max, n_theta)
        j0, dj = _find_index_and_frac(z[i], grid_z_min, grid_z_max, n_z)
        k0, dk = _find_index_and_frac(log_M[i], grid_log_M_min, grid_log_M_max, n_M)

        # Get the 8 corner values for trilinear interpolation
        c000 = log_prof_y[i0, j0, k0]
        c001 = log_prof_y[i0, j0, k0 + 1]
        c010 = log_prof_y[i0, j0 + 1, k0]
        c011 = log_prof_y[i0, j0 + 1, k0 + 1]
        c100 = log_prof_y[i0 + 1, j0, k0]
        c101 = log_prof_y[i0 + 1, j0, k0 + 1]
        c110 = log_prof_y[i0 + 1, j0 + 1, k0]
        c111 = log_prof_y[i0 + 1, j0 + 1, k0 + 1]

        # Trilinear interpolation
        c00 = c000 * (1 - di) + c100 * di
        c01 = c001 * (1 - di) + c101 * di
        c10 = c010 * (1 - di) + c110 * di
        c11 = c011 * (1 - di) + c111 * di

        c0 = c00 * (1 - dj) + c10 * dj
        c1 = c01 * (1 - dj) + c11 * dj

        log_y = c0 * (1 - dk) + c1 * dk
        result[i] = np.exp(log_y)

    return result


@dataclass
class ParallelBattagliaInterpolator:
    """
    Parallel interpolator for Battaglia tSZ profiles.

    This is a drop-in replacement for BattagliaLogInterpolator that uses
    Numba parallel loops instead of JAX/scipy for much faster evaluation
    on multi-core CPUs.
    """
    grid_log_thetas: np.ndarray
    grid_redshifts: np.ndarray
    grid_log_masses: np.ndarray
    grid_log_prof_y: np.ndarray

    @classmethod
    def from_pickle(cls, path: Path):
        """Load from the same pickle format as BattagliaLogInterpolator."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        log_thetas = np.ascontiguousarray(data["log_thetas"], dtype=np.float64)
        redshifts = np.ascontiguousarray(data["redshifts"], dtype=np.float64)
        log_masses = np.ascontiguousarray(data["log_masses"], dtype=np.float64)
        prof_y = np.array(data["prof_y"], dtype=np.float64)
        log_prof_y = np.ascontiguousarray(np.log(prof_y + 1e-100))

        return cls(
            grid_log_thetas=log_thetas,
            grid_redshifts=redshifts,
            grid_log_masses=log_masses,
            grid_log_prof_y=log_prof_y,
        )

    @classmethod
    def from_matrices(
        cls,
        log_thetas: np.ndarray,
        redshifts: np.ndarray,
        log_masses: np.ndarray,
        prof_y: np.ndarray,
    ):
        """Create from raw grid arrays."""
        log_prof_y = np.log(prof_y + 1e-100)

        return cls(
            grid_log_thetas=np.ascontiguousarray(log_thetas, dtype=np.float64),
            grid_redshifts=np.ascontiguousarray(redshifts, dtype=np.float64),
            grid_log_masses=np.ascontiguousarray(log_masses, dtype=np.float64),
            grid_log_prof_y=np.ascontiguousarray(log_prof_y, dtype=np.float64),
        )

    def eval_for_logs(self, log_theta, z, log_M):
        """
        Evaluate the interpolator at given points (parallel).

        Same interface as BattagliaLogInterpolator.eval_for_logs().
        """
        log_theta = np.ascontiguousarray(log_theta, dtype=np.float64)
        z = np.ascontiguousarray(z, dtype=np.float64)
        log_M = np.ascontiguousarray(log_M, dtype=np.float64)

        return trilinear_interp_parallel(
            log_theta, z, log_M,
            self.grid_log_thetas[0], self.grid_log_thetas[-1], len(self.grid_log_thetas),
            self.grid_redshifts[0], self.grid_redshifts[-1], len(self.grid_redshifts),
            self.grid_log_masses[0], self.grid_log_masses[-1], len(self.grid_log_masses),
            self.grid_log_prof_y,
        )

    def eval(self, theta, z, m):
        """Evaluate at non-log inputs."""
        log_theta = np.log(theta)
        log_M = np.log10(m)
        return self.eval_for_logs(log_theta, z, log_M)

    def warmup(self):
        """
        Trigger Numba JIT compilation with a small test array.
        Call this once before timing to avoid including compilation time.
        """
        test_theta = np.array([self.grid_log_thetas[0]], dtype=np.float64)
        test_z = np.array([self.grid_redshifts[0]], dtype=np.float64)
        test_M = np.array([self.grid_log_masses[0]], dtype=np.float64)
        _ = self.eval_for_logs(test_theta, test_z, test_M)
