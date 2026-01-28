import time
import numpy as np
import healpy as hp
import csv
from pathlib import Path

from tszpaint.paint import (
    paint_y,
    load_interpolator,
    JAX_PATH,
)
from tszpaint.y_profile import create_battaglia_profile

NSIDE = 1024
MODEL = create_battaglia_profile()


def create_mock_halo_catalog(n_halos: int, seed: int = 123):
    """Create mock halo catalog."""
    rng = np.random.default_rng(seed)
    halo_theta = np.pi * rng.random(n_halos)
    halo_phi = 2 * np.pi * rng.random(n_halos)
    log_M = rng.uniform(13.5, 15.5, size=n_halos)
    M_halos = 10.0 ** log_M
    return halo_theta, halo_phi, M_halos


def create_mock_particle_counts(nside: int, seed: int = 28):
    """Create mock particle counts map"""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    baseline = 1_000_000_000
    contrast = rng.lognormal(mean=0.0, sigma=2.0, size=npix)
    lam = baseline * contrast
    particle_counts = rng.poisson(lam=lam).astype(np.int64)
    return particle_counts


def benchmark_paint(
    n_halos: int,
    nside: int,
    interpolator,
    particle_counts: np.ndarray,
    z: float = 0.5,
    n_runs: int = 3,
):
    """
    Benchmark weighted vs unweighted painting.
    """
    halo_theta, halo_phi, M_halos = create_mock_halo_catalog(n_halos)

    # Silent warmup run (JIT compilation)
    _ = paint_y(
        halo_theta=halo_theta,
        halo_phi=halo_phi,
        M_halos=M_halos,
        particle_counts=particle_counts,
        interpolator=interpolator,
        z=z,
        nside=nside,
        use_weights=True,
    )

    # Benchmark weighted
    weighted_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = paint_y(
            halo_theta=halo_theta,
            halo_phi=halo_phi,
            M_halos=M_halos,
            particle_counts=particle_counts,
            interpolator=interpolator,
            z=z,
            nside=nside,
            use_weights=True,
        )
        weighted_times.append(time.perf_counter() - start)

    # Benchmark unweighted
    unweighted_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = paint_y(
            halo_theta=halo_theta,
            halo_phi=halo_phi,
            M_halos=M_halos,
            particle_counts=particle_counts,
            interpolator=interpolator,
            z=z,
            nside=nside,
            use_weights=False,
        )
        unweighted_times.append(time.perf_counter() - start)

    return {
        "n_halos": n_halos,
        "nside": nside,
        "weighted_mean": np.mean(weighted_times),
        "weighted_std": np.std(weighted_times),
        "unweighted_mean": np.mean(unweighted_times),
        "unweighted_std": np.std(unweighted_times),
        "overhead_ratio": np.mean(weighted_times) / np.mean(unweighted_times),
    }


def main():
    interpolator = load_interpolator(JAX_PATH)

    particle_counts = create_mock_particle_counts(NSIDE)

    halo_counts = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]

    results = []

    for n_halos in halo_counts:
        print(f"\nBenchmarking with {n_halos:,} halos...")
        result = benchmark_paint(
            n_halos=n_halos,
            nside=NSIDE,
            interpolator=interpolator,
            particle_counts=particle_counts,
            n_runs=3,
        )
        results.append(result)

        print(f"  Weighted:   {result['weighted_mean']:.2f} ± {result['weighted_std']:.2f} s")
        print(f"  Unweighted: {result['unweighted_mean']:.2f} ± {result['unweighted_std']:.2f} s")

    output_file = Path("benchmark_weights.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'N Halos':>10} | {'Weighted (s)':>14} | {'Unweighted (s)':>14} | {'Overhead':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_halos']:>10,} | {r['weighted_mean']:>14.2f} | {r['unweighted_mean']:>14.2f} | {r['overhead_ratio']:>10.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
