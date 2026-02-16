"""
Benchmark script comparing KD-tree vs HEALPix native methods for halo painting.
"""

import time
from dataclasses import dataclass

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from tszpaint.converters import convert_rad_to_cart

# Import your implementations
from tszpaint.cosmology.model import get_angular_size_from_comoving
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.mock_data_generator import MockDataGenerator
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.paint.tree import build_tree, query_tree
from tszpaint.y_profile.y_profile import (
    create_battaglia_profile,
)


@dataclass
class BenchmarkResult:
    """Store benchmark results."""

    method_name: str
    build_time: float
    query_time: float
    total_time: float
    n_pixels_found: int
    n_halos: int
    nside: int


def run_kdtree_benchmark(
    config: PainterConfig,
    halo_xyz: np.ndarray,
    theta_200: np.ndarray,
    search_radii: np.ndarray,
):
    """Benchmark KD-tree implementation."""
    print("\n" + "=" * 70)
    print("KD-TREE METHOD")
    print("=" * 70)

    # Build tree
    start = time.time()
    tree, pix_xyz, pix_indices = build_tree(config)
    build_time = time.time() - start
    print(f"Build time: {build_time:.4f}s")

    # Query tree
    start = time.time()
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = query_tree(
        config, halo_xyz, theta_200, tree, pix_xyz
    )
    query_time = time.time() - start
    print(f"Query time: {query_time:.4f}s")

    total_time = build_time + query_time
    print(f"Total time: {total_time:.4f}s")
    print(f"Pixels found: {len(pix_in_halos):,}")

    return BenchmarkResult(
        method_name="KD-Tree",
        build_time=build_time,
        query_time=query_time,
        total_time=total_time,
        n_pixels_found=len(pix_in_halos),
        n_halos=len(halo_xyz),
        nside=config.nside,
    ), (pix_in_halos, distances, halo_counts)


def run_healpy_benchmark(
    nside: int, halo_xyz: np.ndarray, search_radii: np.ndarray, n_workers: int = None
):
    """Benchmark HEALPix parallel implementation."""
    print("\n" + "=" * 70)
    print(f"HEALPY PARALLEL METHOD ({n_workers or 'auto'} workers)")
    print("=" * 70)

    # No build time - directly query
    start = time.time()
    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = (
        find_pixels_in_halos(
            nside, halo_xyz, search_radii, nest=True, n_workers=n_workers
        )
    )
    total_time = time.time() - start

    print(f"Total time: {total_time:.4f}s")
    print(f"Pixels found: {len(pix_in_halos):,}")

    return BenchmarkResult(
        method_name=f"HEALPix-Parallel-{n_workers or 'auto'}",
        build_time=0.0,
        query_time=total_time,
        total_time=total_time,
        n_pixels_found=len(pix_in_halos),
        n_halos=len(halo_xyz),
        nside=nside,
    ), (pix_in_halos, distances, halo_counts)


def compare_results(result_kdtree, result_healpy):
    """Compare outputs from both methods."""
    pix_kd, dist_kd, counts_kd = result_kdtree
    pix_hp, dist_hp, counts_hp = result_healpy

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    # Compare counts
    print(f"Pixel count match: {len(pix_kd) == len(pix_hp)}")
    print(f"  KD-tree: {len(pix_kd):,} pixels")
    print(f"  HEALPix: {len(pix_hp):,} pixels")

    # Compare halo counts
    counts_match = np.allclose(counts_kd, counts_hp)
    print(f"Halo counts match: {counts_match}")
    if not counts_match:
        diff = np.abs(counts_kd - counts_hp)
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean():.2f}")

    # Compare distance statistics (they may differ due to different methods)
    if len(dist_kd) > 0 and len(dist_hp) > 0:
        print("\nDistance statistics:")
        print(f"  KD-tree:  mean={np.mean(dist_kd):.6f}, std={np.std(dist_kd):.6f}")
        print(f"  HEALPix:  mean={np.mean(dist_hp):.6f}, std={np.std(dist_hp):.6f}")

        # Sample comparison (sort both to align)
        n_sample = min(1000, len(dist_kd), len(dist_hp))
        dist_kd_sorted = np.sort(dist_kd)[:n_sample]
        dist_hp_sorted = np.sort(dist_hp)[:n_sample]
        max_diff = np.max(np.abs(dist_kd_sorted - dist_hp_sorted))
        print(f"  Max diff in sorted sample ({n_sample}): {max_diff:.2e} radians")


def run_benchmark_suite():
    """Run comprehensive benchmark across different configurations."""

    configs = [
        # (nside, n_halos, description)
        (512, 1000, "Small: 512 nside, 100 halos"),
        (1024, 5000, "Medium: 1024 nside, 5k halos"),
        (2048, 10000, "Large: 2048 nside, 10k halos"),
        (2048, 50000, "XLarge: 2048 nside, 50k halos"),
    ]

    all_results = []

    for nside, n_halos, description in configs:
        print("\n" + "#" * 70)
        print(f"# {description}")
        print("#" * 70)

        # Generate mock data
        print("\nGenerating mock data...")
        generator = MockDataGenerator(n_halos=n_halos, nside=nside)
        sim_data = generator.generate_simulation_data()

        # Convert to cartesian
        halo_xyz = convert_rad_to_cart(sim_data.theta, sim_data.phi)

        theta_200 = get_angular_size_from_comoving(
            create_battaglia_profile(), sim_data.radii_halos, 0.5
        )
        # theta_200 = sim_data.radii_halos

        # Create config
        search_radius_multiplier = 2.0
        search_radii = search_radius_multiplier * theta_200

        config = PainterConfig(
            nside=nside,
            search_radius=search_radius_multiplier,
        )

        print(f"  nside={nside} → {hp.nside2npix(nside):,} pixels")
        print(f"  n_halos={n_halos}")
        print(f"  search_radius={search_radius_multiplier}x theta_200")

        # Run KD-tree benchmark
        try:
            result_kd, output_kd = run_kdtree_benchmark(
                config, halo_xyz, theta_200, search_radii
            )
            all_results.append(result_kd)
        except Exception as e:
            print(f"KD-tree failed: {e}")
            result_kd, output_kd = None, None

        # Run HEALPix benchmarks with different worker counts
        for n_workers in [1, 4, 8]:
            try:
                result_hp, output_hp = run_healpy_benchmark(
                    nside, halo_xyz, search_radii, n_workers=n_workers
                )
                all_results.append(result_hp)

                # Compare results (only once, with first HEALPix run)
                if n_workers == 1 and result_kd is not None:
                    compare_results(output_kd, output_hp)

            except Exception as e:
                print(f"HEALPix (n_workers={n_workers}) failed: {e}")

    return all_results


def plot_benchmark_comparison(results, output_path="benchmark_comparison.png"):
    """Create a single comprehensive comparison plot."""
    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(
        [
            {
                "method": r.method_name,
                "nside": r.nside,
                "n_halos": r.n_halos,
                "build_time": r.build_time,
                "query_time": r.query_time,
                "total_time": r.total_time,
            }
            for r in results
        ]
    )

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "KD-Tree vs HEALPix Benchmark Comparison", fontsize=16, fontweight="bold"
    )

    # ========================================================================
    # Plot 1: Total time by configuration (grouped bar chart)
    # ========================================================================
    ax = axes[0, 0]

    # Get unique configurations
    unique_configs = df[["nside", "n_halos"]].drop_duplicates()
    config_labels = [
        f"nside={row.nside}\n{row.n_halos}k halos"
        for _, row in unique_configs.iterrows()
    ]

    methods = df["method"].unique()
    x = np.arange(len(unique_configs))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        times = []
        for _, config_row in unique_configs.iterrows():
            time_val = df[
                (df["nside"] == config_row.nside)
                & (df["n_halos"] == config_row.n_halos)
                & (df["method"] == method)
            ]["total_time"]
            times.append(time_val.iloc[0] if not time_val.empty else 0)

        ax.bar(x + i * width, times, width, label=method, alpha=0.8)

    ax.set_xlabel("Configuration", fontweight="bold")
    ax.set_ylabel("Total Time (seconds)", fontweight="bold")
    ax.set_title("Total Time by Configuration")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # ========================================================================
    # Plot 2: Speedup relative to KD-Tree
    # ========================================================================
    ax = axes[0, 1]

    for _, config_row in unique_configs.iterrows():
        config_df = df[
            (df["nside"] == config_row.nside) & (df["n_halos"] == config_row.n_halos)
        ]

        kdtree_time = config_df[config_df["method"] == "KD-Tree"]["total_time"]
        if kdtree_time.empty:
            continue

        kdtree_time = kdtree_time.iloc[0]

        # Calculate speedups for HEALPix methods
        healpy_methods = config_df[config_df["method"] != "KD-Tree"]
        if healpy_methods.empty:
            continue

        speedups = kdtree_time / healpy_methods["total_time"]
        labels = [
            m.split("-")[-1] for m in healpy_methods["method"]
        ]  # Extract worker count

        config_label = f"nside={config_row.nside}, {config_row.n_halos}k halos"
        ax.plot(
            labels,
            speedups.values,
            marker="o",
            linewidth=2,
            markersize=8,
            label=config_label,
        )

    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="KD-Tree baseline",
        alpha=0.7,
    )
    ax.set_xlabel("Number of Workers", fontweight="bold")
    ax.set_ylabel("Speedup (×)", fontweight="bold")
    ax.set_title("Speedup vs KD-Tree (Higher = Better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 3: Build vs Query time breakdown (KD-Tree only)
    # ========================================================================
    ax = axes[1, 0]

    kdtree_df = df[df["method"] == "KD-Tree"].copy()
    if not kdtree_df.empty:
        kdtree_df["config"] = kdtree_df.apply(
            lambda row: f"{row['nside']}/{row['n_halos']}k", axis=1
        )

        x = np.arange(len(kdtree_df))
        width = 0.6

        ax.bar(
            x,
            kdtree_df["build_time"],
            width,
            label="Build Tree",
            alpha=0.8,
            color="steelblue",
        )
        ax.bar(
            x,
            kdtree_df["query_time"],
            width,
            bottom=kdtree_df["build_time"],
            label="Query",
            alpha=0.8,
            color="coral",
        )

        ax.set_xlabel("Configuration (nside/n_halos)", fontweight="bold")
        ax.set_ylabel("Time (seconds)", fontweight="bold")
        ax.set_title("KD-Tree Time Breakdown")
        ax.set_xticks(x)
        ax.set_xticklabels(kdtree_df["config"], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # ========================================================================
    # Plot 4: Scaling with problem size
    # ========================================================================
    ax = axes[1, 1]

    for method in methods:
        method_df = df[df["method"] == method].sort_values("n_halos")
        if not method_df.empty:
            ax.plot(
                method_df["n_halos"],
                method_df["total_time"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=method,
            )

    ax.set_xlabel("Number of Halos", fontweight="bold")
    ax.set_ylabel("Total Time (seconds)", fontweight="bold")
    ax.set_title("Scaling with Problem Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend(fontsize=8)

    # ========================================================================
    # Adjust layout and save
    # ========================================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    return fig


def print_summary_table(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Config':<25} {'Method':<20} {'Build':<10} {'Query':<10} {'Total':<10}")
    print("-" * 70)

    # Group by configuration
    configs = {}
    for r in results:
        key = (r.nside, r.n_halos)
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    # Print each configuration
    for (nside, n_halos), config_results in sorted(configs.items()):
        config_str = f"nside={nside}, n_halos={n_halos}"
        for i, r in enumerate(config_results):
            if i == 0:
                print(f"{config_str:<25} ", end="")
            else:
                print(f"{'':<25} ", end="")
            print(
                f"{r.method_name:<20} {r.build_time:>8.3f}s  {r.query_time:>8.3f}s  {r.total_time:>8.3f}s"
            )

        # Print speedup
        kdtree_result = next(
            (r for r in config_results if r.method_name == "KD-Tree"), None
        )
        if kdtree_result:
            print(f"{'':<25} {'Speedup vs KD-Tree:':<20}", end="")
            for r in config_results:
                if r.method_name != "KD-Tree":
                    speedup = kdtree_result.total_time / r.total_time
                    print(f" {r.method_name}: {speedup:.2f}×", end="")
            print()
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("HALO PAINTING BENCHMARK: KD-TREE vs HEALPY")
    print("=" * 70)

    # Run benchmarks
    results = run_benchmark_suite()

    # Print summary
    print_summary_table(results)

    # Create plots
    plot_benchmark_comparison(results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
