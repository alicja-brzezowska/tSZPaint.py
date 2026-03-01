"""Clean benchmark: KD-tree vs HEALPix for a single configuration."""

import time
from dataclasses import dataclass

import healpy as hp
import numpy as np

from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_TOTAL_PATH
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import get_angular_size_from_comoving
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.paint.tree import build_tree, query_tree
from tszpaint.y_profile.y_profile import create_battaglia_profile

NSIDE = 8192
SEARCH_RADIUS_MULTIPLIER = 0.3
WORKER_COUNTS = (1,)
HALO_PATH = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
HEALCOUNTS_PATH = (
    HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0671-0676.asdf"
)


@dataclass
class BenchmarkResult:
    method_name: str
    build_time: float
    query_time: float
    total_time: float
    n_pixels_found: int


def run_kdtree_benchmark(config: PainterConfig, halo_xyz: np.ndarray, r_90: np.ndarray):
    start = time.time()
    tree, pix_xyz, _ = build_tree(config)
    build_time = time.time() - start

    start = time.time()
    pix_in_halos, distances, _, halo_counts, _ = query_tree(
        config, halo_xyz, r_90, tree, pix_xyz
    )
    query_time = time.time() - start

    return (
        BenchmarkResult(
            method_name="KD-Tree",
            build_time=build_time,
            query_time=query_time,
            total_time=build_time + query_time,
            n_pixels_found=len(pix_in_halos),
        ),
        (pix_in_halos, distances, halo_counts),
    )


def run_healpy_benchmark(
    nside: int,
    halo_xyz: np.ndarray,
    search_radii: np.ndarray,
    n_workers: int,
):
    start = time.time()
    pix_in_halos, distances, _, halo_counts, _ = find_pixels_in_halos(
        nside, halo_xyz, search_radii, nest=True, n_workers=n_workers
    )
    total_time = time.time() - start

    return (
        BenchmarkResult(
            method_name=f"HEALPix-Parallel-{n_workers}",
            build_time=0.0,
            query_time=total_time,
            total_time=total_time,
            n_pixels_found=len(pix_in_halos),
        ),
        (pix_in_halos, distances, halo_counts),
    )


def compare_results(result_kdtree, result_healpy):
    pix_kd, dist_kd, counts_kd = result_kdtree
    pix_hp, dist_hp, counts_hp = result_healpy

    print("\nRESULTS COMPARISON")
    print(f"Pixel count match: {len(pix_kd) == len(pix_hp)}")
    print(f"  KD-tree: {len(pix_kd):,} pixels")
    print(f"  HEALPix: {len(pix_hp):,} pixels")

    counts_match = np.allclose(counts_kd, counts_hp)
    print(f"Halo counts match: {counts_match}")
    if not counts_match:
        diff = np.abs(counts_kd - counts_hp)
        print(f"  Max difference: {diff.max()}")
        print(f"  Mean difference: {diff.mean():.2f}")

    if len(dist_kd) > 0 and len(dist_hp) > 0:
        print("\nDistance statistics:")
        print(f"  KD-tree:  mean={np.mean(dist_kd):.6f}, std={np.std(dist_kd):.6f}")
        print(f"  HEALPix:  mean={np.mean(dist_hp):.6f}, std={np.std(dist_hp):.6f}")


def print_summary(results: list[BenchmarkResult], kdtree_total: float | None):
    print("\nSUMMARY")
    print(f"{'Method':<20} {'Build':<10} {'Query':<10} {'Total':<10} {'Pixels':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r.method_name:<20} {r.build_time:>8.3f}s  {r.query_time:>8.3f}s  "
            f"{r.total_time:>8.3f}s  {r.n_pixels_found:>8}"
        )
    if kdtree_total:
        print("\nSpeedup vs KD-Tree:")
        for r in results:
            if r.method_name != "KD-Tree":
                print(f"  {r.method_name}: {kdtree_total / r.total_time:.2f}×")


def main():
    print("HALO PAINTING BENCHMARK: KD-Tree vs HEALPix")
    print(f"nside={NSIDE}")
    print(f"search_radius={SEARCH_RADIUS_MULTIPLIER}x r_90")
    print(f"pixels={hp.nside2npix(NSIDE):,}")

    data = load_abacus_for_painting(
        halo_dir=HALO_PATH,
        healcounts_file_1=HEALCOUNTS_PATH,
        nside=NSIDE,
    )
    halo_xyz = convert_rad_to_cart(data.theta, data.phi)
    print(f"n_halos={len(data.m_halos):,}")

    r_90 = get_angular_size_from_comoving(
        create_battaglia_profile(), data.radii_halos, data.redshift
    )
    search_radii = SEARCH_RADIUS_MULTIPLIER * r_90
    config = PainterConfig(nside=NSIDE, search_radius=SEARCH_RADIUS_MULTIPLIER)

    results: list[BenchmarkResult] = []

    print("\nKD-Tree")
    kd_result, kd_output = run_kdtree_benchmark(config, halo_xyz, r_90)
    results.append(kd_result)

    first_healpy_output = None
    for n_workers in WORKER_COUNTS:
        print(f"\nHEALPix (workers={n_workers})")
        hp_result, hp_output = run_healpy_benchmark(
            NSIDE, halo_xyz, search_radii, n_workers
        )
        results.append(hp_result)
        if first_healpy_output is None:
            first_healpy_output = hp_output

    if first_healpy_output is not None:
        compare_results(kd_output, first_healpy_output)

    print_summary(results, kd_result.total_time)


if __name__ == "__main__":
    main()
