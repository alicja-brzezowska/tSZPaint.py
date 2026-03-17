"""
Compare inclusive (chi-padded) vs exclusive (exact chi edges) halo loading across all shells.

Inclusive: chi_min - delta_chi <= chi <= chi_max + delta_chi
Exclusive: chi_min             <= chi <= chi_max

The total inclusive count minus total exclusive count equals the number of halos that are
double-counted between adjacent shells when using inclusive loading.

For a painting comparison, pass --paint to run both versions at low nside and compare
the summed y-maps across all processed shells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import asdf
import healpy as hp
import numpy as np

from tszpaint.config import ABACUS_DATA_PATH, HALO_CATALOGS_PATH
from tszpaint.scripts.match_redshifts import discover_halo_catalogs


THETA_SEARCH = 1.6732e-3  # radians; sets delta_chi = chi_mid * THETA_SEARCH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_chi_range(particle_counts_file: Path) -> tuple[float, float]:
    with asdf.open(particle_counts_file) as af:
        chis = np.asarray(
            [h["CoordinateDistanceHMpc"] for h in af["headers"]], dtype=np.float64
        )
    return float(chis.min()), float(chis.max())


def _count_halos_in_range(halo_file: Path, chi_min: float, chi_max: float) -> int:
    with asdf.open(halo_file) as af:
        chi = np.asarray(af["halo_lightcone"]["Interpolated_ComovingDist"])
    return int(np.count_nonzero((chi >= chi_min) & (chi <= chi_max)))


def _delta_chi(chi_min: float, chi_max: float) -> float:
    chi_mid = 0.5 * (chi_min + chi_max)
    return chi_mid * THETA_SEARCH


# ---------------------------------------------------------------------------
# Count-based comparison (no painting needed)
# ---------------------------------------------------------------------------

def count_comparison(shells: list[Path], catalogs) -> None:
    """For each shell, compare exclusive vs inclusive halo counts and report overlap zones."""

    print(f"\n{'shell':<30} {'chi_min':>8} {'chi_max':>8} {'delta_chi':>9} "
          f"{'excl':>8} {'incl':>8} {'excess':>8} {'excess%':>8}")
    print("-" * 95)

    shell_data = []  # (chi_min, chi_max, n_excl, n_incl)

    for pc_file in sorted(shells):
        if not pc_file.exists():
            continue

        chi_min, chi_max = _read_chi_range(pc_file)
        dc = _delta_chi(chi_min, chi_max)
        chi_min_incl = chi_min - dc
        chi_max_incl = chi_max + dc

        # catalogs overlapping the inclusive range
        matched = [
            c for c in catalogs
            if max(chi_min_incl, c.chi_min) <= min(chi_max_incl, c.chi_max)
        ]

        n_excl = sum(_count_halos_in_range(c.file_path, chi_min, chi_max) for c in matched)
        n_incl = sum(_count_halos_in_range(c.file_path, chi_min_incl, chi_max_incl) for c in matched)
        excess = n_incl - n_excl
        excess_pct = 100.0 * excess / n_excl if n_excl > 0 else float("nan")

        print(
            f"{pc_file.stem:<30} {chi_min:>8.2f} {chi_max:>8.2f} {dc:>9.4f} "
            f"{n_excl:>8d} {n_incl:>8d} {excess:>8d} {excess_pct:>7.2f}%"
        )
        shell_data.append((chi_min, chi_max, n_excl, n_incl))

    if not shell_data:
        print("No shells found.")
        return

    total_excl = sum(d[2] for d in shell_data)
    total_incl = sum(d[3] for d in shell_data)
    total_excess = total_incl - total_excl
    total_excess_pct = 100.0 * total_excess / total_excl if total_excl > 0 else float("nan")
    print("-" * 95)
    print(
        f"{'TOTAL':<30} {'':>8} {'':>8} {'':>9} "
        f"{total_excl:>8d} {total_incl:>8d} {total_excess:>8d} {total_excess_pct:>7.2f}%"
    )

    # --- adjacent-shell overlap zones ---
    shell_data.sort(key=lambda d: d[0])
    print(f"\n{'Adjacent shell boundary overlap zones':}")
    print(f"{'boundary_chi':>12} {'overlap_width':>14} {'halos_in_zone':>14} {'(in both shells)':}")
    print("-" * 60)

    for i in range(len(shell_data) - 1):
        chi_min_a, chi_max_a, _, _ = shell_data[i]
        chi_min_b, chi_max_b, _, _ = shell_data[i + 1]
        boundary = 0.5 * (chi_max_a + chi_min_b)  # approximate boundary

        dc_a = _delta_chi(chi_min_a, chi_max_a)
        dc_b = _delta_chi(chi_min_b, chi_max_b)
        # overlap zone: [chi_max_a - dc_b, chi_max_a + dc_a]
        zone_min = chi_max_a - dc_b
        zone_max = chi_max_a + dc_a
        overlap_width = zone_max - zone_min

        matched = [
            c for c in catalogs
            if max(zone_min, c.chi_min) <= min(zone_max, c.chi_max)
        ]
        n_overlap = sum(_count_halos_in_range(c.file_path, zone_min, zone_max) for c in matched)

        print(f"{boundary:>12.2f} {overlap_width:>14.4f} {n_overlap:>14d}")


# ---------------------------------------------------------------------------
# Painting comparison (optional, requires --paint)
# ---------------------------------------------------------------------------

def paint_comparison(shells: list[Path], catalogs, nside: int = 128) -> None:
    """
    Paint adjacent shells at low nside with both inclusive and exclusive halo loading,
    then compare the summed y-maps.

    Only uses a single (fiducial) Battaglia parameter set for speed.
    """
    from tszpaint.paint.abacus_loader import (
        degrade_healcounts,
        obtain_healcount_edges,
        obtain_healcount_mean_redshift,
        load_abacus_halos,
        SimulationData,
    )
    from tszpaint.paint.config import PainterConfig
    from tszpaint.paint.paint import build_paint_context, paint_y_maps

    config = PainterConfig(nside=nside, search_radius=4.0, weight_bin_width=2e-5)

    y_sum_excl = np.zeros(hp.nside2npix(nside))
    y_sum_incl = np.zeros(hp.nside2npix(nside))

    for pc_file in sorted(shells):
        if not pc_file.exists():
            continue

        chi_min, chi_max = obtain_healcount_edges(pc_file)
        redshift = obtain_healcount_mean_redshift(pc_file)
        dc = _delta_chi(chi_min, chi_max)
        particle_counts = degrade_healcounts(
            np.array(asdf.open(pc_file)["data"]["heal-counts"]), nside_out=nside
        )

        def _load_halos_for_range(chi_lo: float, chi_hi: float):
            matched = [
                c for c in catalogs
                if max(chi_lo, c.chi_min) <= min(chi_hi, c.chi_max)
            ]
            all_xyz, all_npart, all_r, all_ev, all_evec = [], [], [], [], []
            pm = None
            for cat in matched:
                xyz, npart, p, _, r, chi, ev, evec = load_abacus_halos(cat.file_path)
                if pm is None:
                    pm = p
                mask = (chi >= chi_lo) & (chi <= chi_hi)
                all_xyz.append(xyz[mask])
                all_npart.append(npart[mask])
                all_r.append(r[mask])
                all_ev.append(ev[mask])
                all_evec.append(evec[mask])
            if not all_xyz:
                return None
            from tszpaint.converters import convert_comoving_to_sky
            halo_xyz = np.concatenate(all_xyz)
            theta, phi = convert_comoving_to_sky(halo_xyz)
            m = np.concatenate(all_npart).astype(np.float64) * pm
            return SimulationData(
                theta=theta,
                phi=phi,
                m_halos=m,
                particle_counts=particle_counts,
                redshift=redshift,
                radii_halos=np.concatenate(all_r),
                eigenvalues=np.concatenate(all_ev),
                eigenvectors=np.concatenate(all_evec),
            )

        sim_excl = _load_halos_for_range(chi_min, chi_max)
        sim_incl = _load_halos_for_range(chi_min - dc, chi_max + dc)

        for sim, y_sum, label in [
            (sim_excl, y_sum_excl, "excl"),
            (sim_incl, y_sum_incl, "incl"),
        ]:
            if sim is None:
                print(f"  {shell_dir.name}: no halos for {label}, skipping")
                continue
            ctx = build_paint_context(sim, config, use_weights=True)
            # paint with only one interpolator index (fiducial=62) for speed
            y_maps = paint_y_maps(ctx, config, interpolator_indices=[62])
            y_sum += y_maps[0]

    # --- comparison ---
    nonzero = (y_sum_excl > 0) | (y_sum_incl > 0)
    print(f"\nPainting comparison at nside={nside}:")
    print(f"  Pixels with signal: {nonzero.sum()}")
    print(f"  Mean y (excl): {y_sum_excl[nonzero].mean():.4e}")
    print(f"  Mean y (incl): {y_sum_incl[nonzero].mean():.4e}")
    ratio = y_sum_incl[nonzero] / np.where(y_sum_excl[nonzero] > 0, y_sum_excl[nonzero], np.nan)
    print(f"  Mean ratio incl/excl: {np.nanmean(ratio):.4f}  (1.0 = no double counting)")
    print(f"  Max  ratio incl/excl: {np.nanmax(ratio):.4f}")
    excess_frac = (y_sum_incl[nonzero].sum() - y_sum_excl[nonzero].sum()) / y_sum_excl[nonzero].sum()
    print(f"  Fractional signal excess (incl - excl) / excl: {excess_frac:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--healcounts-dir",
        type=Path,
        default=ABACUS_DATA_PATH / "lightcone_healpix" / "halo" / "heal-counts",
        help="Directory containing healcounts .asdf files (default: %(default)s)",
    )
    parser.add_argument(
        "--shells",
        nargs="+",
        type=Path,
        default=None,
        help="Specific healcounts .asdf file paths to analyse. "
             "Defaults to all .asdf files in --healcounts-dir.",
    )
    parser.add_argument(
        "--paint",
        action="store_true",
        help="Also paint at low nside and compare summed y-maps (slow).",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=128,
        help="nside for painting comparison (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.shells:
        shell_files = list(args.shells)
    else:
        healcounts_dir: Path = args.healcounts_dir
        if not healcounts_dir.exists():
            print(f"Healcounts directory not found: {healcounts_dir}")
            return
        shell_files = sorted(healcounts_dir.glob("*.asdf"))

    if not shell_files:
        print(f"No healcounts .asdf files found in {args.healcounts_dir}")
        return

    catalogs = discover_halo_catalogs(HALO_CATALOGS_PATH)
    print(f"Found {len(shell_files)} shell(s), {len(catalogs)} halo catalog file(s).")
    print(f"theta_search={THETA_SEARCH:.6e}")

    count_comparison(shell_files, catalogs)

    if args.paint:
        paint_comparison(shell_files, catalogs, nside=args.nside)


if __name__ == "__main__":
    main()
