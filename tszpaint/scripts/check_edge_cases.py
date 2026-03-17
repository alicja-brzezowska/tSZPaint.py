from __future__ import annotations

from pathlib import Path

import asdf
import numpy as np

from tszpaint.config import ABACUS_DATA_PATH, HALO_CATALOGS_PATH
from tszpaint.paint.match_redshifts import discover_halo_catalogs


FOLDER = "z_0.542"
THETA_SEARCH = 1.6732e-3


def _halo_count_in_file(halo_file: Path) -> int:
    with asdf.open(halo_file) as af:
        chi = np.asarray(af["halo_lightcone"]["Interpolated_ComovingDist"])
    return int(chi.size)


def _halo_count_in_file_with_chi_cut(halo_file: Path, chi_min: float, chi_max: float) -> int:
    with asdf.open(halo_file) as af:
        chi = np.asarray(af["halo_lightcone"]["Interpolated_ComovingDist"])
    return int(np.count_nonzero((chi >= chi_min) & (chi <= chi_max)))


def _read_chi_range_from_particle_counts(filepath: Path) -> tuple[float, float]:
    with asdf.open(filepath) as af:
        headers = af["headers"]
        chis = np.asarray([h["CoordinateDistanceHMpc"] for h in headers], dtype=np.float64)
    return float(chis.min()), float(chis.max())


def main():
    output_root = ABACUS_DATA_PATH / "for_tszpaint"
    shell_dir = output_root / FOLDER
    particle_counts_file = shell_dir / "particle_counts.asdf"
    merged_halo_file = shell_dir / "halo_catalog.asdf"

    chi_min, chi_max = _read_chi_range_from_particle_counts(particle_counts_file)
    chi_mid = 0.5 * (chi_min + chi_max)
    delta_chi = chi_mid * THETA_SEARCH
    chi_min_padded = chi_min - delta_chi
    chi_max_padded = chi_max + delta_chi

    catalogs = discover_halo_catalogs(HALO_CATALOGS_PATH)
    matched_halo_files = [
        c.file_path
        for c in catalogs
        if max(chi_min_padded, c.chi_min) <= min(chi_max_padded, c.chi_max)
    ]

    matched_before_cut = sum(_halo_count_in_file(fp) for fp in matched_halo_files)
    matched_after_cut = sum(
        _halo_count_in_file_with_chi_cut(fp, chi_min_padded, chi_max_padded)
        for fp in matched_halo_files
    )
    allocated_after_cut = _halo_count_in_file(merged_halo_file)

    print(f"folder={shell_dir.name}")
    print(f"theta_search={THETA_SEARCH:.6e}")
    print(f"delta_chi={delta_chi:.6f}")
    print(f"halos_matched_before_cut={matched_before_cut}")
    print(f"halos_matched_after_cut={matched_after_cut}")
    print(f"halos_allocated_in_output={allocated_after_cut}")


if __name__ == "__main__":
    main()
