"""
Combine HOD LRG catalogs from multiple snapshots, then filter by comoving distance
(chi) to produce one catalog per healpix step range.

HOD outputs (LRGs.dat in ECSV format) contain x, y, z Cartesian coords in Mpc/h.
chi = sqrt(x^2 + y^2 + z^2).

Step ranges and chi cuts:
  Step0665-0670: chi in [1411.9, 1441.5]
  Step0671-0676: chi in [1382.0, 1411.9]
  Step0677-0682: chi in [1352.3, 1382.0]
"""

from pathlib import Path
import numpy as np
from astropy.table import Table, vstack

HOD_MOCK_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/hod_mocks/lightcone_halos")
OUTPUT_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/lrg_catalogs")

# Snapshots to combine (order doesn't matter)
SNAPSHOTS = ["z0.503", "z0.542"]

STEPS = [
    {"name": "Step0665-0670", "chi_min": 1411.9, "chi_max": 1441.5},
    {"name": "Step0671-0676", "chi_min": 1382.0, "chi_max": 1411.9},
    {"name": "Step0677-0682", "chi_min": 1352.3, "chi_max": 1382.0},
]


def load_lrg(z_snapshot: str) -> Table:
    path = HOD_MOCK_DIR / z_snapshot / "galaxies_rsd" / "LRGs.dat"
    print(f"  Loading {path} ...")
    t = Table.read(path, format="ascii.ecsv")
    print(f"    -> {len(t)} LRGs")
    return t


def filter_by_chi(table: Table, chi_min: float, chi_max: float) -> Table:
    x = np.asarray(table["x"])
    y = np.asarray(table["y"])
    z = np.asarray(table["z"])
    chi = np.sqrt(x**2 + y**2 + z**2)
    mask = (chi >= chi_min) & (chi <= chi_max)
    return table[mask]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: stack all snapshots into one combined catalog
    print("Stacking snapshots:")
    tables = [load_lrg(snap) for snap in SNAPSHOTS]
    combined = vstack(tables)
    print(f"Combined total: {len(combined)} LRGs\n")

    # Step 2: filter combined catalog by chi range for each step
    for step in STEPS:
        name = step["name"]
        chi_min = step["chi_min"]
        chi_max = step["chi_max"]

        filtered = filter_by_chi(combined, chi_min, chi_max)
        print(f"{name}: chi=[{chi_min}, {chi_max}] -> {len(filtered)} LRGs")

        out_path = OUTPUT_DIR / f"{name}_LRGs.dat"
        filtered.write(out_path, format="ascii.ecsv", overwrite=True)
        print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
