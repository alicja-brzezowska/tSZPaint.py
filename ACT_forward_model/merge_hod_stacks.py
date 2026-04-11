"""
Merge all hod_*.npz files into a single full_grid.npz.

Final shapes:
    params       : (N_total, 9)  — [alpha_pressure, beta0, gamma, log10P0,
                                      log_Mcut, log_M1, sigma, alpha_hod, kappa]
    profiles_cap : (N_total, 9)  — CAP stacked signal per aperture
    apertures_cap : (9,)
    param_names  : (9,)

where N_total = N_hod_points × 125  (e.g. 200 × 125 = 25000)
"""

from pathlib import Path
import numpy as np
from loguru import logger

HOD_DIR  = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/hod_lhc")
OUT_FILE = HOD_DIR / "full_grid.npz"


def main():
    npz_files = sorted(HOD_DIR.glob("hod_*.npz"))
    if not npz_files:
        raise RuntimeError(f"No hod_*.npz files found in {HOD_DIR}")
    logger.info(f"Merging {len(npz_files)} files from {HOD_DIR}")

    all_params, all_cap = [], []
    apertures_cap = param_names = None

    for f in npz_files:
        d = np.load(f, allow_pickle=True)

        # Build combined 9-param row for each of the 125 pressure maps
        hod_row      = d["hod_params"]                                   # (5,)
        press_block  = d["pressure_params"]                              # (125, 4)
        hod_block    = np.tile(hod_row, (len(press_block), 1))           # (125, 5)
        params_block = np.hstack([press_block, hod_block])               # (125, 9)

        all_params.append(params_block)
        all_cap.append(d["profiles_cap"])                                # (125, 9)

        if apertures_cap is None:
            apertures_cap = d["apertures_cap"]
            param_names   = list(d["param_names_pressure"]) + list(d["param_names_hod"])

    params       = np.vstack(all_params)   # (N_total, 9)
    profiles_cap = np.vstack(all_cap)      # (N_total, 9)

    logger.info(f"params shape:       {params.shape}")
    logger.info(f"profiles_cap shape: {profiles_cap.shape}")

    np.savez(
        OUT_FILE,
        params        = params,
        profiles_cap  = profiles_cap,
        apertures_cap = apertures_cap,
        param_names   = param_names,
    )
    logger.info(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
