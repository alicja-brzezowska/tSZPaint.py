"""
Zoomed gnomonic plots of Step0628-0634 vs Step0635-0640 y-maps.

Finds the brightest pixel in Step0628-0634, then renders a zoomed gnomview
for both steps at that location so you can compare them side-by-side.

Usage (interactively or via Slurm):
    python -m tszpaint.scripts.check_step0628_zoom
"""
import matplotlib
matplotlib.use("Agg")

import asdf
import healpy as hp
import numpy as np
from pathlib import Path

from tszpaint.config import OUTPUT_PATH
from tszpaint.paint.visualize import Visualizer, PlotConfig

NSIDE = 8192
STEPS = ["Step0628-0634", "Step0635-0640"]
# Use first file (arbitrary param set — just for visual check)
DATA_DIR = OUTPUT_PATH
OUT_DIR = OUTPUT_PATH / "visualization" / "step0628_check"


def load_y_map(step: str, fname: str) -> np.ndarray:
    fpath = DATA_DIR / step / fname
    with asdf.open(str(fpath), lazy_load=True) as f:
        return np.asarray(f.tree["data"]["y_map"], dtype=np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fname = sorted((DATA_DIR / STEPS[0]).glob("*.asdf"))[0].name
    print(f"File: {fname}")

    # Load Step0628-0634 to find brightest pixel
    print("Loading Step0628-0634 ...")
    y_ref = load_y_map(STEPS[0], fname)
    ipix = int(np.nanargmax(y_ref))
    theta, phi = hp.pix2ang(NSIDE, ipix, nest=True)
    lon = float(np.degrees(phi))
    lat = float(90.0 - np.degrees(theta))
    print(f"Brightest pixel: ipix={ipix}, RA={lon:.3f}, Dec={lat:.3f}, y={y_ref[ipix]:.3e}")
    print(f"Non-zero: {np.count_nonzero(y_ref)/y_ref.size*100:.2f}%, sum={y_ref.sum():.4e}")

    # config: use standard log_offset, skip halo-center overlay
    cfg = PlotConfig.standard()

    for step in STEPS:
        print(f"\nPlotting {step} ...")
        if step == STEPS[0]:
            y = y_ref
        else:
            print(f"Loading {step} ...")
            y = load_y_map(step, fname)
            print(f"Non-zero: {np.count_nonzero(y)/y.size*100:.2f}%, sum={y.sum():.4e}")

        stub = str(OUT_DIR / step)
        viz = Visualizer(nside=NSIDE, output_file_stub=stub)

        # zoom to the brightest-pixel location (from Step0628-0634)
        viz._plot_gnomview(
            y,
            rot=(lon, lat),
            config=cfg,
            suffix="y_zoom",
            sim_data=None,
            show_halo_centers=False,
        )
        print(f"  Saved: {stub}_y_zoom.png")
        del y

    print("\nDone. Plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
