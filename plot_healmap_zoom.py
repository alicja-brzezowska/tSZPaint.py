"""
Standalone script to plot a zoomed gnomview of a total-particle HEALPix map.

Usage:
    python plot_healmap_zoom.py [path/to/file.asdf] [--ra RA] [--dec DEC] [--out output.png]

Defaults to the first available total heal-counts file and zooms to the brightest pixel.
"""

import argparse
import sys
from pathlib import Path

import asdf
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

HEALCOUNTS_TOTAL_PATH = Path(
    "/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/total/heal-counts"
)

NSIDE = 1024  # adjust if your maps use a different nside


def load_healmap(filepath: Path) -> np.ndarray:
    with asdf.open(filepath) as af:
        keys = list(af.keys())
        print(f"File keys: {keys}")
        # try common key names
        for key in ["heal_counts", "healcounts", "map", "counts", keys[0]]:
            if key in af:
                data = np.array(af[key])
                print(f"Loaded '{key}': shape={data.shape} dtype={data.dtype}")
                return data
    raise KeyError(f"Could not find map array in {filepath}. Keys: {keys}")


def plot_zoom(
    healmap: np.ndarray,
    nside: int,
    ra: float | None = None,
    dec: float | None = None,
    output: str | None = None,
    xsize: int = 2000,
    scale: float = 6.0,
):
    if ra is None or dec is None:
        ipix = int(np.nanargmax(healmap))
        theta, phi = hp.pix2ang(nside, ipix, nest=True)
        ra = np.degrees(phi)
        dec = 90.0 - np.degrees(theta)
        print(f"Zooming to brightest pixel: RA={ra:.3f} Dec={dec:.3f}")
    else:
        print(f"Zooming to specified RA={ra:.3f} Dec={dec:.3f}")

    reso = hp.nside2resol(nside, arcmin=True) / scale

    map_data = np.log10(healmap.astype(float) + 1)  # log10(counts + 1)

    plt.figure(figsize=(10, 10))
    hp.gnomview(
        map_data,
        rot=[ra, dec],
        xsize=xsize,
        reso=reso,
        nest=True,
        title=f"Total particle counts (RA={ra:.2f} Dec={dec:.2f})",
        unit="log10(counts + 1)",
        hold=True,
    )
    hp.graticule()
    plt.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot zoomed HEALPix total particle map")
    parser.add_argument(
        "filepath",
        nargs="?",
        help="Path to .asdf heal-counts file (default: first file in total/heal-counts/)",
    )
    parser.add_argument("--ra", type=float, default=None, help="RA centre for zoom (degrees)")
    parser.add_argument("--dec", type=float, default=None, help="Dec centre for zoom (degrees)")
    parser.add_argument("--nside", type=int, default=NSIDE, help=f"HEALPix nside (default: {NSIDE})")
    parser.add_argument("--scale", type=float, default=6.0, help="Zoom scale factor (default: 6)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: display)")
    args = parser.parse_args()

    if args.filepath:
        filepath = Path(args.filepath)
    else:
        files = sorted(HEALCOUNTS_TOTAL_PATH.glob("*.asdf"))
        if not files:
            print(f"No .asdf files found in {HEALCOUNTS_TOTAL_PATH}", file=sys.stderr)
            sys.exit(1)
        filepath = files[0]
        print(f"No file specified, using: {filepath.name}")

    healmap = load_healmap(filepath)

    # infer nside from map size if possible
    nside = args.nside
    try:
        nside = hp.npix2nside(len(healmap))
        print(f"Inferred nside={nside} from map length {len(healmap)}")
    except Exception:
        print(f"Using nside={nside}")

    plot_zoom(healmap, nside, ra=args.ra, dec=args.dec, output=args.out, scale=args.scale)


if __name__ == "__main__":
    main()
