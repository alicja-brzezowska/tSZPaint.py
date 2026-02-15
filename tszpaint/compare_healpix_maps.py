import asdf
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from tszpaint.config import ABACUS_DATA_PATH, HALO_CATALOGS_PATH, HEALCOUNTS_PATH, HEALCOUNTS_TOTAL_PATH
from tszpaint.abacus_loader import load_abacus_healcounts

total_file = HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0671-0676.asdf"
halo_file = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"


def plot_zoom(y_map, nside, outpng="y_map_zoom.png"):
    """Zoom to brightest pixel."""
    ipix = int(np.nanargmax(y_map))
    theta, phi = hp.pix2ang(nside, ipix, nest=True)
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)

    pix_arcmin = hp.nside2resol(nside, arcmin=True)
    reso = pix_arcmin / 6.0
    xsize = 4000

    plt.figure(figsize=(10, 10))
    hp.gnomview(
        np.log10(y_map + 10**-20),
        rot=[lon, lat],
        xsize=xsize,
        reso=reso,
        nest=True,
        title=f"Zoomed (nside={nside})",
        unit="log10(y)",
        hold=True,
    )
    hp.graticule()

    plt.savefig(outpng, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")


def plot_ra_dec(y_map, nside, outpng="y_map_zoom_radec.png"):
    """Zoom to specific RA/Dec."""
    ra_deg = 140.609
    dec_deg = -0.047

    pix_arcmin = hp.nside2resol(nside, arcmin=True)
    reso = pix_arcmin / 6.0
    xsize = 4000

    plt.figure(figsize=(10, 10))
    hp.gnomview(
        np.log10(y_map + 1),
        rot=[ra_deg, dec_deg],
        xsize=xsize,
        reso=reso,
        nest=True,
        title=f"Painting output; (RA={ra_deg:.3f} Dec={dec_deg:.3f})",
        unit="log10(y)",
        hold=True,
    )
    hp.graticule()
    plt.savefig(outpng, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")



def main():
    halo_counts = load_abacus_healcounts(halo_file)
    total_counts = load_abacus_healcounts(total_file)

    nside = hp.npix2nside(len(halo_counts))

    plot_ra_dec(halo_counts, nside, outpng="halo_counts_zoom.png")
    plot_ra_dec(total_counts, nside, outpng="total_counts_zoom.png")


if __name__ == "__main__":
    main()