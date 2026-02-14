import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def plot_zoom(
    y_map: np.ndarray,
    nside: int,
    halo_theta: np.ndarray | None = None,
    halo_phi: np.ndarray | None = None,
    outpng: str = "y_map_zoom.png",
):
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

    # Plot halo centers to check consistency with healpix pixels;
    if halo_theta is not None and halo_phi is not None:
        halo_lon = np.degrees(halo_phi)
        halo_lat = 90.0 - np.degrees(halo_theta)
        hp.projscatter(
            halo_lon,
            halo_lat,
            lonlat=True,
            marker="x",
            color="red",
            s=20,
            alpha=0.7,
            label="Halo centers",
        )
        plt.legend()

    plt.savefig(outpng, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")


def plot_ra_dec(
    y_map, nside, halo_theta=None, halo_phi=None, outpng="y_map_zoom_radec.png"
):
    """Zoom to specific RA/Dec."""
    ra_deg = 140.609
    dec_deg = -0.047

    pix_arcmin = hp.nside2resol(nside, arcmin=True)
    reso = pix_arcmin / 6.0
    xsize = 4000

    plt.figure(figsize=(10, 10))
    hp.gnomview(
        np.log10(y_map + 10**-20),
        rot=[ra_deg, dec_deg],
        xsize=xsize,
        reso=reso,
        nest=True,
        title=f"Painting output; merged 3 files (RA={ra_deg:.3f} Dec={dec_deg:.3f})",
        unit="log10(y)",
        hold=True,
    )
    hp.graticule()

    if halo_theta is not None and halo_phi is not None:
        halo_lon = np.degrees(halo_phi)
        halo_lat = 90.0 - np.degrees(halo_theta)
        hp.projscatter(
            halo_lon,
            halo_lat,
            lonlat=True,
            marker="x",
            color="red",
            s=20,
            alpha=0.7,
            label="Halo centers",
        )
        plt.legend()

    plt.savefig(outpng, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")


def plot_Y_vs_M(
    M_halos,
    Y_per_halo,
    outpng="Y_vs_M.png",
    nbins_plot=80,
):
    """Binned log-log plot of integrated Y vs halo mass with weighted linear fit."""
    mask = (
        (M_halos > 0)
        & (Y_per_halo > 0)
        & np.isfinite(M_halos)
        & np.isfinite(Y_per_halo)
    )
    logM = np.log10(M_halos[mask])
    logY = np.log10(Y_per_halo[mask])

    bins = np.linspace(logM.min(), logM.max(), nbins_plot + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    y_mean = np.full(nbins_plot, np.nan)
    y_err = np.full(nbins_plot, np.nan)

    for i in range(nbins_plot):
        in_bin = (logM >= bins[i]) & (logM < bins[i + 1])
        n = in_bin.sum()
        if n < 5:
            continue
        vals = logY[in_bin]
        y_mean[i] = np.mean(vals)
        y_err[i] = np.std(vals, ddof=1) / np.sqrt(n)

    good = np.isfinite(y_mean) & np.isfinite(y_err) & (y_err > 0)

    # Weighted linear fit: logY = intercept + slope * logM
    # np.polyfit w = 1/sigma for WLS (it squares internally)
    coeffs, cov = np.polyfit(
        centers[good],
        y_mean[good],
        1,
        w=1.0 / y_err[good],
        cov=True,
    )
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0, 0])

    _, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        centers[good],
        y_mean[good],
        yerr=y_err[good],
        fmt="o",
        ms=4,
        capsize=2,
        label=r"Binned mean $\pm\,\sigma/\sqrt{N}$",
    )

    xx = np.linspace(centers[good].min(), centers[good].max(), 200)
    ax.plot(
        xx,
        intercept + slope * xx,
        "-",
        lw=2,
        label=f"Fit: slope = {slope:.3f} $\\pm$ {slope_err:.3f}",
    )

    ax.set_xlabel(r"$\log_{10}\,M_{200c}\;[M_\odot]$")
    ax.set_ylabel(r"$\log_{10}\,Y_{200}\;[\mathrm{sr}]$")
    ax.set_title("Integrated Compton-$y$ vs Halo Mass")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpng}")
    print(f"  Slope = {slope:.3f} +/- {slope_err:.3f}")
    print(f"  Intercept = {intercept:.3f}")
