from dataclasses import dataclass

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from tszpaint.paint.abacus_loader import SimulationData


@dataclass
class Visualizer:
    data: SimulationData
    y_map: np.ndarray
    y_per_halo: np.ndarray
    nside: int
    output_file_stub: str | None = None
    scale: float = 6.0
    output_png_dpi: int = 250

    @property
    def resolution(self) -> float:
        pix_arcmin: float = hp.nside2resol(self.nside, arcmin=True)
        return pix_arcmin / self.scale

    def finalize_plot(self, suffix: str):
        plt.tight_layout()
        if stub := self.output_file_stub:
            outpath = f"{stub}_{suffix}.png"
            plt.savefig(outpath, dpi=self.output_png_dpi, bbox_inches="tight")
            logger.info(f"Saved: {outpath}")
        else:
            plt.show()
        plt.close()

    def plot_zoom(self):
        """Zoom to brightest pixel."""
        ipix = int(np.nanargmax(self.y_map))
        theta, phi = hp.pix2ang(self.nside, ipix, nest=True)
        lon = np.degrees(phi)
        lat = 90.0 - np.degrees(theta)

        xsize = 4000

        plt.figure(figsize=(10, 10))
        hp.gnomview(
            np.log10(self.y_map + 10**-20),
            rot=[lon, lat],
            xsize=xsize,
            reso=self.resolution,
            nest=True,
            title=f"Zoomed (nside={self.nside})",
            unit="log10(y)",
            hold=True,
        )
        hp.graticule()

        # Plot halo centers to check consistency with healpix pixels;
        halo_lon = np.degrees(self.data.phi)
        halo_lat = 90.0 - np.degrees(self.data.theta)
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
        self.finalize_plot("y_zoom")

    def plot_ra_dec(self):
        """Zoom to specific RA/Dec."""
        ra_deg = 140.609
        dec_deg = -0.047

        xsize = 4000

        plt.figure(figsize=(10, 10))
        hp.gnomview(
            np.log10(self.y_map + 10**-20),
            rot=[ra_deg, dec_deg],
            xsize=xsize,
            reso=self.resolution,
            nest=True,
            title=f"Painting output; merged 3 files (RA={ra_deg:.3f} Dec={dec_deg:.3f})",
            unit="log10(y)",
            hold=True,
        )
        hp.graticule()

        halo_lon = np.degrees(self.data.phi)
        halo_lat = 90.0 - np.degrees(self.data.theta)
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
        self.finalize_plot("ra_dec")

    def plot_Y_vs_M(
        self,
        outpng="Y_vs_M.png",
        nbins_plot=80,
    ):
        """Binned log-log plot of integrated Y vs halo mass with weighted linear fit."""
        mask = (
            (self.data.m_halos > 0)
            & (self.y_per_halo > 0)
            & np.isfinite(self.data.m_halos)
            & np.isfinite(self.y_per_halo)
        )
        logM = np.log10(self.data.m_halos[mask])
        logY = np.log10(self.y_per_halo[mask])

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
        self.finalize_plot("y_vs_m")
        print(f"  Slope = {slope:.3f} +/- {slope_err:.3f}")
        print(f"  Intercept = {intercept:.3f}")

    def visualize_y_map(self):
        hp.mollview(
            self.y_map,
            title="tSZ y-map on real data (z = 0.542)",
            unit="y",
            norm="log",
            min=1e-12,
            nest=True,
        )
        hp.graticule()
        self.finalize_plot("y_map")
