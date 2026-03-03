from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from tszpaint.logging import time_calls
from tszpaint.paint.abacus_loader import SimulationData
from tszpaint.scripts.radial_profile import RadialProfile


@dataclass
class PlotConfig:
    """Configuration for different plot modes."""

    mode: Literal["standard", "healpix"]
    log_offset: float
    title_prefix: str
    unit: str

    @classmethod
    def standard(cls):
        return cls("standard", 10**-20, "Painting output", "log10(y)")

    @classmethod 
    def healpix(cls): # for initial maps: pre-painting 
        return cls("healpix", 1, "Healpix output", "log10(counts+1)")


@dataclass
class Visualizer:
    """Visualization tools for HEALPix maps"""

    nside: int
    output_file_stub: str | None = None
    scale: float = 6.0  # for zoomed plots
    output_png_dpi: int = 250  # for zoomed plots

    @staticmethod
    def validate_config_and_sim_data(
        config: PlotConfig, sim_data: SimulationData | None
    ):
        if config.mode == "standard" and sim_data is None:
            raise ValueError(
                "Can't plot with `standard` PlotConfig without passing simulation data for halo centers"
            )

    @property
    def resolution(self) -> float:
        pix_arcmin: float = hp.nside2resol(self.nside, arcmin=True)
        return pix_arcmin / self.scale

    def finalize_plot(self, suffix: str):
        plt.tight_layout()
        if stub := self.output_file_stub:
            outpath = f"{stub}_{suffix}.png"
            Path(outpath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outpath, dpi=self.output_png_dpi, bbox_inches="tight")
            logger.info(f"Saved: {outpath}")
        else:
            plt.show()
        plt.close()

    def _plot_gnomview(
        self,
        y_map: np.ndarray,
        rot: tuple[float, float],
        config: PlotConfig,
        suffix: str,
        sim_data: SimulationData | None = None,
        xsize: int = 4000,
        zoom_scale: float | None = None,
        show_halo_centers: bool = True,
    ):
        """Core gnomonic projection plotting logic."""
        plt.figure(figsize=(10, 10))

        map_data = np.log10(y_map + config.log_offset)
        title = f"{config.title_prefix} (RA={rot[0]:.3f} Dec={rot[1]:.3f})"

        hp.gnomview(
            map_data,
            rot=list(rot),
            xsize=xsize,
            reso=hp.nside2resol(self.nside, arcmin=True)
            / (zoom_scale if zoom_scale is not None else self.scale),
            nest=True,
            title=title,
            unit=config.unit,
            hold=True,
        )
        hp.graticule()

        if sim_data is not None and show_halo_centers:
            halo_lon = np.degrees(sim_data.phi)
            halo_lat = 90.0 - np.degrees(sim_data.theta)
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

        self.finalize_plot(suffix)

    @time_calls
    def plot_zoom(
        self,
        y_map: np.ndarray,
        config: PlotConfig,
        sim_data: SimulationData | None = None,
    ):
        """Zoom to brightest pixel in the map."""
        self.validate_config_and_sim_data(config, sim_data)

        ipix = int(np.nanargmax(y_map))
        theta, phi = hp.pix2ang(self.nside, ipix, nest=True)
        lon = np.degrees(phi)
        lat = 90.0 - np.degrees(theta)

        suffix = "healpix_zoom" if config == PlotConfig.healpix() else "y_zoom"
        self._plot_gnomview(y_map, (lon, lat), config, suffix, sim_data)

    @time_calls
    def plot_ra_dec(
        self,
        y_map: np.ndarray,
        config: PlotConfig,
        ra_deg: float = 140.609,
        dec_deg: float = -0.047,
        sim_data: SimulationData | None = None,
        filename_suffix: str = "",
        zoom_scale: float | None = None,
        show_halo_centers: bool = True,
    ):
        """Zoom to specific RA/Dec coordinates."""
        self.validate_config_and_sim_data(config, sim_data)

        suffix = (
            f"healpix_ra_dec_zoom_{filename_suffix}"
            if config == PlotConfig.healpix()
            else f"y_ra_dec_zoom_{filename_suffix}"
        )

        self._plot_gnomview(
            y_map,
            (ra_deg, dec_deg),
            config,
            suffix,
            sim_data,
            zoom_scale=zoom_scale,
            show_halo_centers=show_halo_centers,
        )

    @time_calls
    def plot_Y_vs_M(
        self,
        sim_data: SimulationData,
        y_per_halo: np.ndarray,
        nbins_plot: int = 80,
    ):
        """Binned log-log plot of integrated Y vs halo mass with weighted linear fit."""
        mask = (
            (sim_data.m_halos > 0)
            & (y_per_halo > 0)
            & np.isfinite(sim_data.m_halos)
            & np.isfinite(y_per_halo)
        )
        logM = np.log10(sim_data.m_halos[mask])
        logY = np.log10(y_per_halo[mask])

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

        coeffs, cov = np.polyfit(
            centers[good],
            y_mean[good],
            1,
            w=1.0 / y_err[good],
            cov=True,
        )
        slope, intercept = coeffs
        slope_err = np.sqrt(cov[0, 0])

        _, ax = plt.subplots(figsize=(9, 7))
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

    @time_calls
    def plot_Y_vs_R200(self, radial_profiles: Iterable[RadialProfile], suffix: str = "y_vs_r200"):
        def _fmt_1sf(value: float) -> str:
            exponent = int(np.floor(np.log10(value)))
            mantissa = int(np.round(value / (10**exponent)))
            if mantissa == 10:
                mantissa = 1
                exponent += 1
            return rf"{mantissa}\times10^{{{exponent}}}"

        def _mass_bin_label(logm_center: float, halfwidth: float = 0.15) -> str:
            m_lo = 10 ** (logm_center - halfwidth)
            m_hi = 10 ** (logm_center + halfwidth)
            lo_tex = _fmt_1sf(m_lo)
            hi_tex = _fmt_1sf(m_hi)
            return rf"${lo_tex} < M_{{200c}} < {hi_tex}\,M_\odot$"

        radial_profiles = list(radial_profiles)
        _, ax = plt.subplots(figsize=(8, 6))

        common_good = np.ones_like(radial_profiles[0].x_centers, dtype=bool)
        for profile in radial_profiles:
            common_good &= (
                np.isfinite(profile.x_centers)
                & np.isfinite(profile.y_mean)
                & (profile.y_mean > 0)
            )

        for profile in radial_profiles:
            good = common_good
            x = profile.x_centers[good]
            y = profile.y_mean[good]
            yerr = profile.y_err[good]
            (line,) = ax.plot(
                x,
                y,
                "o",
                ms=4,
                label=_mass_bin_label(profile.logM_center),
            )
            line_color = line.get_color()
            y_lo = np.maximum(y - yerr, 1e-20)
            y_hi = y + yerr
            ax.fill_between(x, y_lo, y_hi, color=line_color, alpha=0.15, linewidth=0)

            ref_good = np.isfinite(profile.y_battaglia) & (profile.y_battaglia > 0)
            ax.plot(
                profile.x_ref[ref_good],
                profile.y_battaglia[ref_good],
                "--",
                lw=2,
                color=line_color,
                alpha=0.5,
                label="_nolegend_",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$r/R_{200}$", fontsize=16)
        ax.set_ylabel(r"$y$", fontsize=16)
        ax.set_title(
            "Compton-$y$ profile vs $r/R_{200}$",
            fontsize=18,
        )
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=13)
        ax.legend(fontsize=10)
        self.finalize_plot(suffix)

    @time_calls
    def visualize_y_map(self, y_map: np.ndarray):
        """Full-sky Mollweide projection of the y-map."""
        hp.mollview(
            y_map,
            title="tSZ y-map",
            unit="y",
            norm="log",
            min=1e-12,
            nest=True,
        )
        hp.graticule()
        self.finalize_plot("y_map")
