from __future__ import annotations

from pathlib import Path

import asdf
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from tszpaint.scripts.radial_profile import RadialProfile

MAP_PATH = Path(
    "/home/ab2927/rds/tSZPaint.py/data/visualization/2026-03-03/triaxial_small_mass.asdf"
)


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
    return rf"${lo_tex}\,M_\odot < M < {hi_tex}\,M_\odot$"


def _plot_two_halo_for_mass(total: RadialProfile, isolated: RadialProfile):
    x = np.asarray(total.x_centers, dtype=np.float64)
    y_total = np.asarray(total.y_mean, dtype=np.float64)
    y_one = np.asarray(isolated.y_mean, dtype=np.float64)
    pct_diff = np.where(y_one != 0.0, (y_total / y_one - 1.0) * 100.0, np.nan)

    x_dense = np.logspace(np.log10(x.min()), np.log10(x.max()), 400)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    ax_top.plot(x, y_total, "o", ms=4, label=r"Total ($1h+2h$)")
    ax_top.plot(x_dense, np.interp(x_dense, x, y_total), lw=2.2, label="_nolegend_")
    ax_top.plot(x, y_one, "o", ms=4, label=r"One-halo (1h)")
    ax_top.plot(x_dense, np.interp(x_dense, x, y_one), lw=2.2, label="_nolegend_")
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylim(bottom=1e-10)
    ax_top.set_ylabel(r"$y$", fontsize=16)
    ax_top.tick_params(axis="both", which="major", labelsize=13)
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=12)

    ax_bottom.plot(x, pct_diff, lw=2.0)
    ax_bottom.axhline(0.0, color="k", lw=1.0, alpha=0.5)
    ax_bottom.set_xscale("log")
    ax_bottom.set_xlim(right=1e1)
    ax_bottom.set_xlabel(r"$r/R_{200}$", fontsize=16)
    ax_bottom.set_ylabel(r"$\%\,(h_{1+2}-h_{1})$", fontsize=14)
    ax_bottom.set_ylim(0.0, 500.0)
    ax_bottom.tick_params(axis="both", which="major", labelsize=12)
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()

    out = MAP_PATH.with_stem(MAP_PATH.stem + f"_halo_terms_logM{total.logM_center:.1f}")
    fig.savefig(out.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {out.with_suffix('.png')}")


def _plot_all_masses_halo_terms(
    total_profiles: list[RadialProfile],
    isolated_by_mass: dict[float, RadialProfile],
):
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
    )

    common_top_good = np.ones_like(total_profiles[0].x_centers, dtype=bool)
    common_bottom_good = np.ones_like(total_profiles[0].x_centers, dtype=bool)
    for total in total_profiles:
        isolated = isolated_by_mass[total.logM_center]
        x = np.asarray(total.x_centers, dtype=np.float64)
        y_total = np.asarray(total.y_mean, dtype=np.float64)
        y_one = np.asarray(isolated.y_mean, dtype=np.float64)
        pct_diff = np.where(y_one != 0.0, (y_total / y_one - 1.0) * 100.0, np.nan)
        common_top_good &= np.isfinite(x) & np.isfinite(y_total) & (y_total > 0)
        common_bottom_good &= np.isfinite(x) & np.isfinite(pct_diff)

    for total in total_profiles:
        isolated = isolated_by_mass[total.logM_center]
        x = np.asarray(total.x_centers, dtype=np.float64)
        y_total = np.asarray(total.y_mean, dtype=np.float64)
        yerr_total = np.asarray(total.y_err, dtype=np.float64)
        y_one = np.asarray(isolated.y_mean, dtype=np.float64)
        pct_diff = np.where(y_one != 0.0, (y_total / y_one - 1.0) * 100.0, np.nan)

        good = common_top_good
        x_plot = x[good]
        y_plot = y_total[good]
        yerr_plot = yerr_total[good]

        (line,) = ax_top.plot(
            x_plot,
            y_plot,
            "o",
            ms=4,
            label=_mass_bin_label(total.logM_center),
        )
        color = line.get_color()
        y_lo = np.maximum(y_plot - yerr_plot, 1e-20)
        y_hi = y_plot + yerr_plot
        ax_top.fill_between(x_plot, y_lo, y_hi, color=color, alpha=0.15, linewidth=0)

        ref_x = np.asarray(total.x_ref, dtype=np.float64)
        ref_y = np.asarray(total.y_battaglia, dtype=np.float64)
        ref_good = np.isfinite(ref_x) & np.isfinite(ref_y) & (ref_y > 0)
        ax_top.plot(
            ref_x[ref_good],
            ref_y[ref_good],
            "--",
            lw=2,
            color=color,
            alpha=0.5,
            label="_nolegend_",
        )

        pct_good = common_bottom_good
        ax_bottom.plot(x[pct_good], pct_diff[pct_good], "--", lw=1.8, color=color)

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylim(bottom=1e-10)
    ax_top.set_ylabel(r"$y$", fontsize=16)
    ax_top.tick_params(axis="both", which="major", labelsize=13)
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=10)

    ax_bottom.axhline(0.0, color="k", lw=1.0, alpha=0.5)
    ax_bottom.set_xscale("log")
    ax_bottom.set_xlim(right=1e1)
    ax_bottom.set_xlabel(r"$r/R_{200}$", fontsize=16)
    ax_bottom.set_ylabel(r"$\%\,(h_{1+2}-h_{1})$", fontsize=14)
    ax_bottom.set_ylim(0.0, 500.0)
    ax_bottom.tick_params(axis="both", which="major", labelsize=12)
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    out = MAP_PATH.with_stem(MAP_PATH.stem + "_halo_terms_all_masses")
    fig.savefig(out.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved to {out.with_suffix('.png')}")


def main():
    with asdf.open(MAP_PATH) as af:
        data = af.tree["data"]
        total_profiles = [RadialProfile.from_dict(d) for d in data["radial_profile"]]
        isolated_profiles = [
            RadialProfile.from_dict(d) for d in data["radial_profile_isolated"]
        ]

    isolated_by_mass = {rp.logM_center: rp for rp in isolated_profiles}
    #for total in total_profiles:
    #    isolated = isolated_by_mass[total.logM_center]
    #    _plot_two_halo_for_mass(total, isolated)

    _plot_all_masses_halo_terms(total_profiles, isolated_by_mass)


if __name__ == "__main__":
    main()
