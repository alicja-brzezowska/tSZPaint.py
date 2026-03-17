from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter

from tszpaint.config import (
    HALO_CATALOGS_PATH,
    HEALCOUNTS_TOTAL_PATH,
    OUTPUT_PATH,
)
from tszpaint.logging import setup_logging
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import compute_theta_200, get_angular_size_from_comoving
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.paint import MODEL, get_real_space_from_eigenvals
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.y_profile.y_profile import (
    P_E_FACTOR,
    compute_R_delta,
    generalized_nfw,
    get_params,
)


def tangent_basis(n0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(n0, ref)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = np.cross(ref, n0)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n0, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2


def choose_halos_by_mass_targets(log_masses: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(log_masses) < len(targets):
        raise ValueError(f"Requested {len(targets)} halos but only {len(log_masses)} are available")

    chosen: list[int] = []
    for target in targets:
        order = np.argsort(np.abs(log_masses - target))
        picked = None
        for candidate in order:
            cand = int(candidate)
            if cand not in chosen:
                picked = cand
                break
        if picked is None:
            raise ValueError("Could not choose unique halos for requested mass targets")
        chosen.append(picked)

    return np.array(chosen, dtype=np.int64)


def _per_halo_slice(starts: np.ndarray, counts: np.ndarray, i: int) -> slice:
    start = int(starts[i])
    count = int(counts[i])
    return slice(start, start + count)


def _analytic_los_values(
    *,
    nside: int,
    halo_xyz: np.ndarray,
    m_halos: np.ndarray,
    redshift: float,
    semi_axes_angular_base: np.ndarray,
    eigenvectors: np.ndarray,
    halo_starts: np.ndarray,
    halo_counts: np.ndarray,
    pixel_indices: np.ndarray,
    n_los_samples: int,
    los_s_max_r200: float,
) -> np.ndarray:
    n_pairs = len(pixel_indices)
    y_values = np.zeros(n_pairs, dtype=np.float64)
    if n_pairs == 0:
        return y_values

    n_halos = len(halo_xyz)
    if n_halos == 0:
        return y_values

    s_grid = np.linspace(-los_s_max_r200, los_s_max_r200, n_los_samples, dtype=np.float64)
    rho_crit_kg_m3 = MODEL.cosmo.critical_density(redshift).to("kg/m3").value
    theta200 = compute_theta_200(MODEL, m_halos, redshift, delta=200)

    for i in range(n_halos):
        sl = _per_halo_slice(halo_starts, halo_counts, i)
        pix = pixel_indices[sl]
        if len(pix) == 0:
            continue

        n0 = halo_xyz[i]
        t1, t2 = tangent_basis(n0)

        px, py, pz = hp.pix2vec(nside, pix, nest=True)
        p = np.stack([px, py, pz], axis=1).astype(np.float64)

        denom = np.clip(p @ n0, 1e-14, None)
        u = (p @ t1) / denom
        v = (p @ t2) / denom

        los_ang = s_grid * max(float(theta200[i]), 1e-14)
        rvec = (
            u[:, None, None] * t1[None, None, :]
            + v[:, None, None] * t2[None, None, :]
            + los_ang[None, :, None] * n0[None, None, :]
        )

        evec = np.asarray(eigenvectors[i], dtype=np.float64)
        comp = rvec @ evec.T

        axes = np.maximum(np.asarray(semi_axes_angular_base[i], dtype=np.float64), 1e-14)
        m_ell = np.sqrt(np.sum((comp / axes[None, None, :]) ** 2, axis=2))

        xc, alpha, beta, gamma, P0 = get_params(float(m_halos[i]), redshift)
        gnfw = generalized_nfw(m_ell, xc, alpha, beta, gamma)
        p_tilde = P0 * np.trapezoid(gnfw, s_grid, axis=1)

        mass_kg = float(m_halos[i]) * 1.98847e30
        norm = 6.67430e-11 * mass_kg * 200.0 * rho_crit_kg_m3 * MODEL.f_b / 2.0
        y_values[sl] = 0.5176 * norm * p_tilde * P_E_FACTOR

    return y_values


def build_analytic_mode(
    mode: str,
    config: PainterConfig,
    halo_xyz: np.ndarray,
    radii_halos: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    m_halos: np.ndarray,
    redshift: float,
    n_los_samples: int,
    los_s_max_r200: float,
    axis_norm_mode: str,
):
    n_halos = len(halo_xyz)
    if mode == "triaxial":
        raw_axes_comoving = get_real_space_from_eigenvals(eigenvalues, radii_halos)
        evec_use = eigenvectors
    else:
        spherical_r_comoving = np.asarray(radii_halos, dtype=np.float64)
        raw_axes_comoving = np.repeat(spherical_r_comoving[:, None], 3, axis=1)
        evec_use = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n_halos, axis=0)

    if axis_norm_mode == "r200_volume":
        r200_phys = compute_R_delta(MODEL, m_halos, redshift, delta=200)
        r200_comoving = np.asarray(r200_phys, dtype=np.float64) * (1.0 + redshift)
        geom_mean_raw = np.cbrt(np.prod(np.maximum(raw_axes_comoving, 1e-30), axis=1))
        scale = r200_comoving / np.maximum(geom_mean_raw, 1e-30)
        semi_axes_comoving = raw_axes_comoving * scale[:, None]
    else:
        semi_axes_comoving = raw_axes_comoving

    semi_axes_angular_base = get_angular_size_from_comoving(MODEL, semi_axes_comoving, redshift)
    semi_axes_angular_search = semi_axes_angular_base * config.search_radius

    pix_in_halos, zeta, halo_starts, halo_counts, _, r_search = find_pixels_in_halos(
        config.nside,
        halo_xyz,
        semi_axes_angular_search,
        evec_use,
        n_workers=8,
        geometry=mode,
    )

    y_values = _analytic_los_values(
        nside=config.nside,
        halo_xyz=halo_xyz,
        m_halos=m_halos,
        redshift=redshift,
        semi_axes_angular_base=semi_axes_angular_base,
        eigenvectors=evec_use,
        halo_starts=halo_starts,
        halo_counts=halo_counts,
        pixel_indices=pix_in_halos,
        n_los_samples=n_los_samples,
        los_s_max_r200=los_s_max_r200,
    )

    return {
        "pixel_indices": pix_in_halos,
        "zeta": zeta,
        "halo_starts": halo_starts,
        "halo_counts": halo_counts,
        "y_values": y_values,
        "r_search": r_search,
    }


def _integrated_y_per_halo(mode_data: dict, n_halos: int) -> np.ndarray:
    y_int = np.zeros(n_halos, dtype=np.float64)
    for i in range(n_halos):
        sl = _per_halo_slice(mode_data["halo_starts"], mode_data["halo_counts"], i)
        y_int[i] = float(np.sum(mode_data["y_values"][sl]))
    return y_int


def _log_integrated_y_comparison(
    spherical: dict,
    triaxial: dict,
    m_halos: np.ndarray,
) -> None:
    n_halos = len(m_halos)
    y_sph = _integrated_y_per_halo(spherical, n_halos)
    y_tri = _integrated_y_per_halo(triaxial, n_halos)

    logger.info("Integrated Compton-y comparison for selected halos (analytic LOS)")
    for i in range(n_halos):
        delta = y_tri[i] - y_sph[i]
        frac = 100.0 * delta / max(abs(y_sph[i]), 1e-30)
        logger.info(
            f"  halo[{i}] M={m_halos[i]:.0e} Msun | "
            f"Y_sph={y_sph[i]:.6e}, Y_tri={y_tri[i]:.6e}, "
            f"ΔY={delta:.6e}, ΔY/Y_sph={frac:.3f}%"
        )

    total_sph = float(np.sum(y_sph))
    total_tri = float(np.sum(y_tri))
    total_delta = total_tri - total_sph
    total_frac = 100.0 * total_delta / max(abs(total_sph), 1e-30)
    logger.info(
        f"  TOTAL(4 halos) | Y_sph={total_sph:.6e}, Y_tri={total_tri:.6e}, "
        f"ΔY={total_delta:.6e}, ΔY/Y_sph={total_frac:.3f}%"
    )


def _values_for_pixels(target_pixels: np.ndarray, pixels: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(target_pixels), dtype=np.float64)
    if len(pixels) == 0 or len(target_pixels) == 0:
        return out

    order = np.argsort(pixels)
    pixels_sorted = pixels[order]
    values_sorted = values[order]
    idx = np.searchsorted(pixels_sorted, target_pixels)
    valid = idx < len(pixels_sorted)
    if np.any(valid):
        valid_idx = idx[valid]
        matched = pixels_sorted[valid_idx] == target_pixels[valid]
        if np.any(matched):
            out_idx = np.where(valid)[0][matched]
            out[out_idx] = values_sorted[valid_idx[matched]]
    return out


def _rasterize_local_patch(
    nside: int,
    n0: np.ndarray,
    panel_pixels: np.ndarray,
    panel_values: np.ndarray,
    half_width_arcmin: float,
    npix_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-half_width_arcmin, half_width_arcmin, npix_side)
    xx_arcmin, yy_arcmin = np.meshgrid(axis, axis)

    x_rad = np.radians(xx_arcmin / 60.0)
    y_rad = np.radians(yy_arcmin / 60.0)
    u = np.tan(x_rad)
    v = np.tan(y_rad)

    t1, t2 = tangent_basis(n0)
    vec = n0[None, None, :] + u[..., None] * t1[None, None, :] + v[..., None] * t2[None, None, :]
    vec /= np.linalg.norm(vec, axis=2, keepdims=True)

    theta = np.arccos(np.clip(vec[..., 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(vec[..., 1], vec[..., 0]), 2.0 * np.pi)
    pix_grid = hp.ang2pix(nside, theta, phi, nest=True)

    flat_vals = _values_for_pixels(pix_grid.ravel(), panel_pixels, panel_values)
    return axis, axis, flat_vals.reshape(npix_side, npix_side)


def _apply_optional_smoothing(grid: np.ndarray, smooth_fwhm_arcmin: float, pixel_scale_arcmin: float) -> np.ndarray:
    if smooth_fwhm_arcmin <= 0.0:
        return grid
    sigma_pix = smooth_fwhm_arcmin / (2.355 * pixel_scale_arcmin)
    if sigma_pix <= 0.0:
        return grid
    return gaussian_filter(grid, sigma=sigma_pix, mode="nearest")


def _plot_single_panel(
    ax,
    x_axis_arcmin: np.ndarray,
    y_axis_arcmin: np.ndarray,
    image: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    half_width_arcmin: float,
):
    im = ax.imshow(
        image,
        origin="lower",
        extent=[x_axis_arcmin[0], x_axis_arcmin[-1], y_axis_arcmin[0], y_axis_arcmin[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlabel("x [arcmin]", fontsize=7, labelpad=1)
    ax.set_ylabel("y [arcmin]", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6, pad=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half_width_arcmin, half_width_arcmin)
    ax.set_ylim(-half_width_arcmin, half_width_arcmin)
    return im


def make_figure(
    config: PainterConfig,
    halo_xyz: np.ndarray,
    m_halos: np.ndarray,
    redshift: float,
    spherical: dict,
    triaxial: dict,
    output_path: Path,
    window_scale: float,
    panel_npix: int,
    smooth_fwhm_arcmin: float,
):
    n_rows = len(halo_xyz)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.3 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    for i in range(n_rows):
        half_width_arcmin = np.degrees(float(triaxial["r_search"][i]) * window_scale) * 60.0
        half_width_arcmin = max(half_width_arcmin, 0.5)

        sl_sph = _per_halo_slice(spherical["halo_starts"], spherical["halo_counts"], i)
        sl_tri = _per_halo_slice(triaxial["halo_starts"], triaxial["halo_counts"], i)

        pix_s = spherical["pixel_indices"][sl_sph]
        val_s = spherical["y_values"][sl_sph]
        pix_t = triaxial["pixel_indices"][sl_tri]
        val_t = triaxial["y_values"][sl_tri]

        x_axis_arcmin, y_axis_arcmin, y_s = _rasterize_local_patch(
            config.nside,
            halo_xyz[i],
            pix_s,
            val_s,
            half_width_arcmin,
            panel_npix,
        )
        _, _, y_t = _rasterize_local_patch(
            config.nside,
            halo_xyz[i],
            pix_t,
            val_t,
            half_width_arcmin,
            panel_npix,
        )

        pixel_scale_arcmin = (2.0 * half_width_arcmin) / max(panel_npix - 1, 1)
        y_s = _apply_optional_smoothing(y_s, smooth_fwhm_arcmin, pixel_scale_arcmin)
        y_t = _apply_optional_smoothing(y_t, smooth_fwhm_arcmin, pixel_scale_arcmin)

        denom_floor = max(np.max(np.abs(y_s)) * 1e-3, 1e-30)
        residual_pct = 100.0 * (y_t - y_s) / np.maximum(np.abs(y_s), denom_floor)

        vmax_signal = float(max(np.max(y_s), np.max(y_t), 1e-30))
        vmin_signal = float(max(vmax_signal * 1e-4, 1e-30))
        y_s_plot = np.log10(np.maximum(y_s, vmin_signal))
        y_t_plot = np.log10(np.maximum(y_t, vmin_signal))

        vmax_res = float(np.percentile(np.abs(residual_pct), 99))
        if not np.isfinite(vmax_res) or vmax_res <= 0:
            vmax_res = 1.0

        sc0 = _plot_single_panel(
            axes[i, 0],
            x_axis_arcmin,
            y_axis_arcmin,
            y_s_plot,
            cmap="turbo",
            vmin=np.log10(vmin_signal),
            vmax=np.log10(vmax_signal),
            half_width_arcmin=half_width_arcmin,
        )
        cbar0 = fig.colorbar(sc0, ax=axes[i, 0], fraction=0.046, pad=0.02)
        cbar0.set_label("log10(y)", fontsize=6, labelpad=2)
        cbar0.ax.tick_params(labelsize=6, pad=1)
        axes[i, 0].text(
            0.03,
            0.97,
            rf"$z={redshift:.3f}$" + "\n" + rf"$M={m_halos[i]:.0e}\,M_\odot$",
            transform=axes[i, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.25, "edgecolor": "none", "pad": 1.5},
        )

        sc1 = _plot_single_panel(
            axes[i, 1],
            x_axis_arcmin,
            y_axis_arcmin,
            y_t_plot,
            cmap="turbo",
            vmin=np.log10(vmin_signal),
            vmax=np.log10(vmax_signal),
            half_width_arcmin=half_width_arcmin,
        )
        cbar1 = fig.colorbar(sc1, ax=axes[i, 1], fraction=0.046, pad=0.02)
        cbar1.set_label("log10(y)", fontsize=6, labelpad=2)
        cbar1.ax.tick_params(labelsize=6, pad=1)

        sc2 = _plot_single_panel(
            axes[i, 2],
            x_axis_arcmin,
            y_axis_arcmin,
            residual_pct,
            cmap="coolwarm",
            vmin=-vmax_res,
            vmax=vmax_res,
            half_width_arcmin=half_width_arcmin,
        )
        cbar2 = fig.colorbar(sc2, ax=axes[i, 2], fraction=0.046, pad=0.02)
        cbar2.set_label("% (triaxial - spherical)", fontsize=6, labelpad=2)
        cbar2.ax.tick_params(labelsize=6, pad=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analytic LOS compare: spherical vs triaxial painting for 4 halos (no interpolator grid)."
    )
    parser.add_argument(
        "--halo-dir",
        type=Path,
        default=HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf",
        help="Path to halo catalog ASDF.",
    )
    parser.add_argument(
        "--healcounts-file",
        type=Path,
        default=HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0671-0676.asdf",
        help="Path to total heal-counts ASDF.",
    )
    parser.add_argument("--nside", type=int, default=8192, help="HEALPix nside for loading/degrading counts.")
    parser.add_argument(
        "--mass-targets",
        type=float,
        nargs="+",
        default=[12.0, 13.0, 13.5, 14.0],
        help="Target log10(M) values used to pick nearest halos.",
    )
    parser.add_argument("--search-radius", type=float, default=4.0, help="Search-radius multiplier.")
    parser.add_argument(
        "--window-scale",
        type=float,
        default=4.0,
        help="Half-width of each halo panel in units of that halo's r_search.",
    )
    parser.add_argument(
        "--panel-npix",
        type=int,
        default=1000,
        help="Number of pixels per side in each rendered panel.",
    )
    parser.add_argument(
        "--smooth-fwhm-arcmin",
        type=float,
        default=0.0,
        help="Optional Gaussian smoothing FWHM in arcmin (display only).",
    )
    parser.add_argument(
        "--los-samples",
        type=int,
        default=64,
        help="Number of LOS samples for explicit integral.",
    )
    parser.add_argument(
        "--los-s-max",
        type=float,
        default=4.0,
        help="Half-range for LOS integral in units of R200 (default: 4).",
    )
    parser.add_argument(
        "--axis-norm-mode",
        choices=["r98", "r200_volume"],
        default="r200_volume",
        help="Geometry normalization: keep raw R98 axes or rescale axes to enforce abc=R200^3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path; if omitted, writes under data/visualization/<date>/.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    setup_logging("compare_analytic_spherical_triaxial_four_halos")

    logger.info("Loading Abacus data for analytic comparison plot")
    data = load_abacus_for_painting(
        halo_dir=args.halo_dir,
        healcounts_file_1=args.healcounts_file,
        nside=args.nside,
    )

    valid = np.isfinite(data.radii_halos) & (data.radii_halos > 0)
    valid_indices = np.where(valid)[0]
    targets = np.asarray(args.mass_targets, dtype=np.float64)
    if len(valid_indices) < len(targets):
        raise ValueError("Not enough valid halos to sample from")

    logm_valid = np.log10(data.m_halos[valid_indices])
    chosen_local = choose_halos_by_mass_targets(logm_valid, targets)
    chosen_global = valid_indices[chosen_local]

    theta = data.theta[chosen_global]
    phi = data.phi[chosen_global]
    halo_xyz = convert_rad_to_cart(theta, phi)
    radii_halos = data.radii_halos[chosen_global]
    eigenvalues = data.eigenvalues[chosen_global]
    eigenvectors = data.eigenvectors[chosen_global]
    m_halos = data.m_halos[chosen_global]

    config = PainterConfig(
        nside=args.nside,
        search_radius=args.search_radius,
        weight_bin_width=2e-5,
        halo_geometry="triaxial",
    )

    logger.info("Running spherical analytic LOS pass")
    spherical = build_analytic_mode(
        mode="spherical",
        config=config,
        halo_xyz=halo_xyz,
        radii_halos=radii_halos,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        m_halos=m_halos,
        redshift=data.redshift,
        n_los_samples=max(int(args.los_samples), 8),
        los_s_max_r200=max(float(args.los_s_max), 0.5),
        axis_norm_mode=args.axis_norm_mode,
    )

    logger.info("Running triaxial analytic LOS pass")
    triaxial = build_analytic_mode(
        mode="triaxial",
        config=config,
        halo_xyz=halo_xyz,
        radii_halos=radii_halos,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        m_halos=m_halos,
        redshift=data.redshift,
        n_los_samples=max(int(args.los_samples), 8),
        los_s_max_r200=max(float(args.los_s_max), 0.5),
        axis_norm_mode=args.axis_norm_mode,
    )

    _log_integrated_y_comparison(spherical, triaxial, m_halos)

    if args.output is None:
        date_dir = datetime.now().strftime("%Y-%m-%d")
        output_name = "four_halo_analytic_spherical_vs_triaxial.png"
        output_path = OUTPUT_PATH / "visualization" / date_dir / output_name
    else:
        output_path = args.output

    logger.info(f"Creating plot: {output_path}")
    make_figure(
        config=config,
        halo_xyz=halo_xyz,
        m_halos=m_halos,
        redshift=data.redshift,
        spherical=spherical,
        triaxial=triaxial,
        output_path=output_path,
        window_scale=args.window_scale,
        panel_npix=args.panel_npix,
        smooth_fwhm_arcmin=args.smooth_fwhm_arcmin,
    )

    logger.info("Done")


if __name__ == "__main__":
    main()
