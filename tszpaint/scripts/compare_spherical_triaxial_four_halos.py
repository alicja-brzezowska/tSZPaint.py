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
    INTERPOLATORS_PATH,
    OUTPUT_PATH,
)
from tszpaint.logging import setup_logging
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import get_angular_size_from_comoving
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.paint import MODEL, get_real_space_from_eigenvals, load_interpolator
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.paint.weights import compute_weights



def find_pixels_in_halos_effective_radius(
    nside: int,
    halo_xyz: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    nest: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find halo pixels using the effective projected radius (no particle info).
    """
    n_halos = len(halo_xyz)
    results = [None] * n_halos
    halo_effective_radius = np.zeros(n_halos, dtype=np.float64)

    for i in range(n_halos):
        evals = eigenvalues[i]
        evecs = eigenvectors[i]
        n0 = halo_xyz[i]

        # normalization
        e_a = evecs[0] / np.linalg.norm(evecs[0])
        e_b = evecs[1] / np.linalg.norm(evecs[1])
        e_c = evecs[2] / np.linalg.norm(evecs[2])

        # matrix Q of the ellipsoid in 3D space: x^T Q x = 1
        inv_a2 = 1.0 / (evals[0] * evals[0])
        inv_b2 = 1.0 / (evals[1] * evals[1])
        inv_c2 = 1.0 / (evals[2] * evals[2])
        Q = (
            np.outer(e_a, e_a) * inv_a2
            + np.outer(e_b, e_b) * inv_b2
            + np.outer(e_c, e_c) * inv_c2
        )

        # project the ellipsoid onto the plane perpendicular to n0
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(n0, ref)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = np.cross(ref, n0)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n0, t1)
        t2 /= np.linalg.norm(t2)

        S11 = t1 @ Q @ t1
        S12 = t1 @ Q @ t2
        S22 = t2 @ Q @ t2

        S = np.array([[S11, S12], [S12, S22]], dtype=np.float64)
        s_eigs = np.linalg.eigvalsh(S)
        a_proj = 1.0 / np.sqrt(max(s_eigs[0], np.finfo(np.float64).tiny))
        c_proj = 1.0 / np.sqrt(max(s_eigs[1], np.finfo(np.float64).tiny))
        r_eff_2d = np.sqrt(a_proj * c_proj)  # effective radius

        pixels = hp.query_disc(
            nside=nside,
            vec=n0,
            radius=r_eff_2d,
            nest=nest,
            inclusive=True,
        )

        if len(pixels) == 0:
            results[i] = (np.array([], dtype=np.int64), np.array([]))
            halo_effective_radius[i] = r_eff_2d
            continue

        x, y, z = hp.pix2vec(nside, pixels, nest=nest)
        pixel_xyz = np.stack([x, y, z], axis=1)
        cosang = np.clip(pixel_xyz @ n0, -1.0, 1.0)
        distances = np.arccos(cosang)

        results[i] = (pixels, distances)
        halo_effective_radius[i] = r_eff_2d

    # Flatten results
    all_pixels = []
    all_distances = []
    halo_counts = np.zeros(n_halos, dtype=np.int64)

    for i, (pixels, distances) in enumerate(results):
        if len(pixels) > 0:
            all_pixels.append(pixels)
            all_distances.append(distances)
            halo_counts[i] = len(pixels)

    pixel_indices = (
        np.concatenate(all_pixels) if all_pixels else np.array([], dtype=np.int64)
    )
    distances = np.concatenate(all_distances) if all_distances else np.array([])

    halo_starts = np.zeros(n_halos, dtype=np.int64)
    halo_starts[1:] = np.cumsum(halo_counts[:-1])

    halo_indices = np.repeat(np.arange(n_halos, dtype=np.int64), halo_counts)

    return (
        pixel_indices,
        distances,
        halo_starts,
        halo_counts,
        halo_indices,
        halo_effective_radius,
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


def build_mode_triaxial_eff_radius(
    config: PainterConfig,
    halo_xyz: np.ndarray,
    radii_halos: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    m_halos: np.ndarray,
    redshift: float,
    interpolator,
) -> dict:
    """Triaxial pixel footprint via effective projected radius; distances are raw angular separations; no weights."""
    semi_axes_comoving = get_real_space_from_eigenvals(eigenvalues, radii_halos)
    semi_axes_angular = get_angular_size_from_comoving(MODEL, semi_axes_comoving, redshift)
    semi_axes_angular *= config.search_radius

    pixel_indices, distances, halo_starts, halo_counts, halo_indices, r_search = (
        find_pixels_in_halos_effective_radius(
            config.nside,
            halo_xyz,
            semi_axes_angular,
            eigenvectors,
            nest=True,
        )
    )

    log_distances = np.log(distances + 1e-40)
    log_m_values = np.log10(m_halos)[halo_indices]
    z_values = np.full_like(distances, redshift, dtype=np.float32)
    y_values = interpolator.eval_for_logs(log_distances, z_values, log_m_values)

    return {
        "pixel_indices": pixel_indices,
        "zeta": distances,
        "halo_starts": halo_starts,
        "halo_counts": halo_counts,
        "y_values": y_values,
        "r_search": r_search,
    }


def build_mode(
    mode: str,
    config: PainterConfig,
    halo_xyz: np.ndarray,
    radii_halos: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    m_halos: np.ndarray,
    redshift: float,
    interpolator,
):
    if mode == "triaxial":
        semi_axes_comoving = get_real_space_from_eigenvals(eigenvalues, radii_halos)
    else:
        spherical_r_comoving = np.asarray(radii_halos, dtype=np.float64)
        semi_axes_comoving = np.repeat(spherical_r_comoving[:, None], 3, axis=1)

    semi_axes_angular = get_angular_size_from_comoving(MODEL, semi_axes_comoving, redshift)
    semi_axes_angular *= config.search_radius

    pix_in_halos, zeta, halo_starts, halo_counts, halo_indices, r_search = find_pixels_in_halos(
        config.nside,
        halo_xyz,
        semi_axes_angular,
        eigenvectors,
        n_workers=8,
        geometry=mode,
    )

    log_distances = np.log(zeta + 1e-40)
    log_m_values = np.log10(m_halos)[halo_indices]
    z_values = np.full_like(zeta, redshift, dtype=np.float32)
    y_values = interpolator.eval_for_logs(log_distances, z_values, log_m_values)

    return {
        "pixel_indices": pix_in_halos,
        "zeta": zeta,
        "halo_starts": halo_starts,
        "halo_counts": halo_counts,
        "y_values": y_values,
        "r_search": r_search,
    }


def _per_halo_slice(starts: np.ndarray, counts: np.ndarray, i: int) -> slice:
    start = int(starts[i])
    count = int(counts[i])
    return slice(start, start + count)


def _integrated_y_per_halo(mode_data: dict, n_halos: int) -> np.ndarray:
    y_int = np.zeros(n_halos, dtype=np.float64)
    for i in range(n_halos):
        sl = _per_halo_slice(mode_data["halo_starts"], mode_data["halo_counts"], i)
        y_int[i] = float(np.sum(mode_data["y_values"][sl]))
    return y_int


def _apply_weights_in_place(
    mode_data: dict,
    *,
    config: PainterConfig,
    particle_counts: np.ndarray,
) -> None:
    r_90_equiv = mode_data["r_search"] / config.search_radius
    weights = compute_weights(
        config,
        pixel_indices=mode_data["pixel_indices"],
        distances=mode_data["zeta"],
        halo_starts=mode_data["halo_starts"],
        halo_counts=mode_data["halo_counts"],
        r_90=r_90_equiv,
        particle_counts=particle_counts,
        method="vectorized",
    )
    mode_data["y_values"] = mode_data["y_values"] * weights


def _log_integrated_y_comparison(
    mode_a: dict,
    mode_b: dict,
    m_halos: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    n_halos = len(m_halos)
    y_a = _integrated_y_per_halo(mode_a, n_halos)
    y_b = _integrated_y_per_halo(mode_b, n_halos)

    logger.info(f"Integrated Compton-y comparison: {label_a} vs {label_b}")
    for i in range(n_halos):
        delta = y_b[i] - y_a[i]
        frac = 100.0 * delta / max(abs(y_a[i]), 1e-30)
        logger.info(
            f"  halo[{i}] M={m_halos[i]:.0e} Msun | "
            f"Y_{label_a}={y_a[i]:.6e}, Y_{label_b}={y_b[i]:.6e}, "
            f"ΔY={delta:.6e}, ΔY/Y_{label_a}={frac:.3f}%"
        )

    total_a = float(np.sum(y_a))
    total_b = float(np.sum(y_b))
    total_delta = total_b - total_a
    total_frac = 100.0 * total_delta / max(abs(total_a), 1e-30)
    logger.info(
        f"  TOTAL(4 halos) | Y_{label_a}={total_a:.6e}, Y_{label_b}={total_b:.6e}, "
        f"ΔY={total_delta:.6e}, ΔY/Y_{label_a}={total_frac:.3f}%"
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


def _rasterize_local_patch_from_fullmap(
    nside: int,
    n0: np.ndarray,
    full_map: np.ndarray,
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

    return axis, axis, full_map[pix_grid]


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
    particle_counts: np.ndarray,
    mode_a: dict,
    mode_b: dict,
    output_path: Path,
    window_scale: float,
    panel_npix: int,
    smooth_fwhm_arcmin: float,
    title_a: str = "Mode A",
    title_b: str = "Mode B",
):
    n_rows = len(halo_xyz)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 3.3 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    for i in range(n_rows):
        half_width_arcmin = np.degrees(max(float(mode_a["r_search"][i]), float(mode_b["r_search"][i])) * window_scale) * 60.0
        half_width_arcmin = max(half_width_arcmin, 0.5)

        sl_a = _per_halo_slice(mode_a["halo_starts"], mode_a["halo_counts"], i)
        sl_b = _per_halo_slice(mode_b["halo_starts"], mode_b["halo_counts"], i)

        pix_s = mode_a["pixel_indices"][sl_a]
        val_s = mode_a["y_values"][sl_a]
        pix_t = mode_b["pixel_indices"][sl_b]
        val_t = mode_b["y_values"][sl_b]

        x_axis_arcmin, y_axis_arcmin, dm_map = _rasterize_local_patch_from_fullmap(
            config.nside,
            halo_xyz[i],
            particle_counts,
            half_width_arcmin,
            panel_npix,
        )
        _, _, y_s = _rasterize_local_patch(
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
        dm_map = _apply_optional_smoothing(dm_map.astype(np.float64), smooth_fwhm_arcmin, pixel_scale_arcmin)
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

        vmax_dm = float(max(np.max(dm_map), 1e-30))
        vmin_dm = float(max(vmax_dm * 1e-4, 1e-30))
        dm_plot = np.log10(np.maximum(dm_map, vmin_dm))

        sc_dm = _plot_single_panel(
            axes[i, 0],
            x_axis_arcmin,
            y_axis_arcmin,
            dm_plot,
            cmap="viridis",
            vmin=np.log10(vmin_dm),
            vmax=np.log10(vmax_dm),
            half_width_arcmin=half_width_arcmin,
        )
        cbar_dm = fig.colorbar(sc_dm, ax=axes[i, 0], fraction=0.046, pad=0.02)
        cbar_dm.set_label("log10(DM counts)", fontsize=6, labelpad=2)
        cbar_dm.ax.tick_params(labelsize=6, pad=1)
        axes[i, 0].plot(
            0.0,
            0.0,
            marker="x",
            markersize=6,
            markeredgewidth=1.2,
            color="black",
            alpha=0.9,
        )

        sc0 = _plot_single_panel(
            axes[i, 1],
            x_axis_arcmin,
            y_axis_arcmin,
            y_s_plot,
            cmap="turbo",
            vmin=np.log10(vmin_signal),
            vmax=np.log10(vmax_signal),
            half_width_arcmin=half_width_arcmin,
        )
        cbar0 = fig.colorbar(sc0, ax=axes[i, 1], fraction=0.046, pad=0.02)
        cbar0.set_label("log10(y)", fontsize=6, labelpad=2)
        cbar0.ax.tick_params(labelsize=6, pad=1)
        axes[i, 1].text(
            0.03,
            0.97,
            rf"$z={redshift:.3f}$" + "\n" + rf"$M={m_halos[i]:.0e}\,M_\odot$",
            transform=axes[i, 1].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.25, "edgecolor": "none", "pad": 1.5},
        )

        sc1 = _plot_single_panel(
            axes[i, 2],
            x_axis_arcmin,
            y_axis_arcmin,
            y_t_plot,
            cmap="turbo",
            vmin=np.log10(vmin_signal),
            vmax=np.log10(vmax_signal),
            half_width_arcmin=half_width_arcmin,
        )
        cbar1 = fig.colorbar(sc1, ax=axes[i, 2], fraction=0.046, pad=0.02)
        cbar1.set_label("log10(y)", fontsize=6, labelpad=2)
        cbar1.ax.tick_params(labelsize=6, pad=1)

        sc2 = _plot_single_panel(
            axes[i, 3],
            x_axis_arcmin,
            y_axis_arcmin,
            residual_pct,
            cmap="coolwarm",
            vmin=-vmax_res,
            vmax=vmax_res,
            half_width_arcmin=half_width_arcmin,
        )
        cbar2 = fig.colorbar(sc2, ax=axes[i, 3], fraction=0.046, pad=0.02)
        cbar2.set_label("% difference", fontsize=6, labelpad=2)
        cbar2.ax.tick_params(labelsize=6, pad=1)

        if i == 0:
            axes[i, 0].set_title("Simulation output (DM only)", fontsize=9)
            axes[i, 1].set_title(title_a, fontsize=9)
            axes[i, 2].set_title(title_b, fontsize=9)
            axes[i, 3].set_title("% difference", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare spherical vs triaxial painting for 4 random halos and plot 3-column panels."
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
        "--interpolator",
        type=Path,
        default=INTERPOLATORS_PATH / "y_values_jax_2.pkl",
        help="Path to Battaglia log-interpolator pickle.",
    )
    parser.add_argument(
        "--triaxial-los-check",
        action="store_true",
        help="Replace triaxial y-values with explicit 3D LOS integral for the selected 4 halos.",
    )
    parser.add_argument(
        "--use-weights",
        action="store_true",
        help="Apply paint-style radial weighting to both spherical and triaxial mode outputs.",
    )
    parser.add_argument(
        "--los-samples",
        type=int,
        default=64,
        help="Number of LOS samples for explicit 3D integral (used with --triaxial-los-check).",
    )
    parser.add_argument(
        "--los-s-max",
        type=float,
        default=8.0,
        help="Half-range in dimensionless LOS coordinate s for explicit 3D integral.",
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

    setup_logging("compare_spherical_triaxial_four_halos")
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logger.info("Loading Abacus data for comparison plot")
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

    interpolator = load_interpolator(args.interpolator)

    logger.info("Running mode A: spherical painting (no weights)")
    mode_a = build_mode(
        mode="spherical",
        config=config,
        halo_xyz=halo_xyz,
        radii_halos=radii_halos,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        m_halos=m_halos,
        redshift=data.redshift,
        interpolator=interpolator,
    )

    logger.info("Running mode B: elliptical painting (with weights)")
    mode_b = build_mode(
        mode="triaxial",
        config=config,
        halo_xyz=halo_xyz,
        radii_halos=radii_halos,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        m_halos=m_halos,
        redshift=data.redshift,
        interpolator=interpolator,
    )
    _apply_weights_in_place(mode_b, config=config, particle_counts=data.particle_counts)

    _log_integrated_y_comparison(
        mode_a, mode_b, m_halos,
        label_a="sph_no_weights", label_b="elliptical_weighted",
    )

    if args.output is None:
        date_dir = datetime.now().strftime("%Y-%m-%d")
        output_name = "weighted_elliptical_vs_unweighted_spherical.png"
        output_path = OUTPUT_PATH / "visualization" / date_dir / output_name
    else:
        output_path = args.output

    logger.info(f"Creating plot: {output_path}")
    make_figure(
        config=config,
        halo_xyz=halo_xyz,
        m_halos=m_halos,
        redshift=data.redshift,
        particle_counts=data.particle_counts,
        mode_a=mode_a,
        mode_b=mode_b,
        output_path=output_path,
        window_scale=args.window_scale,
        panel_npix=args.panel_npix,
        smooth_fwhm_arcmin=args.smooth_fwhm_arcmin,
        title_a="Spherical, no weights",
        title_b="Elliptical, weights",
    )

    logger.info("Done")


if __name__ == "__main__":
    main()
