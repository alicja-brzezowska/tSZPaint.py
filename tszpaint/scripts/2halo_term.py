from __future__ import annotations

from pathlib import Path

import asdf
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH, INTERPOLATORS_PATH
from tszpaint.converters import convert_rad_to_cart
from tszpaint.cosmology.model import compute_theta_200, get_angular_size_from_comoving
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.y_profile.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile.y_profile import create_battaglia_profile

MODEL = create_battaglia_profile()

PROFILE_LOGM_CENTERS = [12.7, 13.0, 13.7, 14.0, 14.7, 15.0]
PROFILE_N_HALOS = 1000
PROFILE_SEED = 123
PROFILE_LOGM_HALFWIDTH = 0.2
PROFILE_N_BINS = 20
N = 4
WEIGHT_BIN_WIDTH = 2e-5
JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"

TOTAL_MAP_PATH = Path(
    "/home/ab2927/rds/tSZPaint.py/data/visualization/2026-02-26/4r98-total-normalized-bin.asdf"
)
ONE_HALO_MAP_PATH = Path(
    "/home/ab2927/rds/tSZPaint.py/data/visualization/2026-02-26/4r98-halo-normalized-bin.asdf"
)
MASS_CENTER_FOR_COMPONENT_PLOT = 14.0


def _read_asdf_map_payload(map_path: Path):
    with asdf.open(map_path) as af:
        tree = af.tree
        data_node = tree.get("data", {})
        header_node = tree.get("header", {})

        if "y_map" not in data_node:
            raise KeyError(f"No 'data/y_map' found in {map_path}")

        y_map = np.asarray(data_node["y_map"])
        nest = bool(data_node.get("nest", header_node.get("nest", True)))
        radial_profile = data_node.get("radial_profile", [])

    return y_map, nest, _to_numpy_profiles(radial_profile), header_node


def _to_numpy_profiles(radial_profiles) -> list[dict]:
    profiles = (
        radial_profiles if isinstance(radial_profiles, list) else [radial_profiles]
    )
    cleaned = []
    for profile in profiles:
        clean = {}
        for key, value in profile.items():
            if (
                isinstance(value, (list, tuple))
                or getattr(value, "shape", None) is not None
            ):
                clean[key] = np.asarray(value)
            else:
                clean[key] = value
        cleaned.append(clean)
    return cleaned


def save_radial_profile_points(radial_profiles: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()
    asdf.AsdfFile({"radial_profile_points": radial_profiles}).write_to(output_path)
    logger.info(f"Saved radial-profile points to {output_path}")


def plot_two_halo_term(
    total_profiles: list[dict], one_halo_profiles: list[dict], output_stub: str
):

    mass_idx = PROFILE_LOGM_CENTERS.index(MASS_CENTER_FOR_COMPONENT_PLOT)
    total = total_profiles[mass_idx]
    one_halo = one_halo_profiles[mass_idx]

    x = np.asarray(total["x_centers"])

    y_total = np.asarray(total["y_mean"])
    y_one = np.asarray(one_halo["y_mean"])
    y_two = y_total - y_one

    x_dense = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 400)
    y_total_curve = np.interp(x_dense, x, y_total)
    y_two_curve = np.interp(x_dense, x, y_two)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y_total, s=28, alpha=0.85, label="_nolegend_")
    ax.plot(x_dense, y_total_curve, lw=2.2, label=r"Total signal ($1h+2h$)")
    ax.plot(x_dense, y_two_curve, lw=2.2, label=r"Two halo term (2h)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r/R_{200}$", fontsize=16)
    ax.set_ylabel(r"$y$", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14)
    fig.tight_layout()

    plot_path = Path(f"{output_stub}_two_halo.png")
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    points_path = Path(f"{output_stub}_two_halo.asdf")
    if points_path.exists() or points_path.is_symlink():
        points_path.unlink()
    asdf.AsdfFile(
        {
            "mass_center": MASS_CENTER_FOR_COMPONENT_PLOT,
            "x_r_over_r200": x,
            "x_r_over_r200_curve": x_dense,
            "y_total": y_total,
            "y_total_curve": y_total_curve,
            "y_one_halo": y_one,
            "y_two_halo": y_two,
            "y_two_halo_curve": y_two_curve,
        }
    ).write_to(points_path)

    logger.info(f"Saved two-halo plot to {plot_path}")
    logger.info(f"Saved two-halo points to {points_path}")


def y_vs_R(
    config: PainterConfig,
    y_map: np.ndarray,
    data,
    interpolator: BattagliaLogInterpolator,
    profile_n_halos: int = PROFILE_N_HALOS,
    profile_seed: int = PROFILE_SEED,
    profile_logM_centers: list[float] | None = None,
    profile_logM_halfwidth: float = PROFILE_LOGM_HALFWIDTH,
):
    if profile_logM_centers is None:
        profile_logM_centers = PROFILE_LOGM_CENTERS
    n_bins_eff = PROFILE_N_BINS

    halo_xyz = convert_rad_to_cart(data.theta, data.phi)
    r_90 = (
        get_angular_size_from_comoving(MODEL, data.radii_halos, data.redshift)
        * config.search_radius
    )

    pix_in_halos, distances, halo_starts, halo_counts, halo_indices = (
        find_pixels_in_halos(config.nside, halo_xyz, r_90, n_workers=8)
    )

    y_values = y_map[pix_in_halos]

    def y_vs_r_mechanism(logM_center: float | None):
        rng = np.random.default_rng(profile_seed)
        logM = np.log10(data.m_halos)
        in_bin = (
            np.abs(logM - logM_center) <= profile_logM_halfwidth
            if logM_center is not None
            else np.ones_like(logM, dtype=bool)
        )
        candidate_halos = np.flatnonzero(in_bin)

        n_total = len(candidate_halos)
        if n_total == 0:
            return {
                "x_centers": np.array([]),
                "y_mean": np.array([]),
                "y_err": np.array([]),
                "counts": np.array([]),
                "n_sample": 0,
                "x_ref": np.array([]),
                "y_battaglia": np.array([]),
                "mass_ref": np.nan,
                "logM_center": logM_center,
            }

        n_sample = min(profile_n_halos, n_total)
        sample_halos = rng.choice(candidate_halos, size=n_sample, replace=False)

        theta_200 = compute_theta_200(MODEL, data.m_halos, data.redshift)
        ratio = r_90[sample_halos] / theta_200[sample_halos]
        x_max = np.nanmax(ratio[np.isfinite(ratio)])
        x_min = max(1e-4, x_max / 1e3)
        bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins_eff + 1)
        x_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

        sum_y = np.zeros(n_bins_eff, dtype=np.float64)
        sum_y2 = np.zeros(n_bins_eff, dtype=np.float64)
        counts = np.zeros(n_bins_eff, dtype=np.float64)

        for h in sample_halos:
            start = halo_starts[h]
            count = halo_counts[h]
            if count == 0:
                continue
            d = distances[start : start + count]
            y = y_values[start : start + count]
            keep = d <= r_90[h]
            if not np.any(keep):
                continue
            d = d[keep]
            y = y[keep]
            x = d / theta_200[h]
            bin_ids = np.searchsorted(bin_edges[1:], x, side="left")
            bin_ids = np.minimum(bin_ids, n_bins_eff - 1)

            sum_y += np.bincount(bin_ids, weights=y, minlength=n_bins_eff)
            sum_y2 += np.bincount(bin_ids, weights=y**2, minlength=n_bins_eff)
            counts += np.bincount(bin_ids, minlength=n_bins_eff)

        with np.errstate(invalid="ignore", divide="ignore"):
            y_mean = sum_y / counts
            y_var = sum_y2 / counts - y_mean**2
            y_std = np.sqrt(np.maximum(y_var, 0.0))
            y_err = y_std / np.sqrt(counts)

        mass_ref = np.median(data.m_halos[sample_halos])
        x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 400)
        theta_ref = compute_theta_200(MODEL, np.array([mass_ref]), data.redshift)[0]
        theta_values = np.maximum(x_ref * theta_ref, 1e-40)
        log_theta = np.log(theta_values)
        log_M = np.full_like(log_theta, np.log10(mass_ref), dtype=np.float64)
        z_values = np.full_like(log_theta, data.redshift, dtype=np.float32)
        y_battaglia = np.asarray(interpolator.eval_for_logs(log_theta, z_values, log_M))

        return {
            "x_centers": x_centers,
            "y_mean": y_mean,
            "y_err": y_err,
            "counts": counts,
            "n_sample": n_sample,
            "x_ref": x_ref,
            "y_battaglia": y_battaglia,
            "mass_ref": mass_ref,
            "logM_center": logM_center,
        }

    return [y_vs_r_mechanism(center) for center in profile_logM_centers]


def main():
    y_total_map, nest_total, _, _ = _read_asdf_map_payload(TOTAL_MAP_PATH)
    if not nest_total:
        y_total_map = hp.reorder(y_total_map, r2n=True)

    y_one_halo_map, nest_one, _, _ = _read_asdf_map_payload(ONE_HALO_MAP_PATH)
    if not nest_one:
        y_one_halo_map = hp.reorder(y_one_halo_map, r2n=True)

    config = PainterConfig(
        nside=hp.get_nside(y_total_map),
        search_radius=N,
        weight_bin_width=WEIGHT_BIN_WIDTH,
    )
    halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
    healcounts_file1 = (
        HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"
    )

    logger.info("Loading halo catalog for profile extraction...")
    data = load_abacus_for_painting(
        halo_dir=halo_dir,
        healcounts_file_1=healcounts_file1,
        nside=config.nside,
    )

    interpolator = BattagliaLogInterpolator.from_pickle(JAX_PATH)
    total_profiles = y_vs_R(config, y_total_map, data, interpolator)
    one_halo_profiles = y_vs_R(config, y_one_halo_map, data, interpolator)

    total_stub = f"{TOTAL_MAP_PATH.with_suffix('')}_after_painting"
    one_stub = f"{ONE_HALO_MAP_PATH.with_suffix('')}_before_painting"

    save_radial_profile_points(
        total_profiles, Path(f"{total_stub}_y_vs_r200_points.asdf")
    )
    save_radial_profile_points(
        one_halo_profiles, Path(f"{one_stub}_y_vs_r200_points.asdf")
    )

    plot_two_halo_term(total_profiles, one_halo_profiles, total_stub)


if __name__ == "__main__":
    main()
