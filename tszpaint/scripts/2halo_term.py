from __future__ import annotations

from pathlib import Path

import asdf
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH, INTERPOLATORS_PATH
from tszpaint.cosmology.model import get_angular_size_from_comoving
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.pixel_search import find_pixels_in_halos
from tszpaint.scripts.radial_profile import (
    RadialProfile,
    RadialProfileBuilder,
    RadialProfileBuilderConfig,
)
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
        radial_profiles = data_node.get("radial_profile", [])

    return (
        y_map,
        nest,
        [RadialProfile.from_dict(d) for d in radial_profiles],
        header_node,
    )


def save_radial_profile_points(radial_profiles: list[RadialProfile], output_path: Path):
    asdf.AsdfFile(
        {"radial_profile_points": [p.as_dict for p in radial_profiles]}
    ).write_to(output_path)
    logger.info(f"Saved radial-profile points to {output_path}")


def plot_two_halo_term(
    total_profiles: list[RadialProfile],
    one_halo_profiles: list[RadialProfile],
    output_stub: str,
):
    mass_idx = PROFILE_LOGM_CENTERS.index(MASS_CENTER_FOR_COMPONENT_PLOT)
    total = total_profiles[mass_idx]
    one_halo = one_halo_profiles[mass_idx]

    x = np.asarray(total.x_centers)

    y_total = np.asarray(total.y_mean)
    y_one = np.asarray(one_halo.y_mean)
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
    r90 = (
        get_angular_size_from_comoving(MODEL, data.radii_halos, data.redshift)
        * config.search_radius
    )
    radial_cfg = RadialProfileBuilderConfig(r_search=r90)
    _, distances, halo_starts, halo_counts, _ = find_pixels_in_halos(
        config.nside, data.halo_xyz, r90, n_workers=8
    )

    total_profiles = RadialProfileBuilder(
        radial_cfg,
        data,
        halo_starts,
        halo_counts,
        distances,
        interpolator,
        MODEL,
    ).build(y_total_map)

    one_halo_profiles = RadialProfileBuilder(
        radial_cfg,
        data,
        halo_starts,
        halo_counts,
        distances,
        interpolator,
        MODEL,
    ).build(y_one_halo_map)

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
