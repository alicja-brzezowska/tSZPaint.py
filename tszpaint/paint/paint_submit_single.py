from datetime import date

import healpy as hp
import numpy as np
from loguru import logger

from tszpaint.config import HEALCOUNTS_TOTAL_PATH, OUTPUT_PATH
from tszpaint.logging import setup_logging
from tszpaint.paint.abacus_loader import load_abacus_for_painting
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.paint import JAX_PATH, build_paint_context
from tszpaint.paint.visualize import Visualizer
from tszpaint.scripts.match_redshifts import load_halo_catalog_index
from tszpaint.scripts.radial_profile import RadialProfileBuilder, RadialProfileBuilderConfig
from tszpaint.y_profile.interpolator import BattagliaLogInterpolator
from tszpaint.y_profile.y_profile import create_battaglia_profile

import asdf

NSIDE = 8192
N = 4
WEIGHT_BIN_WIDTH = 2e-5
HEALCOUNTS_FILE = HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0671-0676.asdf"

RUNS = [
    {"logm_min": 11.5, "name": "logM11.5"},
    {"logm_min": 12.0, "name": "logM12.0"},
]


def run_one(config, halo_catalog_index, logm_min, run_name, out_dir):
    logger.info(f"=== Run: {run_name} (logm_min={logm_min}) ===")

    data = load_abacus_for_painting(
        halo_catalog_index=halo_catalog_index,
        healcounts_file_1=HEALCOUNTS_FILE,
        nside=config.nside,
        logm_min=logm_min,
    )
    logger.info(f"Loaded {len(data.m_halos)} halos above logM={logm_min}")

    ctx = build_paint_context(config, data, use_weights=True)

    interpolator = BattagliaLogInterpolator.from_pickle(JAX_PATH)

    y_values = np.asarray(
        interpolator.eval_for_logs(ctx.log_distances, ctx.z_values, ctx.log_M_values)
    )
    del interpolator
    y_values = y_values * ctx.weights

    ymap = np.zeros(hp.nside2npix(config.nside), dtype=np.float32)
    np.add.at(ymap, ctx.pix_in_halos, y_values)

    out = out_dir / f"{run_name}.asdf"
    af = asdf.AsdfFile(
        {
            "header": {
                "nside": config.nside,
                "nest": True,
                "redshift": data.redshift,
                "search_radius_multiplier": config.search_radius,
                "bin_width": config.weight_bin_width,
                "halo_geometry": config.halo_geometry,
                "healpix_file": str(HEALCOUNTS_FILE),
                "interpolator": str(JAX_PATH),
                "logm_min": logm_min,
            },
            "data": {
                "y_map": ymap,
            },
        }
    )
    af.set_array_compression(af["data"]["y_map"], "blsc")
    af.write_to(out)
    logger.info(f"Saved map: {out}")

    # Radial profile (only mass bins above the cut)
    log_m_centers = [c for c in [12.0, 12.5, 13.0, 13.5, 14.0, 14.5] if c >= logm_min]
    profile_cfg = RadialProfileBuilderConfig(
        r_search=ctx.r_search,
        num_halos=1000,
        nside=NSIDE,
        log_m_centers=log_m_centers,
    )
    builder = RadialProfileBuilder(
        cfg=profile_cfg,
        data=data,
        pix_in_halos=ctx.pix_in_halos,
        halo_starts=ctx.halo_starts,
        halo_counts=ctx.halo_counts,
        distances=ctx.zeta,
        interpolator=BattagliaLogInterpolator.from_pickle(JAX_PATH),
        model=create_battaglia_profile(),
        y_values=y_values,
    )
    profiles = builder.build_isolated()

    viz = Visualizer(nside=NSIDE, output_file_stub=str(out_dir / run_name))
    viz.plot_Y_vs_R200(profiles, suffix="radial_profile")
    logger.info(f"Saved radial profile: {out_dir / run_name}_radial_profile.png")


def main():
    setup_logging("main")

    config = PainterConfig(nside=NSIDE, search_radius=N, weight_bin_width=WEIGHT_BIN_WIDTH)
    halo_catalog_index = load_halo_catalog_index(OUTPUT_PATH / "halo_catalog_index.json")

    out_dir = OUTPUT_PATH / "visualization" / str(date.today())
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    for run in RUNS:
        run_one(config, halo_catalog_index, run["logm_min"], run["name"], out_dir)


if __name__ == "__main__":
    main()
