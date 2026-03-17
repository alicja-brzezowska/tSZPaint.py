from datetime import date

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

NSIDE = 1024
N = 4
WEIGHT_BIN_WIDTH = 2e-5
LOGM_MIN = 12


def main():
    setup_logging("main")

    config = PainterConfig(nside=NSIDE, search_radius=N, weight_bin_width=WEIGHT_BIN_WIDTH)
    halo_catalog_index = load_halo_catalog_index(OUTPUT_PATH / "halo_catalog_index.json")
    healcounts_file = (
        HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0671-0676.asdf"
    )

    data = load_abacus_for_painting(
        halo_catalog_index=halo_catalog_index,
        healcounts_file_1=healcounts_file,
        nside=config.nside,
        logm_min=LOGM_MIN,
    )

    ctx = build_paint_context(config, data, use_weights=True)

    interpolator = BattagliaLogInterpolator.from_pickle(JAX_PATH)

    # Per-pair y values before map accumulation — used for isolated radial profile
    y_values = np.asarray(
        interpolator.eval_for_logs(ctx.log_distances, ctx.z_values, ctx.log_M_values)
    )
    y_values = y_values * ctx.weights

    profile_cfg = RadialProfileBuilderConfig(
        r_search=ctx.r_search,
        num_halos=1000,
        nside=NSIDE,
    )
    builder = RadialProfileBuilder(
        cfg=profile_cfg,
        data=data,
        pix_in_halos=ctx.pix_in_halos,
        halo_starts=ctx.halo_starts,
        halo_counts=ctx.halo_counts,
        distances=ctx.zeta,
        interpolator=interpolator,
        model=create_battaglia_profile(),
        y_values=y_values,
    )
    profiles = builder.build_isolated()
    del interpolator

    out_dir = OUTPUT_PATH / "visualization" / str(date.today())
    out_dir.mkdir(parents=True, exist_ok=True)
    viz = Visualizer(nside=NSIDE, output_file_stub=str(out_dir / "nside1024"))
    viz.plot_Y_vs_R200(profiles, suffix="radial_profile_nside1024")
    logger.info(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
