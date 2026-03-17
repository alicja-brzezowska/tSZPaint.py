from loguru import logger

from tszpaint.config import (
    OUTPUT_PATH,
    HEALCOUNTS_TOTAL_PATH,
)
from tszpaint.logging import setup_logging
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.paint import paint_abacus
from tszpaint.scripts.match_redshifts import load_halo_catalog_index

NSIDE = 8192
N = 4  # Multiple of r_90 / r_95 / r_98 to search
WEIGHT_BIN_WIDTH = 2e-5
USE_WEIGHTS = True
LOGM_MIN = 11.5 # cutoff mass 


def main():
    setup_logging("main")
    config = PainterConfig(
        nside=NSIDE, search_radius=N, weight_bin_width=WEIGHT_BIN_WIDTH
    )
    halo_catalog_index = load_halo_catalog_index(OUTPUT_PATH / "halo_catalog_index.json")
    healcounts_file1 = (
        HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0617-0622.asdf"
    )
    output_file = OUTPUT_PATH
    logger.info("Painting Abacus tSZ map...")
    logger.info(f"Halo catalog index: {len(halo_catalog_index)} entries")
    logger.info(f"Healcounts file 1: {healcounts_file1}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"logM min: {LOGM_MIN}")

    paint_abacus(
        config,
        halo_catalog_index=halo_catalog_index,
        healcounts_file_1=healcounts_file1,
        output_file=output_file,
        use_weights=USE_WEIGHTS,
        logm_min=LOGM_MIN,
    )


if __name__ == "__main__":
    main()
