from loguru import logger

from tszpaint.config import HALO_CATALOGS_PATH, INTERPOLATORS_PATH, HEALCOUNTS_PATH, OUTPUT_PATH
from tszpaint.logging import setup_logging
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.mock_data_generator import MockDataGenerator
from tszpaint.paint.paint import paint_abacus, paint_and_visualize


NSIDE = 8192
N = 0.1  # Base multiple of r_90 to search
NPRIME_VALUES = [2.8, 3, 4]
WEIGHT_BIN_WIDTH = 2e-5
USE_WEIGHTS = True

MOCK = False


def main():
    setup_logging("find_optimal_r_search")
    halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
    healcounts_file1 = (
        HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"
    )
    JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"

    logger.info("Painting Abacus tSZ map...")
    logger.info(f"Halo directory: {halo_dir}")
    logger.info(f"Healcounts file 1: {healcounts_file1}")

    for nprime in NPRIME_VALUES:
        search_radius = N * nprime
        config = PainterConfig(
            nside=NSIDE,
            search_radius=search_radius,
            weight_bin_width=WEIGHT_BIN_WIDTH,
        )
        output_file = OUTPUT_PATH / f"y_map_abacus_Nprimehalos{nprime:g}.fits"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using search_radius = N * N' = {N} * {nprime} = {search_radius}")
        logger.info(f"Output file: {output_file}")

        paint_abacus(
            config,
            halo_dir=halo_dir,
            healcounts_file_1=healcounts_file1,
            output_file=output_file,
            interpolator_path=JAX_PATH,
            use_weights=USE_WEIGHTS,
        )


if __name__ == "__main__":
    main()
