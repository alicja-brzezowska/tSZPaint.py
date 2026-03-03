from datetime import datetime

from loguru import logger

from tszpaint.config import (
    DATA_PATH,
    HALO_CATALOGS_PATH,
    HEALCOUNTS_PATH,
    HEALCOUNTS_TOTAL_PATH,
    INTERPOLATORS_PATH,
)
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.mock_data_generator import MockDataGenerator
from tszpaint.paint.paint import paint_abacus, paint_and_visualize

NSIDE = 8192
N = 4  # Multiple of r_90/ r_95 / r_98 to search
WEIGHT_BIN_WIDTH = 2e-5
USE_WEIGHTS = False

MOCK = False


def main():
    if MOCK:
        nside = 1024
        config = PainterConfig(
            nside=nside, search_radius=N, weight_bin_width=WEIGHT_BIN_WIDTH
        )
        gen = MockDataGenerator(100, nside=nside)
        data = gen.generate_simulation_data()
        paint_and_visualize(config, data, None)

    else:
        config = PainterConfig(
            nside=NSIDE, search_radius=N, weight_bin_width=WEIGHT_BIN_WIDTH
        )
        halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
        healcounts_file1 = (
            HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0671-0676.asdf"
        )
        healcounts_file2 = (
            HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0677-0682.asdf"
        )
        healcounts_file3 = (
            HEALCOUNTS_PATH / "LightCone0_total_heal-counts_Step0665-0670.asdf"
        )
        JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
        date_dir = datetime.now().strftime("%Y-%m-%d")
        output_dir = DATA_PATH / "visualization" / date_dir
        output_file = output_dir / "triaxial_small_mass.asdf"

        logger.info("Painting Abacus tSZ map...")
        logger.info(f"Halo directory: {halo_dir}")
        logger.info(f"Healcounts file 1: {healcounts_file1}")
        # logger.info(f"Healcounts file 2: {healcounts_file2}")
        # logger.info(f"Healcounts file 3: {healcounts_file3}")
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
