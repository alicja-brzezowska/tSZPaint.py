from loguru import logger

from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH, INTERPOLATORS_PATH
from tszpaint.paint.config import PainterConfig
from tszpaint.paint.mock_data_generator import MockDataGenerator
from tszpaint.paint.paint import paint_abacus, paint_and_visualize

NSIDE = 8192
N = 2  # Multiple of theta_200 to search
N_BINS = 20  # NOTE: THINK how many bins!
USE_WEIGHTS = True

MOCK = False


def main():
    if MOCK:
        nside = 1024
        config = PainterConfig(nside, N, N_BINS)
        gen = MockDataGenerator(100, nside=nside)
        data = gen.generate_simulation_data()
        paint_and_visualize(config, data, None)

    else:
        config = PainterConfig(NSIDE, N, N_BINS)
        halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
        healcounts_file1 = (
            HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"
        )
        healcounts_file2 = (
            HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0677-0682.asdf"
        )
        healcounts_file3 = (
            HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0665-0670.asdf"
        )
        JAX_PATH = INTERPOLATORS_PATH / "y_values_jax_2.pkl"
        output_file = "y_map_abacus.fits"

        logger.info("Painting Abacus tSZ map...")
        logger.info(f"Halo directory: {halo_dir}")
        logger.info(f"Healcounts file 1: {healcounts_file1}")
        #logger.info(f"Healcounts file 2: {healcounts_file2}")
        #logger.info(f"Healcounts file 3: {healcounts_file3}")
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
