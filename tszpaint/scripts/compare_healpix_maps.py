from tszpaint.config import HEALCOUNTS_PATH, HEALCOUNTS_TOTAL_PATH
from tszpaint.paint.visualize import Visualizer, PlotConfig
from tszpaint.paint.abacus_loader import load_abacus_healcounts

TOTAL_PATH = HEALCOUNTS_TOTAL_PATH / "LightCone0_total_heal-counts_Step0677-0682.asdf"
HALO_PATH = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0677-0682.asdf"



def main():
    config = PlotConfig.healpix()
    visualizer = Visualizer(nside=8192, output_file_stub="healpix_comparison")

    visualizer.plot_ra_dec(
        y_map=load_abacus_healcounts(HALO_PATH),
        config=config,
        filename_suffix="total",
    )

    #visualizer.plot_ra_dec(
    #    y_map=load_abacus_healcounts(HALO_PATH),
    #    config=config,
    #    filename_suffix="halos",
    #)


if __name__ == "__main__":
    main()