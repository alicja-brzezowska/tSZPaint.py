from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH
import asdf


def obtain_healcount_edges(filepath):
    """Obtain the inner and outer edges (in comoving distance in Mpc/h) of the lightcone from a heal-counts file."""
    with asdf.open(filepath) as f:
        headers = f.tree["headers"]
        chis = [h["CoordinateDistanceHMpc"] for h in headers]
    return min(chis), max(chis)


def obtain_halo_edges(*filepaths):
    """Obtain the inner and outer edges (in comoving distance in Mpc/h) of the lightcone from halo catalog files.
    Inputs: 3 halo catalog files
    Outputs: inner edge (Mpc/h), outer edge (Mpc/h)"""
    chis = []
    for fp in filepaths:
        with asdf.open(fp) as f:
            chis.append(f.tree["header"]["CoordinateDistanceHMpc"])

    chis.sort()
    return 0.5 * (chis[0] + chis[1]), 0.5 * (chis[-1] + chis[-2])


def main():
    # Obtain edges from heal-counts files
    healcount_file_2 = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0677-0682.asdf"
    healcount_file_3 = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0671-0676.asdf"
    healcount_file_4 = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0665-0670.asdf"
    healcount_file_5 = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0659-0664.asdf"
    healcount_file_6 = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0653-0658.asdf"
    healcount_file_7 = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0647-0652.asdf"


    inner_edge, outer_edge = obtain_healcount_edges(healcount_file_2)
    print(f"Halo shell edges for file {healcount_file_2.name} (Mpc/h): {inner_edge:.1f} – {outer_edge:.1f}")

    inner_edge, outer_edge = obtain_healcount_edges(healcount_file_3)
    print(f"Halo shell edges for file {healcount_file_3.name} (Mpc/h): {inner_edge:.1f} – {outer_edge:.1f}")

    inner_edge, outer_edge = obtain_healcount_edges(healcount_file_4)
    print(f"Halo shell edges for file {healcount_file_4.name} (Mpc/h): {inner_edge:.1f} – {outer_edge:.1f}")

    # Obtain edges from halo catalog file
    halo_catalog_file_1 = HALO_CATALOGS_PATH / "z0.503" / "lightcone_halo_info_000.asdf"
    halo_catalog_file_3 = HALO_CATALOGS_PATH / "z0.582" / "lightcone_halo_info_000.asdf"
    halo_catalog_file_2 = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
    halo_catalog_file_4 = HALO_CATALOGS_PATH / "z0.625" / "lightcone_halo_info_000.asdf"

    halo_inner_edge, halo_outer_edge = obtain_halo_edges(halo_catalog_file_1, halo_catalog_file_2, halo_catalog_file_3)
    print(f"Halo catalog comoving distance range (Mpc/h): {halo_inner_edge:.1f} – {halo_outer_edge:.1f}")

    halo_inner_edge, halo_outer_edge = obtain_halo_edges(halo_catalog_file_4, halo_catalog_file_2, halo_catalog_file_3)
    print(f"Halo catalog comoving distance range (Mpc/h): {halo_inner_edge:.1f} – {halo_outer_edge:.1f}")


if __name__ == "__main__":
    main()



















