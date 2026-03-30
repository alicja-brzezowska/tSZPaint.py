from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_TOTAL_PATH
import asdf
from pathlib import Path


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


def obtain_halo_edges_multiple(*filepaths):
    """Obtain the inner and outer edges (in comoving distance in Mpc/h) of the lightcone from halo catalog files.

    Can be called in two ways:
    1. With multiple filepaths: obtain_halo_edges(file1, file2, file3)
    2. With a single filepath: obtain_halo_edges(file_in_z_dir) - automatically finds adjacent redshift files

    Args:
        *filepaths: Either 3 halo catalog files, or 1 file to auto-find adjacent redshifts

    Returns:
        Tuple of (inner_edge, outer_edge) in comoving distance Mpc/h
    """

    if len(filepaths) == 1:
        halo_file = Path(filepaths[0])
        redshift_dir = halo_file.parent
        lightcone_dir = redshift_dir.parent

        redshift_dirs = sorted(
            [
                d
                for d in lightcone_dir.iterdir()
                if d.is_dir() and d.name.startswith("z")
            ]
        )
        redshift_names = [d.name for d in redshift_dirs]

        current_z_name = redshift_dir.name

        current_idx = redshift_names.index(current_z_name)

        if current_idx == 0:
            lower_idx = 0
            upper_idx = 1
        elif current_idx == len(redshift_dirs) - 1:
            lower_idx = len(redshift_dirs) - 2
            upper_idx = len(redshift_dirs) - 1
        else:
            lower_idx = current_idx - 1
            upper_idx = current_idx + 1

        filename = halo_file.name

        filepaths = (
            redshift_dirs[lower_idx] / filename,
            halo_file,
            redshift_dirs[upper_idx] / filename,
        )

    chis = []
    for fp in filepaths:
        with asdf.open(fp) as f:
            chis.append(f.tree["header"]["CoordinateDistanceHMpc"])

    chis.sort()
    return 0.5 * (chis[0] + chis[1]), 0.5 * (chis[-1] + chis[-2])


def main():
    # Discover all healcounts files, sorted by step number
    healcount_files = sorted(HEALCOUNTS_TOTAL_PATH.glob("*.asdf"))

    # Discover all halo z-directories, sorted by redshift
    halo_z_dirs = sorted(
        [d for d in HALO_CATALOGS_PATH.iterdir() if d.is_dir() and d.name.startswith("z")],
        key=lambda d: float(d.name[1:]),
    )

    # Compute edges for each halo z-directory
    halo_edges = {}
    for z_dir in halo_z_dirs:
        halo_file = z_dir / "lightcone_halo_info_000.asdf"
        inner, outer = obtain_halo_edges_multiple(halo_file)
        halo_edges[z_dir.name] = (inner, outer)

    # For each healcounts file, find which halo z-dirs fall within its range
    print(f"{'Healcounts file':<55} {'Range (Mpc/h)':<25} Matching halo z-dirs")
    print("-" * 120)
    for hc_file in healcount_files:
        hc_inner, hc_outer = obtain_healcount_edges(hc_file)
        matches = [
            z_name
            for z_name, (h_inner, h_outer) in halo_edges.items()
            if h_inner < hc_outer and h_outer > hc_inner  # ranges overlap
        ]
        print(
            f"{hc_file.name:<55} {hc_inner:.1f} – {hc_outer:.1f}{'':>10} {', '.join(matches) or 'none'}"
        )


if __name__ == "__main__":
    main()
