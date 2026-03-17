from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"
Y_PROFILE_DATA_PATH = DATA_PATH / "y_profile"

OUTPUT_PATH = Path("/home/ab2927/rds/hpc-work/tSZPaint_data")
LOGS_PATH = Path("/home/ab2927/rds/hpc-work/logs")
INTERPOLATORS_PATH = OUTPUT_PATH / "interpolators"


ABACUS_DATA_PATH = Path("/home/ab2927/rds/hpc-work/backlight_cp999")
HALO_CATALOGS_PATH = ABACUS_DATA_PATH / "lightcone_halos"
HEALCOUNTS_PATH = ABACUS_DATA_PATH / "lightcone_healpix" / "halo" / "heal-counts"
HEALCOUNTS_TOTAL_PATH = ABACUS_DATA_PATH / "lightcone_healpix" / "total" / "heal-counts"
