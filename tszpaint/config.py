from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"
INTERPOLATORS_PATH = DATA_PATH / "interpolators"
Y_PROFILE_DATA_PATH = DATA_PATH / "y_profile"


ABACUS_DATA_PATH = Path("/home/ab2927/rds/hpc-work/backlight_cp999")
HALO_CATALOGS_PATH = ABACUS_DATA_PATH / "halos" / "z0.625" / "halo_info"
HEALCOUNTS_PATH = ABACUS_DATA_PATH / "lightcone_healpix" / "halo" / "heal-counts"


