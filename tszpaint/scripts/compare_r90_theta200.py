from loguru import logger
import numpy as np

from tszpaint.config import HALO_CATALOGS_PATH
from tszpaint.cosmology.model import compute_theta_200, get_angular_size_from_comoving
from tszpaint.paint.abacus_loader import load_abacus_halos
from tszpaint.y_profile.y_profile import create_battaglia_profile

MASS_MIN = 1e13  # Msun
HALO_PATH = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"


def summarize(name: str, values: np.ndarray):
    logger.info(
        f"{name}: n={len(values):,} min={values.min():.3e} "
        f"mean={values.mean():.3e} std={values.std(ddof=1):.3e} "
        f"median={np.median(values):.3e} max={values.max():.3e}"
    )


def main():
    model = create_battaglia_profile()

    _, num_particles, particle_mass, redshift, r90_mpc, _, eigenvalues, _ = load_abacus_halos(HALO_PATH)
    m_halos = num_particles.astype(np.float64) * particle_mass

    mask = m_halos > MASS_MIN
    m_halos = m_halos[mask]
    r90_mpc = r90_mpc[mask]
    eigenvalues = eigenvalues[mask]
    eigenvalues_angular = get_angular_size_from_comoving(model, eigenvalues, redshift)

    logger.info(f"Halo catalog: {HALO_PATH}")
    logger.info(f"Redshift: {redshift}")
    logger.info(f"Mass cut: M > {MASS_MIN:.1e} Msun (kept {len(m_halos):,})")

    r90_theta = get_angular_size_from_comoving(model, r90_mpc, redshift)
    theta_200 = compute_theta_200(model, m_halos, redshift)


    summarize("rN radius", r90_mpc)
    summarize("rN radius (rad)", r90_theta)
    summarize("theta_200 (rad)", theta_200)
    summarize("eigenvalues (rad)", eigenvalues_angular)


if __name__ == "__main__":
    main()
