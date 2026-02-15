from dataclasses import dataclass

import healpy as hp
import numpy as np

from tszpaint.paint.abacus_loader import SimulationData
from tszpaint.y_profile.y_profile import compute_R_delta, create_battaglia_profile

RNG_SEED = 17


@dataclass
class MockDataGenerator:
    """Class for mock particle and halo catalog generation.
    Parameters:
        * n_halos: Number of halos to generate in the mock catalog.
        * n_pixels: Number of pixels for the mock particle count map.
        * baseline_density: Baseline density for the Poisson distribution of particle counts.
        * overdensity_sigma: Sigma for the lognormal distribution of overdensity contrast.
        * seed: Random seed for reproducibility.
    """

    n_halos: int
    n_pixels: int
    nside: int
    baseline_density: float = 1e9
    overdensity_sigma: float = 2.0
    redshift: float = 0.5
    seed: int = RNG_SEED

    def generate_mock_particle_counts(self):
        """Create mock data of particle counts, mimicking Abacussummit data structure."""
        rng = np.random.default_rng(seed=self.seed)
        contrast = rng.lognormal(
            mean=0.0, sigma=self.overdensity_sigma, size=self.n_pixels
        )
        lam = self.baseline_density * contrast
        particle_counts = rng.poisson(lam=lam).astype(int)
        return particle_counts

    def create_mock_halo_catalogs(self):
        """Create halo-catalog mock data for testing."""
        model = create_battaglia_profile()
        rng = np.random.default_rng(RNG_SEED)
        halo_theta = np.pi * rng.random(self.n_halos)
        halo_phi = 2 * np.pi * rng.random(self.n_halos)
        logM = rng.uniform(15.5, 16.5, size=self.n_halos)
        m_halos = 10.0**logM
        radii = compute_R_delta(model, m_halos, self.redshift)
        return halo_theta, halo_phi, m_halos, radii

    def generate_simulation_data(self):
        theta, phi, m_halos, radii = self.create_mock_halo_catalogs()
        particle_counts = self.generate_mock_particle_counts()
        halo_pixels = hp.ang2pix(self.nside, theta, phi, nest=True)
        return SimulationData(
            theta, phi, m_halos, particle_counts, self.redshift, halo_pixels, radii
        )
