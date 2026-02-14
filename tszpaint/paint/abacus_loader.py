from dataclasses import dataclass
from pathlib import Path

import asdf
import healpy as hp
import numpy as np

from tszpaint.converters import convert_comoving_to_sky


@dataclass
class SimulationData:
    theta: np.ndarray
    phi: np.ndarray
    m_halos: np.ndarray
    particle_counts: np.ndarray
    redshift: float
    halo_pixels: np.ndarray


def load_abacus_halos(
    halo_dir,
):
    halo_dir = Path(halo_dir)
    with asdf.open(halo_dir) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]

        d = af["halo_lightcone"]
        positions = np.asarray(d["Interpolated_x_L2com"])
        N_particles = np.asarray(d["Interpolated_N"])

        M_halo = N_particles.astype(np.float64) * particle_mass

        # filter out low-mass halos for painting
        positions = positions[M_halo > 10**13]
        N_particles = N_particles[M_halo > 10**13]
        M_halo = M_halo[M_halo > 10**13]

    return positions, N_particles, particle_mass, redshift


def load_abacus_healcounts(filepath: Path):
    with asdf.open(filepath) as f:
        particle_counts = np.array(f["data"]["heal-counts"])
    return particle_counts


def load_multiple_healcounts(
    filepath1: Path,
    filepath2: Path,
    filepath3: Path,
):
    counts1 = load_abacus_healcounts(filepath1)
    counts2 = load_abacus_healcounts(filepath2)
    counts3 = load_abacus_healcounts(filepath3)

    sum_counts = counts1 + counts2 + counts3
    return sum_counts


def load_abacus_for_painting(
    halo_dir: Path,
    healcounts_file_1: Path,
    healcounts_file_2: Path,
    healcounts_file_3: Path,
    nside=1024,
):
    pos, N_particles, particle_mass, redshift = load_abacus_halos(halo_dir)

    theta, phi = convert_comoving_to_sky(pos)

    M_halos = N_particles.astype(np.float64) * particle_mass
    particle_counts = load_multiple_healcounts(
        healcounts_file_1, healcounts_file_2, healcounts_file_3
    )
    halo_pixels = hp.ang2pix(nside, theta, phi, nest=True)

    return SimulationData(theta, phi, M_halos, particle_counts, redshift, halo_pixels)
