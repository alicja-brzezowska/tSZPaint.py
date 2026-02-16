from dataclasses import dataclass
from pathlib import Path

import asdf
import numpy as np

from tszpaint.converters import convert_comoving_to_sky


@dataclass
class SimulationData:
    theta: np.ndarray
    phi: np.ndarray
    m_halos: np.ndarray
    particle_counts: np.ndarray
    redshift: float
    radii_halos: np.ndarray


def load_abacus_halos(
    halo_dir: Path,
):
    with asdf.open(halo_dir) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]

        d = af["halo_lightcone"]
        positions = np.asarray(d["Interpolated_x_L2com"])
        num_particles = np.asarray(d["Interpolated_N"])
        halo_timeslice_index = np.asarray(d["halo_timeslice_index"])

        m_halos = num_particles.astype(np.float64) * particle_mass

        threshold = 1e13
        mask = m_halos > threshold

        # filter out low-mass halos for painting
        positions = positions[mask]
        num_particles = num_particles[mask]
        m_halos = m_halos[mask]
        halo_timeslice_index = halo_timeslice_index[mask]

        e = af["halo_timeslice"]
        radius = np.asarray(e["r90_L2com_i16"], dtype=np.float64)
        radius = radius[halo_timeslice_index]

    return positions, num_particles, particle_mass, redshift, radius


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
    nside: int = 1024,
):
    pos, num_particles, particle_mass, redshift, radius = load_abacus_halos(halo_dir)

    theta, phi = convert_comoving_to_sky(pos)

    m_halos = num_particles.astype(np.float64) * particle_mass
    particle_counts = load_multiple_healcounts(
        healcounts_file_1, healcounts_file_2, healcounts_file_3
    )

    return SimulationData(theta, phi, m_halos, particle_counts, redshift, radius)
