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
    radii_halos: np.ndarray



def load_abacus_healcounts(filepath: Path):
    with asdf.open(filepath) as f:
        particle_counts = np.array(f["data"]["heal-counts"])
    return particle_counts


def degrade_healcounts(particle_counts: np.ndarray, nside_out: int) -> np.ndarray:

    # TODO: FIX THIS; it doesn't actually work    
    """
    Degrade HEALPix particle counts from their current resolution to a lower resolution.
    
    The input resolution (nside_in) is inferred from the particle_counts array size
    using the HEALPix relation: npix = 12 * nside^2
    
    Args:
        particle_counts: Array of particle counts at original resolution
        nside_out: Output nside (e.g., 2048)
    
    Returns:
        Degraded particle counts array at nside_out resolution
    """
    npix_in = len(particle_counts)
    nside_in = int(np.sqrt(npix_in / 12))
    
    if nside_in == nside_out:
        return particle_counts

    degraded_counts = hp.ud_grade(particle_counts, nside_out=nside_out, order_in='NEST', order_out='NEST')
    
    return degraded_counts


def obtain_healcount_edges(filepath: Path):
    """Obtain the inner and outer edges (in comoving distance in Mpc/h) of the lightcone from a heal-counts file."""
    with asdf.open(filepath) as f:
        headers = f.tree["headers"]
        chis = [h["CoordinateDistanceHMpc"] for h in headers]
    return min(chis), max(chis)


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
        comoving_distance = np.asarray(d["Interpolated_ComovingDist"])
        halo_timeslice_index = np.asarray(d["halo_timeslice_index"])

        m_halos = num_particles.astype(np.float64) * particle_mass

        threshold = 1e13
        mask = m_halos > threshold

        # filter out low-mass halos for painting
        positions = positions[mask]

        num_particles = num_particles[mask]
        m_halos = m_halos[mask]
        halo_timeslice_index = halo_timeslice_index[mask]
        comoving_distance = comoving_distance[mask]

        e = af["halo_timeslice"]
        radius = np.asarray(e["r90_L2com_i16"], dtype=np.float64)
        radius = radius[halo_timeslice_index]

        # Convert radius from kpc/h to Mpc 
        h = 0.6774
        radius = radius / 1000.0 / h

    return positions, num_particles, particle_mass, redshift, radius, comoving_distance


def load_abacus_for_painting(
    halo_dir: Path,
    healcounts_file_1: Path,
    nside: int = 2048,
):
    pos, num_particles, particle_mass, redshift, radius, comoving_distance = load_abacus_halos(
        halo_dir,
    )

    chi_min, chi_max = obtain_healcount_edges(healcounts_file_1)
    chi_mask = (comoving_distance >= chi_min) & (comoving_distance <= chi_max)

    pos = pos[chi_mask]
    num_particles = num_particles[chi_mask]
    radius = radius[chi_mask]

    theta, phi = convert_comoving_to_sky(pos)

    m_halos = num_particles.astype(np.float64) * particle_mass
    particle_counts = load_abacus_healcounts(healcounts_file_1)
    
    particle_counts = degrade_healcounts(particle_counts, nside_out=nside)

    return SimulationData(theta, phi, m_halos, particle_counts, redshift, radius)
