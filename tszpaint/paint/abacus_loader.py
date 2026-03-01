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
    """
    Degrade HEALPix particle counts from their current resolution to a lower resolution for testing.
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

    degraded_counts = hp.ud_grade(
        particle_counts, nside_out=nside_out, order_in="NEST", order_out="NEST"
    )

    return degraded_counts


def obtain_healcount_edges(filepath: Path):
    """Obtain the inner and outer edges (in comoving distance in Mpc/h) of the lightcone from a heal-counts file."""
    with asdf.open(filepath) as f:
        headers = f.tree["headers"]
        chis = [h["CoordinateDistanceHMpc"] for h in headers]
    return min(chis), max(chis)


def load_abacus_halos(
    halo_dir: Path,
):
    with asdf.open(halo_dir) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]
        box_size = h["BoxSizeMpc"]

        d = af["halo_lightcone"]
        positions = np.asarray(d["Interpolated_x_L2com"])
        num_particles = np.asarray(d["Interpolated_N"])
        comoving_distance = np.asarray(d["Interpolated_ComovingDist"])
        halo_timeslice_index = np.asarray(d["halo_timeslice_index"])

        m_halos = num_particles.astype(np.float64) * particle_mass

        # most signal from larger halos (Battaglia 2012)
        threshold = 1e12
        cut = m_halos > threshold

        positions = positions[cut]

        num_particles = num_particles[cut]
        m_halos = m_halos[cut]
        halo_timeslice_index = halo_timeslice_index[cut]
        comoving_distance = comoving_distance[cut]

        # obtain base of search radius for each halo: r{%} with % being the percentage of dark matter enclosed
        # abacus-utils stored data using the custom i16 type, convert to float
        e = af["halo_timeslice"]
        radius = np.asarray(e["r98_L2com_i16"], dtype=np.float64)
        r100_ref = np.asarray(e["r100_L2com"], dtype=np.float64)
        radius = radius[halo_timeslice_index]
        r100_ref = r100_ref[halo_timeslice_index]

        INT16SCALE = 32000
        radius = radius * r100_ref * box_size / INT16SCALE

    return positions, num_particles, particle_mass, redshift, radius, comoving_distance


def load_abacus_for_painting(
    halo_dir: Path,
    healcounts_file_1: Path,
    nside: int | None = None,
):
    pos, num_particles, particle_mass, redshift, radius, comoving_distance = (
        load_abacus_halos(
            halo_dir,
        )
    )

    chi_min, chi_max = obtain_healcount_edges(healcounts_file_1)
    chi_range = (comoving_distance >= chi_min) & (comoving_distance <= chi_max)

    pos = pos[chi_range]
    num_particles = num_particles[chi_range]
    radius = radius[chi_range]

    theta, phi = convert_comoving_to_sky(pos)

    m_halos = num_particles.astype(np.float64) * particle_mass
    particle_counts = load_abacus_healcounts(healcounts_file_1)
    if nside is not None:
        particle_counts = degrade_healcounts(particle_counts, nside_out=nside)

    return SimulationData(theta, phi, m_halos, particle_counts, redshift, radius)
