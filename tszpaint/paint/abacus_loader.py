from dataclasses import dataclass
from pathlib import Path

import asdf
import healpy as hp
import numpy as np

from tszpaint.converters import convert_comoving_to_sky, convert_rad_to_cart
from abacusnbody.data.compaso_halo_catalog import _unpack_euler16 as unpack_euler16  # this is directly from


@dataclass
class SimulationData:
    theta: np.ndarray
    phi: np.ndarray
    m_halos: np.ndarray
    particle_counts: np.ndarray
    redshift: float
    radii_halos: np.ndarray
    eigenvalues: np.ndarray   # (n_halos, 3) semi-axis lengths in physical Mpc
    eigenvectors: np.ndarray  # (n_halos, 3, 3) rows are [min, mid, maj] eigenvectors

    @property
    def halo_xyz(self):
        return convert_rad_to_cart(self.theta, self.phi)


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


def obtain_healcount_mean_redshift(filepath: Path) -> float:
    """Return the mean redshift of the healcounts shell from its headers."""
    with asdf.open(filepath) as f:
        headers = f.tree["headers"]
        zs = [h["Redshift"] for h in headers]
    return float(np.mean(zs))


def load_abacus_halos(
    halo_dir: Path,
    logm_min: float = 11.5,
):
    with asdf.open(halo_dir) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]
        box_size = h["BoxSizeMpc"]

        d = af["halo_lightcone"]
        halo_xyz = np.asarray(d["Interpolated_x_L2com"])
        num_particles = np.asarray(d["Interpolated_N"])
        comoving_distance = np.asarray(d["Interpolated_ComovingDist"])
        halo_timeslice_index = np.asarray(d["halo_timeslice_index"])

        m_halos = num_particles.astype(np.float64) * particle_mass

        # most signal from larger halos (Battaglia 2012)
        threshold = 10**logm_min
        cut = m_halos > threshold

        halo_xyz = halo_xyz[cut]

        num_particles = num_particles[cut]
        m_halos = m_halos[cut]
        halo_timeslice_index = halo_timeslice_index[cut]
        comoving_distance = comoving_distance[cut]

        # obtain base of search radius for each halo: r{%} with % being the percentage of dark matter enclosed
        # abacus-utils stored data using the custom i16 type, convert to float
        e = af["halo_timeslice"]
        radius = np.asarray(e["r98_L2com_i16"], dtype=np.float64)
        r100_ref = np.asarray(e["r100_L2com"], dtype=np.float64)

        # triaxial halos
        eigenvalues = np.asarray(e["sigman_L2com_i16"], dtype=np.float64)
        eigenvectors = np.asarray(e["sigman_eigenvecs_L2com_u16"], dtype=np.float64)

        # u16 is a smart custom encoding of the eigenvectors, again convert to float 
        sigmar_min, sigmar_mid, sigmar_maj = unpack_euler16(eigenvectors)

        radius = radius[halo_timeslice_index]
        r100_ref = r100_ref[halo_timeslice_index]

        eigenvalues = eigenvalues[halo_timeslice_index]   
        sigmar_min = sigmar_min[halo_timeslice_index]     
        sigmar_mid = sigmar_mid[halo_timeslice_index]
        sigmar_maj = sigmar_maj[halo_timeslice_index]

        INT16SCALE = 32000
        radius = radius * r100_ref * box_size / INT16SCALE
        eigenvalues = eigenvalues * r100_ref[:, np.newaxis] * box_size / INT16SCALE

        eigenvectors = np.stack([sigmar_min, sigmar_mid, sigmar_maj], axis=1)  # (n_halos, 3, 3)

    return (
        halo_xyz,
        num_particles,
        particle_mass,
        redshift,
        radius,
        comoving_distance,
        eigenvalues,
        eigenvectors,
    )

def load_abacus_for_painting(
    halo_catalog_index: list,  # list[HaloCatalogInfo] from load_halo_catalog_index()
    healcounts_file_1: Path,
    nside: int | None = None,
    logm_min: float = 11.5,
):
    """
    """
    chi_min, chi_max = obtain_healcount_edges(healcounts_file_1)
    redshift = obtain_healcount_mean_redshift(healcounts_file_1)

    # Buffer the chi edges by the comoving r200 of a reference 10^13 Msun halo,
    # so halos whose centre is within one r200 of the boundary are excluded.
    # This prevents edge halos from being double-counted across adjacent shells.
    # from tszpaint.y_profile.y_profile import compute_R_delta, create_battaglia_profile
    # _model = create_battaglia_profile()
    # r200_physical = compute_R_delta(_model, np.array([1e13]), redshift, delta=200)[0]  # Mpc physical
    # delta_chi = r200_physical * (1 + redshift)  # convert to comoving Mpc
    # chi_min_cut = chi_min + delta_chi
    # chi_max_cut = chi_max - delta_chi
    chi_min_cut = chi_min
    chi_max_cut = chi_max

    matching = [
        cat for cat in halo_catalog_index
        if max(cat.chi_min, chi_min) <= min(cat.chi_max, chi_max)
    ]
    if not matching:
        raise ValueError(
            f"No halo catalogs overlap chi=[{chi_min:.2f}, {chi_max:.2f}]. "
            "Check your halo_catalog_index."
        )

    all_xyz, all_npart, all_radius, all_evals, all_evecs = [], [], [], [], []
    particle_mass = None
    for cat in matching:
        xyz, npart, pm, _, radius, comoving_distance, evals, evecs = load_abacus_halos(cat.file_path, logm_min=logm_min)
        if particle_mass is None:
            particle_mass = pm  # simulation-wide constant, same across files
        in_shell = (comoving_distance >= chi_min_cut) & (comoving_distance <= chi_max_cut)
        all_xyz.append(xyz[in_shell])
        all_npart.append(npart[in_shell])
        all_radius.append(radius[in_shell])
        all_evals.append(evals[in_shell])
        all_evecs.append(evecs[in_shell])

    halo_xyz = np.concatenate(all_xyz)
    num_particles = np.concatenate(all_npart)
    radius = np.concatenate(all_radius)
    eigenvalues = np.concatenate(all_evals)
    eigenvectors = np.concatenate(all_evecs)

    theta, phi = convert_comoving_to_sky(halo_xyz)
    m_halos = num_particles.astype(np.float64) * particle_mass
    particle_counts = load_abacus_healcounts(healcounts_file_1)
    if nside is not None:
        particle_counts = degrade_healcounts(particle_counts, nside_out=nside)

    return SimulationData(theta, phi, m_halos, particle_counts, redshift, radius, eigenvalues, eigenvectors)
