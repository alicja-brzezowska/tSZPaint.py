import numpy as np
import healpy as hp
import asdf
from pathlib import Path


def comoving_to_sky(x, y, z):
    """
    Convert comoving box coordinates to healpix sky coordinates.
    Abacus has the origin at the center of the box.

    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi


def load_abacus_halos(
    dirpath,
    header_file="header.asdf",
    data_file="halo_info.asdf",
):
    """
    Load the Abacus halo catalog files, obtain halo positions from the halo_info file 
    and simulation information from the header file.
    """
    dirpath = Path(dirpath)

    with asdf.open(dirpath / header_file) as f:
        header = dict(f["header"]) if "header" in f else dict(f.tree)

    with asdf.open(dirpath / data_file) as f:
        data = f["data"] if "data" in f else f.tree
        pos = np.array(data["x_L2com"][:])
        N_particles = np.array(data["N"][:])
        SO_radius = np.array(data["SO_radius"][:])

    return pos, N_particles, SO_radius, header


def load_abacus_healcounts(filepath, key="PartCounts/PartCounts_000"):
    
    """
    Load the healpix map with particle counts from the Heal_counts files.
    """
    with asdf.open(filepath) as f:
        header = dict(f["header"]) if "header" in f else {}
        particle_counts = np.array(f[key][:])
    return particle_counts, header


def halos_to_sky(halo_pos, N_particles, header, nside=8192):

    """Convert an Abacus halo catalog to sky coordinates using its header."""

    particle_mass = header["ParticleMassMsun"]
    redshift = header["Redshift"]

    x, y, z = halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2]
    theta, phi = comoving_to_sky(x, y, z)

    M_halos = N_particles.astype(np.float64) * particle_mass
    halo_pixels = hp.ang2pix(nside, theta, phi)

    return theta, phi, M_halos, redshift, halo_pixels


def load_abacus_for_painting(
    halo_dir,
    healcounts_file,
    header_file="header.asdf",
    data_file="halo_info.asdf",
    healcounts_key="PartCounts/PartCounts_000",
    nside=8192,
):
    """
    Prepare halo catalog and particle counts for painting from AbacusSummit data.
    """
    pos, N_particles, SO_radius, header = load_abacus_halos(
        halo_dir, header_file, data_file
    )

    halo_theta, halo_phi, M_halos, redshift, _ = halos_to_sky(
        pos, N_particles, header, nside
    )

    particle_counts, _ = load_abacus_healcounts(healcounts_file, healcounts_key)

    return halo_theta, halo_phi, M_halos, particle_counts, redshift


