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
    phi = np.arctan2(y, x) # better than arctan(y/x), as distinguishes quadrants 
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) # [condition, value if true, value if false]
    # ensures phi is in [0, 2pi]
    return theta, phi


def load_abacus_header(header_path, wanted=("ParticleMassMsun", "Redshift")):

    wanted = set(wanted)
    out = {}
    with open(header_path, "r") as f:
        for line in f:
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k not in wanted:
                continue
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            out[k] = float(v)
            if len(out) == len(wanted):
                break
    return out    



def load_abacus_halos(
    halo_dir, data_file="halo_info/halo_info_000.asdf",
):
    halo_dir = Path(halo_dir)
    with asdf.open(halo_dir / data_file, copy_arrays=False) as af:
        d = af["data"]
        positions = np.asarray(d["x_L2com"])
        N_particles  = np.asarray(d["N"])
        R200 = np.asarray(d["SO_radius"])
    return positions, N_particles, R200



def load_abacus_healcounts(filepath, key="data/heal-counts"):
    
    with asdf.open(filepath) as f:
        header = dict(f["header"]) if "header" in f else {}
        particle_counts = np.array(f['data'][key][:])
    return particle_counts, header



def load_abacus_for_painting(
    halo_dir,
    healcounts_file,
    header_file="header",
    nside=1024,
    return_pixels=False,
):
    pos, N_particles, SO_radius = load_abacus_halos(halo_dir)

    h = load_abacus_header(Path(halo_dir) / header_file, wanted=("ParticleMassMsun", "Redshift"))
    particle_mass = h["ParticleMassMsun"]
    redshift = h["Redshift"]

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    theta, phi = comoving_to_sky(x, y, z)

    M_halos = N_particles.astype(np.float64) * particle_mass
    particle_counts, _ = load_abacus_healcounts(healcounts_file)

    if return_pixels:
        halo_pixels = hp.ang2pix(nside, theta, phi)
        return theta, phi, M_halos, particle_counts, redshift, halo_pixels

    return theta, phi, M_halos, particle_counts, redshift

