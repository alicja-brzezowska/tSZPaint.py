import numpy as np
import healpy as hp
import asdf
from pathlib import Path
import psutil
from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH
import matplotlib.pyplot as plt


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

    """Not in use; does not work with the lightcone data format."""

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
    halo_dir,
):
    halo_dir = Path(halo_dir)
    with asdf.open(halo_dir ) as af:
        h = af["header"]
        particle_mass = h["ParticleMassHMsun"]
        redshift = h["Redshift"]

        d = af["halo_lightcone"]
        positions = np.asarray(d["Interpolated_x_L2com"])
        N_particles  = np.asarray(d["Interpolated_N"])

        M_halo = N_particles.astype(np.float64) * particle_mass

        # filter out low-mass halos for painting
        positions = positions[M_halo > 10**13]
        N_particles = N_particles[M_halo > 10**13]
        M_halo = M_halo[M_halo > 10**13]

    return positions, N_particles, particle_mass, redshift



def load_abacus_healcounts(filepath):
    
    with asdf.open(filepath) as f:
        #header = dict(f["headers"]) if "headers" in f else {} # or header_post 
        particle_counts = np.array(f['data']['heal-counts'])
    return particle_counts


def load_multiple_healcounts(filepath1, filepath2, filepath3):
    counts1 = load_abacus_healcounts(filepath1)
    counts2 = load_abacus_healcounts(filepath2)
    counts3 = load_abacus_healcounts(filepath3)

    sum_counts = counts1 + counts2 + counts3
    return sum_counts


def load_abacus_for_painting(
    halo_dir,
    healcounts_file_1,
    healcounts_file_2,
    healcounts_file_3,
    nside=1024,
    return_pixels=False,
):
    pos, N_particles, particle_mass, redshift = load_abacus_halos(halo_dir)

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    theta, phi = comoving_to_sky(x, y, z)

    M_halos = N_particles.astype(np.float64) * particle_mass
    particle_counts = load_multiple_healcounts(healcounts_file_1, healcounts_file_2, healcounts_file_3)

    if return_pixels:
        halo_pixels = hp.ang2pix(nside, theta, phi, nest = True)
        return theta, phi, M_halos, particle_counts, redshift, halo_pixels

    return theta, phi, M_halos, particle_counts, redshift

import psutil
from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH

halo_dir = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"

positions, N_particles, pm, z = load_abacus_halos(halo_dir)
mem_used = psutil.Process().memory_info().rss / 1e9  # GB
print(f"Memory after loading: {mem_used:.2f} GB")
