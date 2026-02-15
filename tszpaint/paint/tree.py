import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from tszpaint.converters import convert_cart_to_rad, convert_rad_to_cart
from tszpaint.decorators import time_calls, trace_calls
from tszpaint.paint.config import PainterConfig


def angular_separation(
    theta1: np.ndarray, phi1: np.ndarray, theta2: np.ndarray, phi2: np.ndarray
):
    dtheta = theta2 - theta1
    dphi = phi2 - phi1
    a = (
        np.sin(dtheta / 2) ** 2
        + np.cos(theta1) * np.cos(theta2) * np.sin(dphi / 2) ** 2
    )
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@time_calls
@trace_calls
def build_tree(config: PainterConfig):
    """
    Build a 3D KDTree of HEALPix pixels.
    Need to convert to cartesian coordinates; as no angular KDTree in scipy.
    """
    npix = hp.nside2npix(config.nside)
    pix_indices = np.arange(npix)
    theta, phi = hp.pix2ang(config.nside, pix_indices, nest=True)
    pix_xyz = convert_rad_to_cart(theta, phi)
    tree = cKDTree(pix_xyz)
    return tree, pix_xyz, pix_indices  # NOTE: do i need pix_xyz?


@time_calls
@trace_calls
def query_tree(
    config: PainterConfig,
    halo_xyz: np.ndarray,
    theta_200: np.ndarray,
    particle_tree: cKDTree,
    particle_xyz: np.ndarray,
):
    """
    Query the tree out to N times theta_200 to find which pixels belong to which halo.
    """
    N_halos = halo_xyz.shape[0]

    search_angles = config.search_radius * theta_200
    search_radii = 2.0 * np.sin(0.5 * search_angles)

    pix_in_halos = particle_tree.query_ball_point(x=halo_xyz, r=search_radii)
    # list of arrays; each array contains pixel indices for that halo (in HealPix map)

    # define and return halo_starts and halo_counts for efficient weighting
    halo_counts = np.array([len(p) for p in pix_in_halos], dtype=np.int64)
    halo_starts = np.zeros(N_halos, dtype=np.int64)
    halo_starts[1:] = np.cumsum(
        halo_counts[:-1]
    )  # cumulative sum: exclude last; as first index is 0

    pix_in_halos = np.concatenate(
        [np.asarray(p, dtype=np.int64) for p in pix_in_halos]
    )  # flatten the array of arrays

    # create a halo index array mapping each pixel to its halo:
    halo_indices = np.repeat(np.arange(N_halos, dtype=np.int64), halo_counts)

    # exclude pixels not in any halo:
    particle_xyz = particle_xyz[pix_in_halos]
    halo_xyz = halo_xyz[halo_indices]

    # obtain the angular distances between particles and the halo centers:
    # chord_distances = np.linalg.norm(particle_xyz - halo_xyz, axis=1)       # NOTE: maybe could use directly the query?
    # distances = 2.0 * np.arcsin(np.clip(chord_distances / 2.0, -1.0, 1.0))

    pix_theta, pix_phi = convert_cart_to_rad(particle_xyz)
    halo_theta, halo_phi = convert_cart_to_rad(halo_xyz)
    distances = angular_separation(halo_theta, halo_phi, pix_theta, pix_phi)

    return pix_in_halos, distances, halo_starts, halo_counts, halo_indices
