import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
import adsf 

from tszpaint.y_profile import (
    create_battaglia_profile,
    compute_R_delta,
    angular_size,
)
from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.config import DATA_PATH

#HEALPix
NSIDE = 1024 # resolution of the map

MODEL = create_battaglia_profile() 
PYTHON_PATH = DATA_PATH / "y_values_python.pkl" 
JAX_PATH = DATA_PATH / "y_values_jax_2.pkl" 
JULIA_PATH = DATA_PATH / "battaglia_interpolation.jld2" 
HALO_CATALOGS_PATH = DATA_PATH / "file_name"


Z = 0.5 
def read_asdf(): 
    """ Given the AbacusSummit ASDF file, find information. """ 
    halo_cat = asdf.open(HALO_CATALOGS_PATH) 
    halo_cat.info()


def create_mock_particle_data(NPIX, m): 
    """ Create mock data for testing, mimicking Abacussummit data structure. 
    Contains random number of particles per pixel. 
    Inputs: HEALPix specific data 
    Outputs: pixel radial coordinates, number of particles per pixel 
    """ 
    rng = np.random.default_rng(seed = 28) 
    rints = rng.integers(low = 10, high = 300, size = NPIX) 
    # convert to radial coordinates 
    theta, phi = hp.pix2ang(NSIDE, m) 
    particle_data = theta, phi, [rints[i] for i in m] 
    return particle_data


def create_mock_halo_catalogs(NPIX, m): 
    """ Create halo-catalog mock data for testing. Contains halo-centre position. """ 
    N_halos = 100 #randomly allocate halo-center positions in HEALPix coordinates 
    halo_theta = np.pi * np.random.rand(N_halos) 
    halo_phi = 2 * np.pi * np.random.rand(N_halos) 
    logM = np.random.uniform(12.0, 14.0, size=N_halos) 
    M_halos = 10.0**logM 
    return halo_theta, halo_phi, M_halos

def convert_rad_to_cart(theta, phi): 
    """ Given radial coordinates, convert to cartesian. Return the whole inputted data set. """ 
    x = np.sin(theta) * np.cos(phi) 
    y = np.sin(theta) * np.sin(phi) 
    z = np.cos(theta) 
    xyz = np.column_stack([x, y, z]) 
    return xyz

def compute_theta_200(
    model: Battaglia16ThermalSZProfile,
    M_halos: np.ndarray,
    Z: float = 0.5,
    delta: int = 200,
):
    """Compute θ_200 (angular radius) for each halo."""
    R_200 = compute_R_delta(model, M_halos, Z, delta=delta)
    return angular_size(model, R_200, Z)

def load_interpolator(path=JAX_PATH): 
    return BattagliaLogInterpolator.from_pickle(path)


def build_tree(nside=NSIDE): 
    """ Build a 3D KDTree of HEALPix pixels """ 
    npix = hp.nside2npix(nside) # number of pixels in the map 
    pix_indices = np.arange(npix) # RING ordering of pixels 
    theta, phi = hp.pix2ang(nside, pix_indices) 
    pix_xyz = convert_rad_to_cart(theta, phi) 
    tree = cKDTree(pix_xyz) 
    return tree, pix_xyz, pix_indices

# account for chi for particles when healpix to cartesian

def query_tree(
    halo_xyz: np.ndarray,  # (N_halos, 3) - Cartesian positions of halo centers (unit vectors)
    theta_200: np.ndarray,  # (N_halos,) - theta200 for each halo (in radians)
    particle_tree: cKDTree,  # cKDTree of pixel positions (we only use it for index lookup)
    particle_xyz: np.ndarray,  # (N_pixels, 3) - Cartesian positions of pixels (unit vectors)
    N: int = 4,  # Multiple of theta_200 to search
):
    """
    Query the tree out to N times theta_200 to find which pixels belong to which halo.
    Return: flat pixel_indices, angular distances (radians), halo_start, halo_count.
    """
    N_halos = len(halo_xyz)

    pixel_indices = []
    distances = []
    halo_starts = np.zeros(N_halos, dtype=np.int64)
    halo_counts = np.zeros(N_halos, dtype=np.int64)

    print(f"Querying {N_halos} halos (search radius = {N}×theta200)...")

    search_angles = N * theta_200
    search_radii = 2.0 * np.sin(0.5 * search_angles) # again chi
    pix_in_halos = particle_tree.query_ball_point(x=halo_xyz, r=search_radii)
    pixel_indices = pix_in_halos.ravel()  # [[1, 2, 3], [4]] -> [1, 2, 3, 4]
    halo_counts = np.array([len(indices) for indices in pixel_indices])
    halo_starts = np.cumsum(halo_counts) - 1  # [1, 2, 3] -> [1, 3, 6] -> [0, 2, 5]

    indices_repeated = [
        [np.repeat(index, count)] for index, count in enumerate(halo_counts)
    ]  # [1, 2, 1] -> [[0], [1, 1], [2]]

    indices_repeated = np.array(
        [idx for row in indices_repeated for idx in row]
    )  # [0, 1, 1, 2]
    pixel_locations = np.array(
        [particle_xyz[idx] for idx in indices_repeated]
    )  # [0, 1, 1, 2] -> [[x0,y0,z0], [x1,y1,z1], [x1,y1,z1], [x2,y2,z2]]

    cosangs = np.clip(pixel_locations @ halo_xyz, -1.0, 1.0)
    distances = np.arccos(cosangs)

    return pixel_indices, distances, halo_starts, halo_counts



def weigh_particle_contr(
    pixel_indices: np.ndarray,
    distances: np.ndarray,
    halo_start: np.ndarray,
    halo_count: np.ndarray,
    theta_200: np.ndarray,
    particle_counts: np.ndarray,
    N: float = 4.0,
    nbins: int = 20,
):
    """
    Compute per-pixel weights that encode asymmetry for each halo.

    For each halo:
      - x = theta / theta_200
      - bin x in [0, N]
      - in each bin, compute mean particle count
      - weight = count / mean_count 
    Returns:
      weights: same shape as `distances` (flat array aligned with pixel_indices).
    """
    N_halos = len(theta_200)
    weights = np.ones_like(distances, dtype=float)

    # radial bins in units of theta_200 (x = theta/theta_200)
    bin_edges = np.linspace(0.0, N, nbins + 1)
    x = distances / theta_200
    x_valid_mask = (x >= 0.0) & (x <= N)
    x = np.where(
        x_valid_mask, x, 0.0
    )  
    thetas = np.where(x_valid_mask, distances, -1)
    counts = np.where(x_valid_mask, particle_counts, 0)
    bin_ids = np.clip(np.digitize(x, bin_edges) - 1, 0, nbins - 1)

    mean_counts = np.zeros(nbins, dtype=float)
    for b in range(nbins):
        in_bin = bin_ids == b
        if np.any(in_bin):
            mean_counts[b] = counts[in_bin].mean()
        else:
            mean_counts[b] = 0.0

    w_h = np.ones_like(thetas)
    for b in range(nbins):
        in_bin = bin_ids == b
        if not np.any(in_bin):
            w_h[in_bin] = 0.0
        m = mean_counts[b]
        if m > 0.0:
            w_h[in_bin] = counts[in_bin] / m
        else:
            w_h[in_bin] = 1.0

    return w_h



def paint_y(
    halo_theta: np.ndarray,
    halo_phi: np.ndarray,
    M_halos: np.ndarray,
    particle_counts: np.ndarray,
    interpolator: BattagliaLogInterpolator,
    z: float = Z,
    nside: int = NSIDE,
    N: float = 4.0,
    nbins: int = 20,
):
    
    tree, pix_xyz, pix_indices = build_tree(nside)
    npix = len(pix_indices)

    
    halo_xyz = convert_rad_to_cart(halo_theta, halo_phi)
    theta_200 = compute_theta_200(MODEL, M_halos, Z=z, delta=200)

    
    pixel_indices_flat, distances_flat, halo_start, halo_count = query_tree(
        halo_xyz=halo_xyz,
        theta_200=theta_200,
        particle_tree=tree,
        particle_xyz=pix_xyz,
        N=N,
    )
    
    weights = weigh_particle_contr(
        pixel_indices=pixel_indices_flat,
        distances=distances_flat,
        halo_start=halo_start,
        halo_count=halo_count,
        theta_200=theta_200,
        particle_counts=particle_counts,
        N=N,
        nbins=nbins,
    )
    # Painting 
    y_map = np.zeros(npix, dtype=float)
    N_halos = len(M_halos)

    for h in range(N_halos):
        start = halo_start[h]
        count = halo_count[h]
        if count == 0:
            continue

        sl = slice(start, start + count)
        pix_idx_h = pixel_indices_flat[sl]
        theta_h = distances_flat[sl]
        w_h = weights[sl]

        log_theta_h = np.log(theta_h + 1e-40)
        log_M_h = np.full_like(theta_h, np.log10(M_halos[h]), dtype=float)
        z_h = np.full_like(theta_h, z, dtype=float)

        y_iso_h = interpolator.eval_for_logs(log_theta_h, z_h, log_M_h)
        y_pix_h = y_iso_h * w_h

        y_map[pix_idx_h] += y_pix_h

    return y_map


def main():
    halo_theta, halo_phi, M_halos = create_mock_halo_catalogs(NSIDE, np.arange(hp.nside2npix(NSIDE)))

    npix = hp.nside2npix(NSIDE)
    _, _, particle_counts = create_mock_particle_data(npix, np.arange(npix))
    particle_counts = np.array(particle_counts, dtype=int)

    interpolator = load_interpolator(JAX_PATH)

    # Paint 
    y_map = paint_y(
        halo_theta=halo_theta,
        halo_phi=halo_phi,
        M_halos=M_halos,
        particle_counts=particle_counts,
        interpolator=interpolator,
        z=Z,
        nside=NSIDE,
    )
    hp.write_map("y_map.fits", y_map, overwrite=True)


if __name__ == "__main__":
    main()













