import numpy as np
import healpy as hp
from scipy.spatial import cKDTree


from tszpaint.y_profile import (
    create_battaglia_profile,
    compute_R_delta,
    angular_size,
)
from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.config import DATA_PATH


#HEALPix
NSIDE = 512 # resolution of the map
NPIX = hp.nside2npix(NSIDE)  # number of pixels in the map 
m = np.arange(NPIX) # RING ordering of pixels


MODEL = create_battaglia_profile()

PYTHON_PATH = DATA_PATH / "y_values_python.pkl"
JAX_PATH = DATA_PATH / "y_values_jax_2.pkl"
JULIA_PATH = DATA_PATH / "battaglia_interpolation.jld2"
HALO_CATALOGS_PATH = DATA_PATH / "file_name"

Z = 0.5 


def read_asdf():
    """
    Given the AbacusSummit ASDF file, find information.
    """
    halo_cat = asdf.open(HALO_CATALOGS_PATH) 
    halo_cat.info()


def create_mock_particle_data(NPIX, m):
    """
    Create mock data for testing, mimicking Abacussummit data structure.
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
    """
    Create halo-catalog mock data for testing.
    Contains halo-centre position.
    """
    N_halos = 100
    #randomly allocate halo-center positions in HEALPix coordinates
    halo_theta = np.pi * np.random.rand(N_halos)
    halo_phi = 2 * np.pi * np.random.rand(N_halos)
    logM = np.random.uniform(12.0, 14.0, size=N_halos)
    M_halos = 10.0**logM


    return halo_theta, halo_phi, M_halos


def convert_rad_to_cart(data):
    """
    Given radial coordinates, convert to cartesian.
    Return the whole inputted data set.
    """
    theta = data[0]
    phi = data[1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.column_stack([x, y, z]), data[2:]
    return xyz


def compute_theta_200(model, M_halos, Z = 0.5, delta=200):
    """
    Compute θ_200 (angular radius) for each halo.
    """
    theta_200 = np.empty_like(M_halos, dtype=float)
    for M in M_halos:
        R_200 = compute_R_delta(model, M, Z, delta=delta)  
        theta_200[M] = angular_size(model, R_200, Z)       
    return theta_200


def load_interpolator(path=JAX_PATH):
    return BattagliaLogInterpolator.from_pickle(path)


def build_tree(nside=NSIDE):
    """
    Build a 3D KDTree of HEALPix pixels
    """
    npix = hp.nside2npix(nside)
    pix_indices = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pix_indices)
    pix_xyz = convert_rad_to_cart(theta, phi)
    tree = cKDTree(pix_xyz)
    return tree, pix_xyz, pix_indices



def query_tree(
    halo_pos_cart,      # (N_halos, 3) - Cartesian positions of halo centers
    theta_200,              # (N_halos,) - theta200 for each halo 
    particle_tree,      # cKDTree of particle/pixel positions
    particle_pos_cart,  # (N_particles, 3) - Cartesian positions of particles
    N=4,                # Multiple of R_200 to search
):
    """
    Query the tree out to N times R_200 to find which particles belong to which halo.
    Calculate the distances of each pixel with the halo centre.

    Assume that the input data contains positions of halo centers in cartesian, and that R_200 has previously been calculated.
    """
    N_halos = len(halo_pos_cart)
    
    pixel_indices = []
    distances = []
    halo_start = np.zeros(N_halos, dtype=np.int64)
    halo_count = np.zeros(N_halos, dtype=np.int64)
    
    print(f"Querying {N_halos} halos (search radius = {N}×R200c)...")
    
    for i in range(N_halos):
        halo_center = halo_pos_cart[i] 
        search_radius = N * theta_200[i]
        
        particles_in_halo = particle_tree.query_ball_point(halo_center, r=search_radius)
        
        halo_start[i] = len(pixel_indices)
        halo_count[i] = len(particles_in_halo)
        
        if len(particles_in_halo) > 0:

            particle_positions = particle_pos_cart[particles_in_halo]
            
            displacements = particle_positions - halo_center
            distances_in_halo = np.linalg.norm(displacements, axis=1)
            
            pixel_indices.extend(particles_in_halo)
            distances.extend(distances_in_halo)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_halos} halos...")
    
    pixel_indices = np.array(pixel_indices, dtype=np.int64)
    distances = np.array(distances, dtype=np.float64)

    return pixel_indices, distances, halo_start, halo_count



def weigh_particle_contr():
    """
    There is a well defined profile for each halo (m(r)) - the GNFW Battaglia profile: which gives a predicted 
    y-compton value at a given R. 
    Calculate the y-compton of each pixel, ensuring the mean gives the predicted value.
    """



def paint_y():











