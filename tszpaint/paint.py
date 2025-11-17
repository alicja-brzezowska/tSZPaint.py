import numpy as np 
import healpy as hp
import asdf 
from scipy.spatial import cKDTree

from config import HALO_CATALOGS_DATA_PATH
from y_profile import compute_R_delta

#CONSTANTS 

#HEALPix
NSIDE = 512 # resolution of the map
NPIX = hp.nside2npix(NSIDE)  # number of pixels in the map 
m = np.arange(NPIX) # RING ordering of pixels


def read_asdf():
    """
    Given the AbacusSummit ASDF file, find information.
    """
    halo_cat = asdf.open(HALO_CATALOGS_DATA_PATH) 
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
    M_halos = np.exp(np.random.log(12,14,N_halos))

    halo_data = halo_theta, halo_phi, M_halos

    return halo_data


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


def build_tree(data):
    """
    Build a KDTree in 3D
    """
    xyz = convert_rad_to_cart() 
    print(f"\nBuilding tree for {NPIX} pixel centers...")
    tree = cKDTree(xyz[0:2])
    return tree



def query_tree(
    halo_pos_cart,      # (N_halos, 3) - Cartesian positions of halo centers
    R_200,              # (N_halos,) - R200 for each halo 
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
        search_radius = N * R_200[i]
        
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











