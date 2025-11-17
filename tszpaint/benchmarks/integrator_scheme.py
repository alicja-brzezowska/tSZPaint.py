import numpy as np 
from scipy.integrate import quad_vec

from y_profile import create_battaglia_profile, get_params, generalized_nfw, compute_R_delta, angular_size

def get_abel_transform(x, xc, alpha, beta, gamma):

    def integrand(z):
        return scale * 