from pathlib import Path
import numpy as np
from loguru import logger

from tszpaint.interpolator import BattagliaLogInterpolator
from tszpaint.config import DATA_PATH

PYTHON_PATH = DATA_PATH/ "y_values_python.pkl"
JULIA_PATH = DATA_PATH / "battaglia_interpolation.jld2"
JAX_PATH = DATA_PATH/ "y_values_jax_2.pkl"

LOG_THETA_MIN = -7.5
LOG_THETA_MAX = 0.5
Z_MIN = 1e-3
Z_MAX = 5.0
LOG_M_MIN = 11.0
LOG_M_MAX = 15.7

#constants:
z = 0.5            
theta = 1e-4   

EPS = 1e-3

np.random.seed(2137)


def get_random_points(num_per_dim: int = 10):
    log_thetas = np.random.uniform(LOG_THETA_MIN, LOG_THETA_MAX, num_per_dim)
    zs = np.random.uniform(Z_MIN, Z_MAX, num_per_dim)
    log_ms = np.random.uniform(LOG_M_MIN, LOG_M_MAX, num_per_dim)

    return np.meshgrid(log_thetas, zs, log_ms, indexing="ij")

if __name__ == "__main__": 
    python_interpolator = BattagliaLogInterpolator.from_pickle(PYTHON_PATH) 
    logger.info("built JAX interpolator from grid") 
    julia_interpolator = BattagliaLogInterpolator.from_jld2(JULIA_PATH) 
    print("Built julia interpolator from grid") 
    jax_interpolator = BattagliaLogInterpolator.from_pickle(JAX_PATH) 
    print("Built JAX interpolator from grid") 
    
    log_thetas, zs, log_ms = get_random_points() 
    python_vals = python_interpolator.eval_for_logs(log_thetas, zs, log_ms) 
    julia_vals = julia_interpolator.eval_for_logs(log_thetas, zs, log_ms) 
    JAX_vals = jax_interpolator.eval_for_logs(log_thetas, zs, log_ms) 

    relative_errors = np.abs(python_vals - julia_vals) / np.maximum(
        np.abs(julia_vals), 1e-12
    )

    relative_errors_jax = np.abs(JAX_vals - julia_vals) / np.maximum(
        np.abs(julia_vals), 1e-12
    )
    max_idx = np.unravel_index(np.argmax(relative_errors), relative_errors.shape)
    max_python_val = python_vals[max_idx]
    max_julia_val = julia_vals[max_idx]
    max_JAX_val = JAX_vals[max_idx]
    max_relative_error = relative_errors[max_idx]
    max_relative_error_JAX = relative_errors_jax[max_idx]

    messages = (
        f"Max relative error: {max_relative_error}",
        f"Max relative error JAX: {max_relative_error_JAX}",
        f"Python value: {max_python_val}",
        f"Julia value: {max_julia_val}",
        f"JAX value: {max_JAX_val}",
    )

if max_relative_error < EPS and max_relative_error_JAX < EPS:
    for message in messages:
        logger.info(message)
else:
    for message in messages:
        logger.error(message)