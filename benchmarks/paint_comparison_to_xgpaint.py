import numpy as np
import healpy as hp
import time
from pathlib import Path

from tszpaint.paint import (
    paint_y_chunked,
    load_interpolator,
)
from tszpaint.config import DATA_PATH, INTERPOLATORS_PATH

# Configuration
NSIDE = 2048
N_HALOS = 10000
Z = 0.5
SEED = 42
OUTPUT_DIR = DATA_PATH / "comparison_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_catalog():
    rng = np.random.default_rng(SEED)
    
    # Halos
    halo_theta = np.pi * rng.random(N_HALOS)
    halo_phi = 2 * np.pi * rng.random(N_HALOS)
    logM = rng.uniform(14.0, 15.5, size=N_HALOS)
    M_halos = 10.0**logM
    
    # Particle counts
    npix = hp.nside2npix(NSIDE)
    baseline = 1_000_000
    contrast = rng.lognormal(mean=0.0, sigma=2.0, size=npix)
    particle_counts = rng.poisson(baseline * contrast).astype(np.int64)
    
    # Convert to RA, Dec for Julia
    dec = np.pi/2 - halo_theta
    ra = halo_phi
    
    return {
        'halo_theta': halo_theta,
        'halo_phi': halo_phi,
        'ra': ra,
        'dec': dec,
        'masses': M_halos,
        'particle_counts': particle_counts,
        'nside': NSIDE,
    }


def paint_python(catalog, use_weights=True):
    """Paint with tszpaint"""
    mode = "weights" if use_weights else "no weights"
    print("\n" + "="*60)
    print(f"TSZpaint - {mode}")
    print("="*60)

    interpolator = load_interpolator(INTERPOLATORS_PATH / "y_values_jax_2.pkl")

    t0 = time.perf_counter()
    y_map = paint_y_chunked(
        halo_theta=catalog['halo_theta'],
        halo_phi=catalog['halo_phi'],
        M_halos=catalog['masses'],
        particle_counts=catalog['particle_counts'],
        interpolator=interpolator,
        z=Z,
        nside=catalog['nside'],
        use_weights=use_weights,
    )
    t1 = time.perf_counter()

    elapsed = t1 - t0

    print(f"\n{'='*60}")
    print(f"Python painting completed in {elapsed:.2f} seconds")
    print(f"{'='*60}")
    print(f"  Min:  {y_map.min():.6e}")
    print(f"  Max:  {y_map.max():.6e}")
    print(f"  Mean: {y_map.mean():.6e}")
    print(f"  Std:  {y_map.std():.6e}")
    print(f"  Non-zero: {np.sum(y_map > 0)}/{len(y_map)}")

    return y_map, elapsed


def save_for_julia(catalog, python_time=None):
    save_dict = {
        'ra': catalog['ra'],
        'dec': catalog['dec'],
        'masses': catalog['masses'],
        'nside': catalog['nside'],
        'seed': SEED,
    }
    if python_time is not None:
        save_dict['python_time'] = python_time
    np.savez(OUTPUT_DIR / "catalog.npz", **save_dict)


def save_python_map(y_map, filename="python_map.fits"):
    hp.write_map(OUTPUT_DIR / filename, y_map, overwrite=True, dtype=np.float64)


def main():
    print(f"Configuration:")
    print(f"  N_halos: {N_HALOS}")
    print(f"  NSIDE:   {NSIDE}")
    print(f"  z:       {Z}")

    catalog = create_catalog()

    # Paint with Python - with weights
    y_map_weighted, python_time_weighted = paint_python(catalog, use_weights=True)
    save_python_map(y_map_weighted, "python_map_weighted.fits")

    # Paint with Python - without weights 
    y_map_no_weights, python_time_no_weights = paint_python(catalog, use_weights=False)
    save_python_map(y_map_no_weights, "python_map.fits")  

    print("\n" + "="*60)
    print("PYTHON: Weighted vs No-Weights comparison")
    print("="*60)
    print(f"  Total signal (weighted):    {y_map_weighted.sum():.6e}")
    print(f"  Total signal (no weights):  {y_map_no_weights.sum():.6e}")
    print(f"  Ratio (weighted/no_weights): {y_map_weighted.sum() / y_map_no_weights.sum():.4f}")

    save_for_julia(catalog, python_time_no_weights)


if __name__ == "__main__":
    main()