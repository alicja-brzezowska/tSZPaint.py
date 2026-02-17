# tSZPaint AI Coding Instructions

## Project Overview
**tSZPaint** is a Python package that paints thermal Sunyaev-Zeldovich (tSZ) maps from cosmological simulations (Abacus) using the Battaglia et al. (2016) pressure profile model. The code processes simulation data, computes Compton y parameters, and generates HEALPix maps for cosmological analysis.

## Core Architecture

### Data Flow
1. **Load Abacus Simulation Data** → `abacus_loader.py`
   - Reads halo catalogs (positions, masses, particle counts) from ASDF files
   - Filters halos by mass threshold (>1e13 M☉) to reduce computation
   - Returns `SimulationData` dataclass containing theta, phi, masses, particle counts, redshift, radii

2. **Compute HEALPix Geometry** → `pixel_search.py`
   - Uses `healpy.query_disc()` to find HEALPix pixels within search radius around each halo
   - Parallelizes across halos using `ThreadPoolExecutor`
   - Returns flattened arrays: pixel indices, distances, halo lookups

3. **Compute y-Parameter Maps** → `paint.py`
   - Interpolates Battaglia pressure profile using pre-computed lookups (`BattagliaLogInterpolator`)
   - Scales distances from comoving to sky coordinates via `get_angular_size_from_comoving()`
   - Computes weights and projects onto HEALPix grid
   - Output: 1D numpy array indexed by HEALPix pixel

4. **Visualize Maps** → `visualize.py`
   - Renders HEALPix maps in standard (log-scaled Compton y) or healpix (raw counts) modes
   - Overlays halo positions via `build_tree()` spatial index
   - Saves PNG output with configurable DPI/scaling

### Key Components

| Module | Purpose |
|--------|---------|
| `paint/main.py` | Entry point; orchestrates Abacus loading and painting pipeline |
| `paint/paint.py` | Core vectorized y-parameter computation; search radius application |
| `y_profile/y_profile.py` | Battaglia16 pressure profile model; parameter generation |
| `cosmology/model.py` | Cosmological calculations (angular size from comoving distance) |
| `constants.py` | Physical constants (Thomson cross-section, electron mass, unit conversions) |
| `config.py` | File paths to Abacus data, interpolators, and output directories |

## Critical Patterns & Conventions

### Configuration via Dataclasses
Use `@dataclass` for configuration (e.g., `PainterConfig`, `PlotConfig`):
```python
@dataclass
class PainterConfig:
    nside: int = 8192          # HEALPix resolution
    search_radius: float = 0.01 # Multiplier for theta_200
    n_bins: int = 20            # Interpolation bins
```
**Why**: Immutable, type-checked, easily serializable for job submission.

### Logging with Decorators
Use `@trace_calls` and `@time_calls` decorators to instrument compute-heavy functions:
```python
@trace_calls      # Logs function entry/exit with arguments
@time_calls       # Logs wall-clock time
def paint_y(...): pass
```
**Why**: Memory/performance profiling critical for HPC jobs; avoid hardcoded prints.

### HEALPix Pixel Operations
- Always use `nest=True` for pixel ordering (spatially localized)
- All angles in **radians**; convert with `convert_rad_to_cart()`, `convert_comoving_to_sky()`
- HEALPix arrays are 1D (12×nside²); no 2D maps in internal code

### JAX Numerics
- `constants.py` sets `jax.config.update("jax_enable_x64", False)` for 32-bit floats
- Interpolator uses JAX arrays internally; use `BattagliaLogInterpolator.from_pickle()` to load
- Do **not** mix numpy operations in hot loops with JAX arrays (performance penalty)

### ASDF File Format
- Abacus data uses HDF5-based ASDF format (similar to HDF5 API)
- Always open with `with asdf.open(filepath) as f:` context manager
- Keys are hierarchical: `f["header"]`, `f["halo_lightcone"]["Interpolated_x_L2com"]`

## Workflows & Commands

### Local Development
```bash
python -m tszpaint.paint.main              # Run main painting pipeline
python -m tszpaint.scripts.compare_healpix_maps  # Compare two maps
python -m tszpaint.scripts.inspect_healcounts    # Inspect ASDF structure
```

### HPC Job Submission (Cambridge CSD3)
```bash
sbatch submit_paint.sh              # Full Abacus painting on icelake (6h, 200GB)
sbatch submit_healpix_comparison.sh # Visualization-only job
```
**Important**: Jobs use `module load rhel8/default-ccl` for dependencies; set `OMP_NUM_THREADS=1, NUMBA_NUM_THREADS=1` to prevent oversubscription.

### Data Locations
- Abacus catalogs: `/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_halos/`
- HEALPix counts: `/home/ab2927/rds/hpc-work/backlight_cp999/lightcone_healpix/`
- Interpolators: `tszpaint/data/interpolators/` (JAX pickle format)

## Common Modifications

### Adjust Search Radius
Change `search_radius` in `PainterConfig` (default 0.01 = ±1% of theta_200):
```python
config = PainterConfig(nside=8192, search_radius=0.05)  # Wider search
```

### Change Mass Threshold
Edit `load_abacus_halos()` threshold parameter (currently 1e13 M☉):
```python
mask = m_halos > 1e14  # Only massive halos
```

### Add New y-Profile Model
Implement profile class with methods matching `Battaglia16ThermalSZProfile` signature; update `paint_y()` to accept model parameter.

### Output Format
Maps export as:
- PNG (rasterized visualization)
- FITS (if extended in future)
- Raw numpy `.npy` (intermediate products)

## Debugging Tips

1. **Memory profiling**: Monitor with `psutil` in `paint.py` (distances array is GB-scale)
2. **HEALPix validation**: Use `hp.nside2npix()` to check pixel counts
3. **Cosmology checks**: Verify `get_angular_size_from_comoving()` returns sub-degree sizes
4. **Float precision**: 32-bit JAX vs 64-bit numpy in conversions can cause subtle mismatches
5. **Redshift tracking**: ASDF files store `Redshift` in header; ensure it matches halo catalog

## Dependencies
- **Core**: abacusutils, healpy, h5py, jax, numba, scipy, pandas
- **Cosmology**: astropy, cosmo, wcosmo, quadax (numerical integration)
- **Visualization**: matplotlib, healpy (rendering)
- **Dev**: ipython, pyright (type checking)
