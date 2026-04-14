"""
Forward model: stack tSZ y-map at mock LRG positions and compute CAP or RING-RING-filtered profiles.

  1. Load y-map 
  2. Convolve with ACT beam (FWHM=1.6 arcmin)
  3. Load mock LRG catalog (AbacusHOD ECSV output)
  4. Convert comoving (x,y,z) → angular (theta, phi)
  5. For each galaxy: apply CAP or RING-RING filter at apertures 1–6 arcmin
  6. Stack (mean) over all galaxies → y_CAP(theta) or y_RR(theta)
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import healpy as hp
import asdf
from astropy.table import Table
from time import perf_counter
from loguru import logger

APERTURES_CAP = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])  # arcmin, CAP filter
APERTURES_RR  = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])  # arcmin, ring-ring filter
APERTURES = APERTURES_CAP  # default
BEAM_FWHM = 1.6   # Liu 2025, arcmin

def load_ymap(ymap_file):
    """Load y-map from ASDF or pre-convolved .npy file."""
    if str(ymap_file).endswith('.npy'):
        ymap  = np.load(str(ymap_file))
        nside = hp.get_nside(ymap)
        return ymap, nside
    with asdf.open(ymap_file) as f:
        ymap = np.array(f['data']['y_map'])
        nside = int(f['header']['nside'])
    return ymap, nside


def convolve_beam(ymap, fwhm_arcmin=BEAM_FWHM):
    """Convolve y-map with ACT beam using custom Gaussian beam filter (Fourier space)."""
    logger.info(f"Convolving beam (custom): FWHM={fwhm_arcmin} arcmin")
    t0 = perf_counter()
    nside = hp.get_nside(ymap)
    ell_max = 12000
    alm = hp.map2alm(hp.reorder(ymap, n2r=True), lmax=ell_max)  # map2alm requires RING
    ell = np.arange(ell_max+1)
    def gauss_beam(ellsq, fwhm):
        tht_fwhm = np.deg2rad(fwhm/60.)
        return np.exp(-0.5*(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))
    fl = gauss_beam(ell*(ell+1), fwhm_arcmin)
    alm_fl = hp.almxfl(alm, fl[:alm.size]) if fl.size > alm.size else hp.almxfl(alm, fl)
    ymap_fl = hp.reorder(hp.alm2map(alm_fl, nside), r2n=True)  # back to NESTED
    logger.info(f"Beam convolution done in {perf_counter()-t0:.1f}s")
    return ymap_fl


def load_lrg_catalog(lrg_file):
    """Load AbacusHOD ECSV catalog. Returns (theta, phi) in radians (HEALPix convention)."""
    t = Table.read(lrg_file, format='ascii.ecsv')
    x, y, z = np.array(t['x']), np.array(t['y']), np.array(t['z'])
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)           
    phi = np.arctan2(y, x) % (2*np.pi) 
    return theta, phi


def cap_filter(ymap, nside, theta, phi, apertures_arcmin=APERTURES, n_workers=None):

    if n_workers is None:
        n_workers = os.cpu_count()

    N_gal = len(theta)
    N_ap  = len(apertures_arcmin)
    cap_values = np.zeros((N_gal, N_ap))

    ap_rad    = np.deg2rad(apertures_arcmin / 60.0)
    outer_rad = np.sqrt(2) * ap_rad
    max_rad   = outer_rad[-1]

    def process_galaxy(i):
        vec       = hp.ang2vec(theta[i], phi[i])
        ipix_ring = hp.query_disc(nside, vec, max_rad, nest=False)  # RING: no nside limit
        if len(ipix_ring) == 0:
            return i, np.zeros(N_ap)

        pix_vec = np.array(hp.pix2vec(nside, ipix_ring, nest=False)).T
        ang     = np.arccos(np.clip(pix_vec @ vec, -1.0, 1.0))

        sort_idx  = np.argsort(ang)
        ang_s     = ang[sort_idx]
        ipix_nest = hp.ring2nest(nside, ipix_ring)
        y_s       = ymap[ipix_nest[sort_idx]].astype(np.float64)
        cumsum    = np.cumsum(y_s)

        row = np.zeros(N_ap)
        for j in range(N_ap):
            n_disc  = np.searchsorted(ang_s, ap_rad[j],    side='right')
            n_outer = np.searchsorted(ang_s, outer_rad[j], side='right')
            n_ann   = n_outer - n_disc
            if n_disc == 0 or n_ann == 0:
                continue
            disc_sum = cumsum[n_disc - 1]
            ann_sum  = cumsum[n_outer - 1] - cumsum[n_disc - 1]
            row[j]   = (disc_sum / n_disc) - (ann_sum / n_ann)
        return i, row

    CHUNK = 50_000   # max futures in flight at once — keeps memory bounded
    t0 = perf_counter()
    k  = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for start in range(0, N_gal, CHUNK):
            end = min(start + CHUNK, N_gal)
            chunk_futures = [executor.submit(process_galaxy, i) for i in range(start, end)]
            for future in as_completed(chunk_futures):
                i, row = future.result()
                cap_values[i] = row
                k += 1
                if k % 50000 == 0:
                    elapsed = perf_counter() - t0
                    rate = k / elapsed if elapsed > 0 else 0
                    eta  = (N_gal - k) / rate if rate > 0 else float('inf')
                    logger.info(f"CAP: {k}/{N_gal} ({100*k/N_gal:.0f}%)  {rate:.0f} gal/s  ETA {eta/60:.1f} min")

    logger.info(f"CAP filter done in {perf_counter()-t0:.1f}s")
    return cap_values


def ring_ring_filter(ymap, nside, theta, phi, apertures_arcmin=APERTURES_RR, ring_width_arcmin=0.5, n_workers=None):
    """Non-cumulative ring-ring filter. For each aperture θ_d:
      θ_0 = θ_d - ring_width (inner edge), θ_outer = sqrt(2θ_d² - θ_0²) (equal-area outer edge)
      F = mean(y in [θ_0, θ_d]) - mean(y in [θ_d, θ_outer])
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    N_gal = len(theta)
    N_ap  = len(apertures_arcmin)
    rr_values = np.zeros((N_gal, N_ap))

    theta_d   = np.deg2rad(apertures_arcmin / 60.0)
    theta_0   = np.maximum(np.deg2rad((apertures_arcmin - ring_width_arcmin) / 60.0), 0.0)
    theta_out = np.sqrt(2 * theta_d**2 - theta_0**2)
    max_rad   = theta_out[-1]

    def process_galaxy(i):
        vec       = hp.ang2vec(theta[i], phi[i])
        ipix_ring = hp.query_disc(nside, vec, max_rad, nest=False)
        if len(ipix_ring) == 0:
            return i, np.zeros(N_ap)

        pix_vec = np.array(hp.pix2vec(nside, ipix_ring, nest=False)).T
        ang     = np.arccos(np.clip(pix_vec @ vec, -1.0, 1.0))

        sort_idx  = np.argsort(ang)
        ang_s     = ang[sort_idx]
        ipix_nest = hp.ring2nest(nside, ipix_ring)
        y_s       = ymap[ipix_nest[sort_idx]].astype(np.float64)
        cumsum    = np.cumsum(y_s)

        row = np.zeros(N_ap)
        for j in range(N_ap):
            i0   = np.searchsorted(ang_s, theta_0[j],   side='right')
            id_  = np.searchsorted(ang_s, theta_d[j],   side='right')
            iout = np.searchsorted(ang_s, theta_out[j], side='right')

            n_inner = id_  - i0
            n_outer = iout - id_
            if n_inner == 0 or n_outer == 0:
                continue

            sum_inner = cumsum[id_ - 1]  - (cumsum[i0 - 1] if i0 > 0 else 0.0)
            sum_outer = cumsum[iout - 1] - cumsum[id_ - 1]
            row[j]    = (sum_inner / n_inner) - (sum_outer / n_outer)
        return i, row

    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_galaxy, i): i for i in range(N_gal)}
        for k, future in enumerate(as_completed(futures)):
            if k % 50000 == 0:
                elapsed = perf_counter() - t0
                rate = k / elapsed if elapsed > 0 else 0
                eta  = (N_gal - k) / rate if rate > 0 else float('inf')
                logger.info(f"RR: {k}/{N_gal} ({100*k/N_gal:.0f}%)  {rate:.0f} gal/s  ETA {eta/60:.1f} min")
            i, row = future.result()
            rr_values[i] = row

    logger.info(f"Ring-ring filter done in {perf_counter()-t0:.1f}s")
    return rr_values


def stack_profiles(ymap_file, theta, phi, output_file=None, apply_beam=True, n_workers=None,
                   filter_type="ring_ring"):
    """
    ymap_file   : str — single pre-summed y-map (ASDF)
    theta, phi  : pre-loaded, chi-filtered galaxy positions in radians
    filter_type : "ring_ring" or "cap"
    """
    t_total = perf_counter()
    ymap, nside = load_ymap(ymap_file)
    logger.info(f"Loaded y-map, nside={nside}")

    if apply_beam:
        ymap = convolve_beam(ymap)
    else:
        logger.info("Skipping beam convolution")

    logger.info(f"{len(theta):,} galaxies")

    if filter_type == "cap":
        logger.info("Applying CAP filter...")
        values = cap_filter(ymap, nside, theta, phi, n_workers=n_workers)
        apertures = APERTURES_CAP
        label = "CAP"
    else:
        logger.info("Applying ring-ring filter...")
        values = ring_ring_filter(ymap, nside, theta, phi, n_workers=n_workers)
        apertures = APERTURES_RR
        label = "ring-ring"

    y_stacked = values.mean(axis=0)
    y_err     = values.std(axis=0) / np.sqrt(len(theta))
    logger.info(f"Stacked {label} profile:")
    for ap, y, e in zip(apertures, y_stacked, y_err):
        logger.info(f"  {ap:.1f} arcmin: {y:.4e} ± {e:.4e}")
    logger.info(f"Total stack_profiles time: {perf_counter()-t_total:.1f}s")

    if output_file:
        np.savez(output_file,
                 apertures_arcmin=apertures,
                 y_stacked=y_stacked,
                 y_err=y_err,
                 values=values)
        logger.info(f"Saved to {output_file}")

    return y_stacked, y_err


