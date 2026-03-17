"""
Forward model: stack tSZ y-map at mock LRG positions and compute CAP-filtered profiles.

  1. Load y-map 
  2. Convolve with ACT beam (FWHM=1.6 arcmin)
  3. Load mock LRG catalog (AbacusHOD ECSV output)
  4. Convert comoving (x,y,z) → angular (theta, phi)
  5. For each galaxy: apply CAP filter at apertures 1–6 arcmin
  6. Stack (mean) over all galaxies → y_CAP(theta)
"""

import numpy as np
import healpy as hp
import asdf
from astropy.table import Table
from time import perf_counter
from loguru import logger

APERTURES = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])  # arcmin
BEAM_FWHM = 1.6   # Liu 2025, arcmin

def load_ymap(ymap_file):
    """Load y-map from ASDF"""
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
    alm = hp.map2alm(ymap, lmax=ell_max)
    ell = np.arange(ell_max+1)
    def gauss_beam(ellsq, fwhm):
        tht_fwhm = np.deg2rad(fwhm/60.)
        return np.exp(-0.5*(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))
    fl = gauss_beam(ell*(ell+1), fwhm_arcmin)
    # healpy.almxfl expects fl up to lmax of alm
    alm_fl = hp.almxfl(alm, fl[:alm.size]) if fl.size > alm.size else hp.almxfl(alm, fl)
    ymap_fl = hp.alm2map(alm_fl, nside)
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


def cap_filter(ymap, nside, theta, phi, apertures_arcmin=APERTURES):

    N_gal = len(theta)
    N_ap = len(apertures_arcmin)
    cap_values = np.zeros((N_gal, N_ap))

    ap_rad    = np.deg2rad(apertures_arcmin / 60.0)   # (N_ap,)
    outer_rad = np.sqrt(2) * ap_rad                    # (N_ap,)
    max_rad   = outer_rad[-1]                          # apertures sorted → last is largest

    t0 = perf_counter()
    for i in range(N_gal):
        if i % 50000 == 0:
            elapsed = perf_counter() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (N_gal - i) / rate if rate > 0 else float('inf')
            logger.info(f"CAP: {i}/{N_gal} ({100*i/N_gal:.0f}%)  {rate:.0f} gal/s  ETA {eta/60:.1f} min")

        vec  = hp.ang2vec(theta[i], phi[i])
        ipix = hp.query_disc(nside, vec, max_rad, nest=True)
        if len(ipix) == 0:
            continue

        pix_vec = np.array(hp.pix2vec(nside, ipix, nest=True)).T  # (N_pix, 3)
        ang     = np.arccos(np.clip(pix_vec @ vec, -1.0, 1.0))    # radians

        # Sort once by angle; use cumulative sum to get bin means without masking
        sort_idx  = np.argsort(ang)
        ang_s     = ang[sort_idx]
        y_s       = ymap[ipix[sort_idx]].astype(np.float64)
        cumsum    = np.cumsum(y_s)

        for j in range(N_ap):
            n_disc  = np.searchsorted(ang_s, ap_rad[j],    side='right')
            n_outer = np.searchsorted(ang_s, outer_rad[j], side='right')
            n_ann   = n_outer - n_disc
            if n_disc == 0 or n_ann == 0:
                continue
            disc_sum = cumsum[n_disc - 1]
            ann_sum  = cumsum[n_outer - 1] - cumsum[n_disc - 1]
            # Divide both by their respective number of pixels (means)
            cap_values[i, j] = (disc_sum / n_disc) - (ann_sum / n_ann)

    logger.info(f"CAP filter done in {perf_counter()-t0:.1f}s")
    return cap_values


def stack_profiles(ymap_file, theta, phi, output_file=None, apply_beam=True):
    """
    ymap_file : str — single pre-summed y-map (ASDF)
    theta, phi: pre-loaded, chi-filtered galaxy positions in radians
    """
    t_total = perf_counter()
    ymap, nside = load_ymap(ymap_file)
    logger.info(f"Loaded y-map, nside={nside}")

    if apply_beam:
        ymap = convolve_beam(ymap)
    else:
        logger.info("Skipping beam convolution")

    logger.info(f"{len(theta):,} galaxies")

    logger.info("Applying CAP filter...")
    cap_values = cap_filter(ymap, nside, theta, phi)

    y_stacked = cap_values.mean(axis=0)
    y_err     = cap_values.std(axis=0) / np.sqrt(len(theta))
    logger.info("Stacked CAP profile:")
    for ap, y, e in zip(APERTURES, y_stacked, y_err):
        logger.info(f"  {ap:.0f} arcmin: {y:.4e} ± {e:.4e}")
    logger.info(f"Total stack_profiles time: {perf_counter()-t_total:.1f}s")

    if output_file:
        np.savez(output_file,
                 apertures_arcmin=APERTURES,
                 y_stacked=y_stacked,
                 y_err=y_err,
                 cap_values=cap_values)
        logger.info(f"Saved to {output_file}")

    return y_stacked, y_err


