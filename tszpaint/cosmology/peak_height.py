"""
σ(M), growth factor, and peak height for AbacusSummit (Planck 2018) cosmology.

Uses the Eisenstein & Hu (1998) no-wiggle transfer function, normalized to σ_8=0.811.
All masses in M_sun/h, distances in Mpc/h.
"""
import numpy as np
from scipy.integrate import quad

# AbacusSummit base cosmology (Planck 2018)
H       = 0.6736
OMEGA_M = 0.3075
OMEGA_B = 0.0486
NS      = 0.9649
SIGMA8  = 0.811
DELTA_C = 1.686

RHO_CRIT0 = 2.775e11   # (M_sun/h) / (Mpc/h)^3


# ── Growth factor ────────────────────────────────────────────────────────────

def _efunc(z):
    return np.sqrt(OMEGA_M * (1.0 + z)**3 + (1.0 - OMEGA_M))


def _growth_integrand(z):
    return (1.0 + z) / _efunc(z)**3


_D0 = quad(_growth_integrand, 0.0, np.inf, limit=300)[0]


def growth_factor(z):
    """Linear growth factor D(z), normalized to D(0)=1."""
    Dz = quad(_growth_integrand, z, np.inf, limit=300)[0]
    return _efunc(z) * Dz / _D0   # E(0)=1 for flat ΛCDM


# ── Transfer function + σ(M) ─────────────────────────────────────────────────

def _eh98_transfer(k_h):
    """Eisenstein & Hu (1998) no-wiggle transfer function.
    k_h : wavenumber in h/Mpc (scalar or array).
    """
    Gamma = OMEGA_M * H * np.exp(-OMEGA_B * (1.0 + np.sqrt(2.0 * H) / OMEGA_M))
    q  = k_h / Gamma
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def _tophat_w(x):
    """Top-hat window function W(x) = 3(sin x − x cos x)/x³."""
    return np.where(
        x < 1e-3,
        1.0 - x**2 / 10.0,
        3.0 * (np.sin(x) - x * np.cos(x)) / x**3,
    )


def _sigma_integrand(lnk, R):
    k  = np.exp(lnk)
    T  = _eh98_transfer(k)
    Pk = k**NS * T**2          # unnormalized shape: P(k) ∝ k^ns T²(k)
    W  = _tophat_w(k * R)
    return k**3 * Pk * W**2 / (2.0 * np.pi**2)


_R8 = 8.0  # Mpc/h — radius enclosing 8 Mpc/h sphere
_SIG8_UNNORM_SQ = quad(
    _sigma_integrand, np.log(1e-5), np.log(1e4), args=(_R8,), limit=1000
)[0]


def sigma_m(M_h, z=0.0):
    """RMS linear matter fluctuation σ(M, z) for M in M_sun/h.

    Normalized to σ_8 = 0.811 (AbacusSummit). Includes linear growth factor.
    """
    rho_m0 = OMEGA_M * RHO_CRIT0                          # mean matter density at z=0
    R      = (3.0 * M_h / (4.0 * np.pi * rho_m0))**(1.0/3.0)   # Mpc/h
    sig_sq = quad(_sigma_integrand, np.log(1e-5), np.log(1e4), args=(R,), limit=1000)[0]
    sig0   = SIGMA8 * np.sqrt(sig_sq / _SIG8_UNNORM_SQ)
    return sig0 * growth_factor(z)


def peak_height(M_h, z):
    """Peak height ν(M, z) = δ_c / σ(M, z)."""
    return DELTA_C / sigma_m(M_h, z)
