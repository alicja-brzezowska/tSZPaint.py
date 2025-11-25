import time
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad_vec, quad, simpson, trapezoid

from tszpaint.y_profile import (
    get_params,
    generalized_nfw,
    nfw_los,
)

# CONSTANTS: give M200 and z and rest follows from the GNFW
M200 = 10e14
z = 0.5

params = get_params(M200, z)
xc, alpha, beta, gamma, c200 = get_params(M200, z)

# quad_vec: for integration of vectors 

def los_quadvec(x, zmax=1e5, rtol=1e-9, scale = 1):
    """Obtain the gNFW distribution along the line of sight"""
    def integrand(y):
        r_3d = np.sqrt(y**2 + x**2)
        return scale * generalized_nfw(r_3d, xc, alpha, beta, gamma)

    integral, error = quad_vec(integrand, 0.0, zmax, epsrel=rtol) 
    return 2.0 * integral / scale


def los_quad(x, zmax=1e5, rtol=1e-9, scale = 1):
    def integrand(y):
        r_3d = np.sqrt(y**2 + x**2)
        return scale * generalized_nfw(r_3d, xc, alpha, beta, gamma)
    res, err = quad(integrand, 0.0, zmax, epsrel=rtol)
    return 2.0 * res / scale, err


def los_simpson(x, zmax=1e5, n=2000000, scale = 1):
    y = np.linspace(0.0, zmax, n)
    r_3d = np.sqrt(y**2 + x**2)
    integrand = scale * generalized_nfw(r_3d, xc, alpha, beta, gamma)
    res = simpson(integrand, y)
    return 2.0 * res / scale

def los_trapezoid(x, zmax = 1e5, scale = 1):
    y = np.linspace(0.0, zmax, 200000)
    r_3d = np.sqrt(y**2 + x**2)
    integrand = scale * generalized_nfw(r_3d, xc, alpha, beta, gamma)
    res = trapezoid(integrand, y)
    return 2.0 * res / scale


def los_abel_quad(x, rmax=1e5, rtol=1e-9, scale = 1):
    """
    Transform the variables to integrate with respect to r, not z; 
    Still uses the quad function for integration. 
    """
    def integrand(r):
        return scale * generalized_nfw(r, xc, alpha, beta, gamma) * r / np.sqrt(r**2 - x**2)
    r0 = x * (1.0 + 1e-6)
    res, err = quad(integrand, r0, rmax, epsrel=rtol)
    return 2.0 * res/ scale, err
    
x_vals = np.logspace(-2, 1, 50) * xc  


def los_abel_simpson(x, rmax=1e5, n=2000000, scale = 1):
    r = np.linspace(0.0, rmax, n)
    integrand = scale * generalized_nfw(r, xc, alpha, beta, gamma) * r / np.sqrt(r**2 - x**2)
    res = simpson(integrand, r)
    return 2.0 * res / scale   


# ANALYSIS

# WITH SCALING FACTOR

I_qv  = np.array([los_quadvec(x, scale = 1e12) for x in x_vals])
I_q   = np.array([los_quad(x, scale = 1e12)[0] for x in x_vals])      
I_s   = np.array([los_simpson(x, scale = 1e12) for x in x_vals])
I_ab_quad  = np.array([los_abel_quad(x, scale = 1e12)[0] for x in x_vals])
I_ab_s = np.array([los_abel_simpson(x, scale = 1e12) for x in x_vals])
I_trap = np.array([los_trapezoid(x, scale = 1e12) for x in x_vals])

rel_err_q_vec = np.abs(I_qv - I_q) / np.abs(I_q)
rel_err_q  = np.abs(I_q  - I_q) / np.abs(I_q)
rel_err_s  = np.abs(I_s  - I_q) / np.abs(I_q)
rel_err_ab = np.abs(I_ab_quad - I_q) / np.abs(I_q)
rel_err_ab_s  = np.abs(I_ab_s  - I_q) / np.abs(I_q)
rel_err_trap = np.abs(I_trap  - I_q) / np.abs(I_q)


print("max rel error quad_vec:     ", rel_err_q_vec.max())
print("max rel error quad:     ", rel_err_q.max())
print("max rel error simpson:  ", rel_err_s.max())
print("max rel error abel:     ", rel_err_ab.max())
print("max rel error abel simpson:     ", rel_err_ab_s.max())
print("max rel error trapezoid:     ", rel_err_ab_s.max())


# plot the difference with respect to quad
plt.figure()
plt.semilogx(x_vals/xc, rel_err_q_vec, label="quad_vec - quad")
plt.semilogx(x_vals/xc, rel_err_s, label="simpson - quad")
plt.semilogx(x_vals/xc, rel_err_ab, label="abel quad - quad")
plt.semilogx(x_vals/xc, rel_err_ab_s, label="abel simpson - quad")
plt.semilogx(x_vals/xc, rel_err_trap, label="trap - quad")

plt.axhline(0.0, color="k", lw=0.5)
plt.ylabel("relative difference")
plt.xlabel(r"$x/x_c$")
plt.legend()
plt.tight_layout()
plt.show()


"""
plt.figure()
plt.loglog(x_vals / xc, I_qv, "--", label="quad_vec")
plt.loglog(x_vals / xc, I_q,  ":", label="quad")
plt.loglog(x_vals / xc, I_s,  "-.", label="simpson")
plt.loglog(x_vals / xc, I_ab_quad, ".", label="abel quad")
plt.loglog(x_vals / xc, I_ab_s,  "-..", label="abel simpson")
plt.xlabel(r"$x / x_c$")
plt.ylabel("LOS integral value")
#plt.ylim(1/1000, 1/100)
plt.legend()
plt.tight_layout()
plt.show()
"""

# WITHOUT SCALING FACTOR:

I_qv_no_scale  = np.array([los_quadvec(x) for x in x_vals])
I_q_no_scale   = np.array([los_quad(x)[0] for x in x_vals])      
I_s_no_scale   = np.array([los_simpson(x) for x in x_vals])
I_ab_no_scale  = np.array([los_abel_quad(x)[0] for x in x_vals])

plt.figure()
plt.loglog(x_vals / xc, I_qv_no_scale, "--", label="quad_vec")
plt.loglog(x_vals / xc, I_q_no_scale,  ":", label="quad")
plt.loglog(x_vals / xc, I_s_no_scale,  "-.", label="simpson")
plt.loglog(x_vals / xc, I_ab_no_scale, ".", label="abel")
plt.xlabel(r"$x / x_c$")
plt.ylabel("LOS integral value")
plt.legend()
plt.tight_layout()
plt.show()


#Comments:

#1. 
#QUAD VS QUAD VEC:

#QUAD:
# uses the fortran package QUADPACK 

#QUAD-VEC
#Uses Fortrans QUADPACK implementation DQAG: however have some difference to the QUADPACK approach;
# Instead of subdividing one interval at a time, 
# the algorithm subdivides N intervals with largest errors at once. 
# This enables (partial) parallelization of the integration.
#error of 0.6% for a scalar value; 

#In XGPAINT in julia there is the function: QUADGK: which is the equivalent of python quad;



#2.
#ABEL transform vs (x^{2} + y^{2}):
#in the denominator: and singular at the lower limit: 15% off




# Time of simpson with large number of steps vs quad



x0 = xc

scale_bench = 1e12

I_q_ref, err_q_ref = los_quad(x0, scale=scale_bench)

N_vals = np.unique((np.logspace(2, 6, 20)).astype(int))  
times_simpson = []
errs_simpson = []

for N in N_vals:
    if N % 2 == 0:
        N += 1

    t0 = time.perf_counter()
    I_s_val = los_simpson(x0, n=N, scale=scale_bench)
    t1 = time.perf_counter()

    times_simpson.append(t1 - t0)
    rel_err = np.abs(I_s_val - I_q_ref) / np.abs(I_q_ref)
    errs_simpson.append(rel_err)

times_simpson = np.array(times_simpson)
errs_simpson = np.array(errs_simpson)

tol = 1e-9
mask_good = errs_simpson < tol
if np.any(mask_good):
    idx_star = np.argmax(mask_good) 
    N_star = N_vals[idx_star]
    t_sim_star = times_simpson[idx_star]
else:
    N_star = None
    t_sim_star = None

t0 = time.perf_counter()
_ = los_quad(x0, scale=scale_bench)
t1 = time.perf_counter()
t_quad = t1 - t0



plt.figure()
plt.loglog(N_vals, errs_simpson, marker="o", label="Simpson rel. error")
plt.axhline(tol, ls="--", color="k", label=r"target $10^{-9}$")
if N_star is not None:
    plt.axvline(N_star, ls=":", color="grey", label=fr"$N^\ast = {N_star}$")
plt.xlabel("N (Simpson grid points)")
plt.ylabel(r"relative error $|I_\mathrm{S} - I_\mathrm{Q}| / |I_\mathrm{Q}|$")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(N_vals, times_simpson, marker="o", label="Simpson runtime")
if N_star is not None:
    plt.axvline(N_star, ls=":", color="grey", label=fr"$N^\ast = {N_star}$")
plt.axhline(t_quad, ls="--", color="k", label="quad runtime")
plt.xlabel("N (Simpson grid points)")
plt.ylabel("runtime [s]")
plt.legend()
plt.tight_layout()
plt.show()


if N_star is not None:
    print(f"Target rel error {tol:g} reached at N* = {N_star}")
    print(f"Simpson time at N*: {t_sim_star:.3e} s")
    print(f"quad time:          {t_quad:.3e} s")
    print(f"Simpson / quad time ratio: {t_sim_star / t_quad:.2f}")
else:
    print(f"Target rel error {tol:g} not reached up to N = {N_vals.max()}")
