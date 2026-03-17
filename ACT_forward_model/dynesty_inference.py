"""
Nested sampling inference of Battaglia parameters using the GP emulator.

  python dynesty_inference.py
  python dynesty_inference.py --nlive 500 --plot
"""

import pickle
import numpy as np
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from gp_emulator import load_emulator, predict

# Data from Liu et al. 2025, Table 1, Fig. 6
ACT_AP  = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
ACT_Y   = np.array([8.9375901426e-08, 3.5286669084e-07, 7.0189522333e-07,
                    1.2090546136e-06, 1.6113528108e-06, 1.9116110451e-06,
                    2.3413244009e-06, 2.9703678477e-06, 3.6016090038e-06])
ACT_ERR = np.array([9.4267502323e-09, 2.2639352683e-08, 3.6185523443e-08,
                    5.6818925154e-08, 8.2360102801e-08, 1.1498411717e-07,
                    1.4746886290e-07, 1.8369545709e-07, 2.1632720061e-07])

# From Liu 2025
_corr    = np.load(Path(__file__).parent / "fig6_pz2.npy")
COV_DATA = np.diag(ACT_ERR) @ _corr @ np.diag(ACT_ERR)

EMULATOR_PKL = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/stacked_profiles/all_steps_stacked.pkl")
OUT_DIR      = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/inference")
OUT_DIR.mkdir(exist_ok=True)
OUT_PKL      = OUT_DIR / "dynesty_results.pkl"
OUT_CORNER   = OUT_DIR / "corner.png"


_npz        = np.load(EMULATOR_PKL.with_suffix(".npz"), allow_pickle=True)
PARAM_NAMES = list(_npz["param_names"])
PRIOR_BOUNDS = [
    (float(np.percentile(_npz["params"][:, i], 5)), float(np.percentile(_npz["params"][:, i], 95)))
    for i in range(_npz["params"].shape[1])
]


def prior_transform(u):
    theta = np.empty(len(PARAM_NAMES))
    for i, (lo, hi) in enumerate(PRIOR_BOUNDS):
        theta[i] = lo + u[i] * (hi - lo)
    return theta


def make_log_likelihood(gps, X_scaler, y_scalers):
    n = len(ACT_Y)

    def log_likelihood(theta):
        mu, sig_gp = predict(gps, X_scaler, y_scalers, np.array(theta).reshape(1, -1))
        mu     = mu[0]
        sig_gp = sig_gp[0]

        cov_total = COV_DATA + np.diag(sig_gp ** 2)
        try:
            L = np.linalg.cholesky(cov_total)
        except np.linalg.LinAlgError:
            return -1e30
        delta     = mu - ACT_Y
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, delta))
        log_det_t = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (delta @ alpha_vec + log_det_t + n * np.log(2 * np.pi))

    return log_likelihood


def summarise(results):
    from dynesty.utils import resample_equal
    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)
    print("\nPosterior summary:")
    for i, name in enumerate(PARAM_NAMES):
        lo, mid, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"  {name}: {mid:.4f} +{hi-mid:.4f} -{mid-lo:.4f}")
    return samples


def plot_corner(results):
    n = len(PARAM_NAMES)
    fig, _ = dyplot.cornerplot(
        results,
        labels=PARAM_NAMES,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        fig=plt.subplots(n, n, figsize=(3 * n, 3 * n)),
    )
    fig.savefig(OUT_CORNER, dpi=150, bbox_inches="tight")
    print(f"Corner plot → {OUT_CORNER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    gps, X_scaler, y_scalers, _ = load_emulator()

    print(f"ACT_Y:  {ACT_Y}")
    print(f"ACT_ERR: {ACT_ERR}")
    print(f"COV diag sqrt: {np.sqrt(np.diag(COV_DATA))}")

    log_likelihood = make_log_likelihood(gps, X_scaler, y_scalers)

    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, len(PARAM_NAMES),
        nlive=args.nlive, bound="multi", sample="rwalk",
    )
    sampler.run_nested(dlogz=0.5, print_progress=True)
    results = sampler.results

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f)
    print(f"Results → {OUT_PKL}")

    summarise(results)

    if args.plot:
        plot_corner(results)
