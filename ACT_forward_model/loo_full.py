"""
Full leave-one-out validation: all 125 points, Matérn 5/2 kernel.

Outputs:
  - printed table: point | bdist (L2) | per-dim bdist | frac error at each aperture
  - boundary distance cutoff for <3% max error
  - loo_full.png: RMS error vs bdist with cutoff line
  - loo_full_spaghetti.png: fractional error vs aperture, all 125 lines

Usage:
  python loo_full.py
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

warnings.filterwarnings("ignore")   # suppress lbfgs convergence noise

from gp_emulator import load_data, train, predict

OUT_SCATTER   = Path(__file__).parent / "loo_full.png"
OUT_SPAGHETTI = Path(__file__).parent / "loo_full_spaghetti.png"
OUT_RESULTS   = Path(__file__).parent / "loo_full_results.txt"
ERROR_THRESH  = 3.0   # percent


def boundary_distance(params):
    """Return per-dimension distances to nearest boundary wall (n x n_dims)
    and the L2 norm across all dimensions as a scalar (n,)."""
    lo, hi = params.min(axis=0), params.max(axis=0)
    normed = (params - lo) / (hi - lo)
    per_dim = np.minimum(normed, 1 - normed)   # shape (n, n_dims), each in [0, 0.5]
    scalar = np.sqrt(np.sum(per_dim ** 2, axis=1))
    return scalar, per_dim


def main():
    import sys
    params, profiles, apertures, param_names = load_data()
    _stdout = sys.stdout
    sys.stdout = open(OUT_RESULTS, "w")
    n = len(params)
    bdist, _ = boundary_distance(params)

    # ── print table header ────────────────────────────────────────────────────
    ap_header = "  ".join(f"θ={a:.2f}" for a in apertures)
    param_header = "  ".join(f"{nm[:8]:>8}" for nm in param_names)
    print(f"{'Point':>7}  {'bdist(L2)':>9}  {param_header}  {'mean_y':>12}  {ap_header}  max|err|  max_abs")
    print("-" * (10 + 11 * len(param_names) + 9 * len(apertures) + 20))

    fracs = np.zeros((n, len(apertures)))   # fractional errors [%]
    abs_errs = np.zeros((n, len(apertures)))  # absolute errors (same units as Y_CAP)

    for idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        gps, X_sc, y_sc = train(params[mask], profiles[mask], verbose=False)
        mu, _ = predict(gps, X_sc, y_sc, params[[idx]])
        abs_errs[idx] = mu[0] - profiles[idx]
        fracs[idx] = abs_errs[idx] / profiles[idx] * 100
        max_err = np.abs(fracs[idx]).max()
        max_abs = np.abs(abs_errs[idx]).max()
        mean_y = profiles[idx].mean()
        param_str = "  ".join(f"{v:8.4f}" for v in params[idx])
        print(f"  #{idx:3d}  {bdist[idx]:9.4f}  {param_str}  mean_y={mean_y:.3e}  "
              + "  ".join(f"{v:+6.2f}%" for v in fracs[idx])
              + f"  {max_err:.2f}%  max_abs={max_abs:.3e}")

    # ── find boundary cutoff for <ERROR_THRESH% ───────────────────────────────
    max_err_per_point = np.abs(fracs).max(axis=1)

    # sort points by bdist; find smallest bdist above which all errors < thresh
    order = np.argsort(bdist)
    cutoff = None
    for i in range(len(order) - 1, -1, -1):
        idx = order[i]
        # check all points with bdist >= bdist[idx]
        mask_check = bdist >= bdist[idx]
        if max_err_per_point[mask_check].max() < ERROR_THRESH:
            cutoff = bdist[idx]
        else:
            break

    n_kept = (bdist >= cutoff).sum() if cutoff is not None else 0
    print(f"\n── Boundary cutoff for <{ERROR_THRESH}% max error ──")
    print(f"   bdist >= {cutoff:.4f}  →  {n_kept}/{n} points kept")
    print(f"   ({n - n_kept} points excluded)")

    # ── scatter: RMS error vs bdist ───────────────────────────────────────────
    norm = plt.Normalize(vmin=0, vmax=0.5)
    cmap = cm.RdYlBu

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(bdist, max_err_per_point, c=bdist, cmap=cmap, norm=norm,
                    s=30, edgecolors="k", linewidths=0.3, zorder=3)
    if cutoff is not None:
        ax.axvline(cutoff, color="k", lw=1.5, ls="--",
                   label=f"cutoff bdist={cutoff:.3f}")
    ax.axhline(ERROR_THRESH, color="red", lw=1.0, ls=":",
               label=f"{ERROR_THRESH}% threshold")
    ax.set_xlabel("boundary distance")
    ax.set_ylabel("max |fractional error| over apertures [%]")
    ax.legend(fontsize=9)
    fig.colorbar(sc, ax=ax, label="boundary distance")
    fig.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_SCATTER}")

    # ── spaghetti: all 125 frac error lines ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    order_plot = np.argsort(bdist)[::-1]
    for idx in order_plot:
        color = cmap(norm(bdist[idx]))
        lw = 0.6 if bdist[idx] > (cutoff or 0) else 1.2
        ax2.plot(apertures, fracs[idx], "-", color=color, lw=lw, alpha=0.6)
    if cutoff is not None:
        # highlight excluded points
        for idx in np.where(bdist < cutoff)[0]:
            ax2.plot(apertures, fracs[idx], "-", color="r", lw=1.5, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.axhline( ERROR_THRESH, color="red", lw=1.0, ls=":", alpha=0.7)
    ax2.axhline(-ERROR_THRESH, color="red", lw=1.0, ls=":", alpha=0.7)
    ax2.set_xlabel(r"Aperture $\theta$ [arcmin]")
    ax2.set_ylabel(r"$(Y_{\rm pred} - Y_{\rm true})\,/\,Y_{\rm true}$ [%]")
    ax2.set_title("Matérn 5/2 kernel", loc="left", fontsize=10, fontweight="bold")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig2.colorbar(sm, ax=ax2, label="boundary distance")
    fig2.tight_layout()
    fig2.savefig(OUT_SPAGHETTI, dpi=150, bbox_inches="tight")

    sys.stdout.close()
    sys.stdout = _stdout
    print(f"Saved → {OUT_RESULTS}")
    print(f"Saved → {OUT_SCATTER}")
    print(f"Saved → {OUT_SPAGHETTI}")


if __name__ == "__main__":
    main()
