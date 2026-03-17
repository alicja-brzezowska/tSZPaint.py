"""
Leave-one-out validation for the GP emulator.
Runs both RBF and Matérn 5/2 kernels on the same held-out points and compares.

Usage:
  python loo_validation.py [--seed N] [--n N]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import argparse

from gp_emulator import load_data, train, predict

OUT_PATH        = Path(__file__).parent / "loo_validation.png"
OUT_PATH_MATERN = Path(__file__).parent / "loo_validation_matern.png"

KERNELS = ["rbf", "matern"]
KERNEL_LABELS = {"rbf": "RBF kernel", "matern": "Matérn 5/2 kernel"}


def boundary_distance(params):
    lo, hi = params.min(axis=0), params.max(axis=0)
    normed = (params - lo) / (hi - lo)
    dist_to_edge = np.minimum(normed, 1 - normed)
    return dist_to_edge.min(axis=1)


def run_loo(params, profiles, indices, kernel):
    results = []
    n = len(indices)
    for k, idx in enumerate(indices):
        print(f"  [{k+1}/{n}] idx={idx}")
        mask = np.ones(len(params), dtype=bool)
        mask[idx] = False
        gps, X_scaler, y_scalers = train(params[mask], profiles[mask])
        mu, sig = predict(gps, X_scaler, y_scalers, params[[idx]])
        results.append(dict(idx=idx, mu=mu[0], sig=sig[0], truth=profiles[idx]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    params, profiles, apertures, param_names = load_data()
    bdist = boundary_distance(params)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(params), size=args.n, replace=False)
    dists = bdist[indices]
    print(f"Hold-out indices: {indices}\n")

    all_results = {}
    for kernel in KERNELS:
        print(f"── {KERNEL_LABELS[kernel]} ──")
        all_results[kernel] = run_loo(params, profiles, indices, kernel)

    # ── summary / comparison figure ──────────────────────────────────────────
    norm = plt.Normalize(vmin=0, vmax=0.5)
    cmap = cm.RdYlBu

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, kernel in enumerate(KERNELS):
        results = all_results[kernel]
        fracs = np.array([(r["mu"] - r["truth"]) / r["truth"] * 100 for r in results])
        rms_per_point = np.sqrt(np.mean(fracs**2, axis=1))

        # ── spaghetti ────────────────────────────────────────────────────────
        ax = axes[0][col]
        order = np.argsort(dists)[::-1]
        for i in order:
            color = cmap(norm(dists[i]))
            ax.plot(apertures, fracs[i], "-o", color=color, ms=4, lw=1.2, alpha=0.85)
            ax.text(apertures[-1] + 0.08, fracs[i, -1], f"#{results[i]['idx']}",
                    fontsize=6.5, va="center", color=color)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(r"Aperture $\theta$ [arcmin]")
        ax.set_ylabel(r"$(Y_{\rm pred} - Y_{\rm true})\,/\,Y_{\rm true}$ [%]")
        ax.set_xlim(apertures[0] - 0.2, apertures[-1] + 0.8)
        ax.set_title(KERNEL_LABELS[kernel], loc="left", fontsize=10, fontweight="bold")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="boundary distance")

        # ── RMS vs bdist ─────────────────────────────────────────────────────
        ax2 = axes[1][col]
        ax2.scatter(dists, rms_per_point, color="C0", s=60,
                    edgecolors="k", linewidths=0.5, zorder=3)
        ax2.set_xlabel("boundary distance")
        ax2.set_ylabel(r"RMS fractional error [%]")
        for i in range(len(results)):
            ax2.annotate(f"#{results[i]['idx']}", (dists[i], rms_per_point[i]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)

        pulls = np.concatenate([(r["mu"] - r["truth"]) / r["sig"] for r in results])
        print(f"\n{KERNEL_LABELS[kernel]}  pulls: mean={pulls.mean():.3f}  std={pulls.std():.3f}")
        print(f"  mean RMS frac error: {rms_per_point.mean():.2f}%  max: {rms_per_point.max():.2f}%")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")

    # ── Matérn-only validation figure (spaghetti + RMS scatter) ──────────────
    results_m = all_results["matern"]
    fracs_m   = np.array([(r["mu"] - r["truth"]) / r["truth"] * 100 for r in results_m])
    rms_m     = np.sqrt(np.mean(fracs_m**2, axis=1))

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes2[0]
    for i in np.argsort(dists)[::-1]:
        color = cmap(norm(dists[i]))
        ax.plot(apertures, fracs_m[i], "-o", color=color, ms=4, lw=1.2, alpha=0.85)
        ax.text(apertures[-1] + 0.08, fracs_m[i, -1], f"#{results_m[i]['idx']}",
                fontsize=6.5, va="center", color=color)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"Aperture $\theta$ [arcmin]")
    ax.set_ylabel(r"$(Y_{\rm pred} - Y_{\rm true})\,/\,Y_{\rm true}$ [%]")
    ax.set_xlim(apertures[0] - 0.2, apertures[-1] + 0.8)
    ax.set_title("Matérn 5/2 kernel", loc="left", fontsize=10, fontweight="bold")
    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax, label="boundary distance")

    ax2 = axes2[1]
    ax2.scatter(dists, rms_m, color="C0", s=60, edgecolors="k", linewidths=0.5, zorder=3)
    ax2.set_xlabel("boundary distance")
    ax2.set_ylabel(r"RMS fractional error [%]")
    for i in range(len(results_m)):
        ax2.annotate(f"#{results_m[i]['idx']}", (dists[i], rms_m[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)

    fig2.tight_layout()
    fig2.savefig(OUT_PATH_MATERN, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH_MATERN}")


if __name__ == "__main__":
    main()
