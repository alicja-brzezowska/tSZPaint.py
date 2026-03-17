"""
Submit 125 LOO jobs (one per held-out point) via submitit, then collect results.

Usage:
  python submit_loo_full.py          # submit all 125 jobs
  python submit_loo_full.py --collect  # collect results + make plots (run after jobs finish)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

RESULTS_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/loo_results")
OUT_SCATTER   = Path(__file__).parent / "loo_full.png"
OUT_SPAGHETTI = Path(__file__).parent / "loo_full_spaghetti.png"
OUT_PARAM     = Path(__file__).parent / "loo_full_param_space.png"
OUT_TXT       = Path(__file__).parent / "loo_full_results.txt"
OUT_TXT2      = Path(__file__).parent / "loo_full_per_axis.txt"
ERROR_THRESH  = 3.0  # percent


# ── worker function (runs on the cluster) ────────────────────────────────────

def run_loo_worker(hold_out_idx: int):
    import warnings
    warnings.filterwarnings("ignore")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gp_emulator import load_data, train, predict

    params, profiles, apertures, _ = load_data()

    mask = np.ones(len(params), dtype=bool)
    mask[hold_out_idx] = False

    gps, X_sc, y_sc = train(params[mask], profiles[mask],
                             kernel="matern", verbose=False)
    mu, sig = predict(gps, X_sc, y_sc, params[[hold_out_idx]])

    result = dict(
        idx=hold_out_idx,
        mu=mu[0],
        sig=sig[0],
        truth=profiles[hold_out_idx],
        apertures=apertures,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / f"loo_{hold_out_idx:03d}.npy", result)
    return hold_out_idx


# ── boundary distance helper ──────────────────────────────────────────────────

def boundary_distance(params):
    lo, hi = params.min(axis=0), params.max(axis=0)
    normed = (params - lo) / (hi - lo)
    return np.minimum(normed, 1 - normed).min(axis=1)


def boundary_distance_per_axis(params):
    """Returns (N, 4) array — distance to nearest boundary for each dimension."""
    lo, hi = params.min(axis=0), params.max(axis=0)
    normed = (params - lo) / (hi - lo)
    return np.minimum(normed, 1 - normed)


# ── collect + plot ────────────────────────────────────────────────────────────

def collect():
    import sys

    log = open(OUT_TXT, "w")

    def echo(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log)

    from gp_emulator import load_data
    params, profiles, apertures, _ = load_data()
    n = len(params)
    bdist = boundary_distance(params)

    result_files = sorted(RESULTS_DIR.glob("loo_*.npy"))
    if len(result_files) < n:
        echo(f"Warning: only {len(result_files)}/{n} result files found — some jobs may not have finished.")

    fracs = np.full((n, len(apertures)), np.nan)
    for f in result_files:
        r = np.load(f, allow_pickle=True).item()
        idx = r["idx"]
        fracs[idx] = (r["mu"] - r["truth"]) / r["truth"] * 100

    # ── print table ──────────────────────────────────────────────────────────
    ap_header = "  ".join(f"θ={a:.2f}" for a in apertures)
    echo(f"\n{'Point':>5}  {'bdist':>6}  {ap_header}  {'max|err|':>8}")
    echo("-" * (5 + 8 + 9 * len(apertures) + 10))
    for idx in range(n):
        if np.isnan(fracs[idx]).any():
            continue
        max_e = np.abs(fracs[idx]).max()
        row = "  ".join(f"{v:+6.2f}%" for v in fracs[idx])
        echo(f"  #{idx:3d}  {bdist[idx]:.4f}  {row}  {max_e:7.2f}%")

    # ── find cutoff ───────────────────────────────────────────────────────────
    max_err = np.abs(fracs).max(axis=1)
    order   = np.argsort(bdist)
    cutoff  = None
    for i in range(len(order) - 1, -1, -1):
        check = bdist >= bdist[order[i]]
        if np.nanmax(max_err[check]) < ERROR_THRESH:
            cutoff = bdist[order[i]]
        else:
            break

    n_kept = int((bdist >= cutoff).sum()) if cutoff is not None else n
    echo(f"\n── Cutoff for <{ERROR_THRESH}% max error ──────────────────────────")
    echo(f"   bdist >= {cutoff:.4f}  →  {n_kept}/{n} points kept  ({n - n_kept} excluded)")

    # ── scatter: max error vs bdist ───────────────────────────────────────────
    norm = plt.Normalize(vmin=0, vmax=0.5)
    cmap = cm.RdYlBu

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(bdist, max_err, color="C0", s=30, edgecolors="k", linewidths=0.3, zorder=3)
    ax.axhline(ERROR_THRESH, color="red", lw=1.0, ls=":",
               label=f"{ERROR_THRESH}% threshold")
    ax.set_xlabel("closest distance from boundary")
    ax.set_ylabel("max |fractional error| over apertures [%]")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=150, bbox_inches="tight")
    echo(f"Saved → {OUT_SCATTER}")

    # ── spaghetti ─────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for idx in np.argsort(bdist)[::-1]:
        if np.isnan(fracs[idx]).any():
            continue
        excluded = cutoff is not None and bdist[idx] < cutoff
        color = "red" if excluded else cmap(norm(bdist[idx]))
        lw    = 1.5 if excluded else 0.6
        ax2.plot(apertures, fracs[idx], "-", color=color, lw=lw, alpha=0.7)
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.axhline( ERROR_THRESH, color="red", lw=1.0, ls=":", alpha=0.7)
    ax2.axhline(-ERROR_THRESH, color="red", lw=1.0, ls=":", alpha=0.7)
    ax2.set_xlabel(r"Aperture $\theta$ [arcmin]")
    ax2.set_ylabel(r"$(Y_{\rm pred} - Y_{\rm true})\,/\,Y_{\rm true}$ [%]")
    ax2.set_title("Matérn 5/2 kernel", loc="left", fontsize=10, fontweight="bold")
    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax2, label="boundary distance")
    fig2.tight_layout()
    fig2.savefig(OUT_SPAGHETTI, dpi=150, bbox_inches="tight")
    echo(f"Saved → {OUT_SPAGHETTI}")

    # ── parameter space error map (all 6 pairwise projections) ───────────────
    from itertools import combinations
    param_names = ["alpha", "beta0", "gamma", "log10P0"]
    pairs = list(combinations(range(4), 2))   # 6 pairs

    fig3, axes3 = plt.subplots(2, 3, figsize=(13, 8))
    axes3 = axes3.flatten()

    err_norm = plt.Normalize(vmin=0, vmax=max_err.max())
    err_cmap = cm.YlOrRd

    for ax, (i, j) in zip(axes3, pairs):
        sc = ax.scatter(params[:, i], params[:, j], c=max_err,
                        cmap=err_cmap, norm=err_norm,
                        s=40, edgecolors="k", linewidths=0.3, zorder=3)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        fig3.colorbar(sc, ax=ax, label="max |frac error| [%]")

    fig3.tight_layout()
    fig3.savefig(OUT_PARAM, dpi=150, bbox_inches="tight")
    echo(f"Saved → {OUT_PARAM}")
    log.close()

    # ── per-axis distance table ───────────────────────────────────────────────
    param_names = ["alpha", "beta0", "gamma", "log10P0"]
    per_axis = boundary_distance_per_axis(params)   # (125, 4)
    mean_err = np.nanmean(np.abs(fracs), axis=1)

    with open(OUT_TXT2, "w") as f:
        header = (f"{'Point':>5}  {'mean_err':>8}  "
                  + "  ".join(f"d_{n:>8}" for n in param_names))
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for idx in range(n):
            if np.isnan(fracs[idx]).any():
                continue
            dists_row = "  ".join(f"{per_axis[idx, k]:>10.4f}" for k in range(4))
            f.write(f"  #{idx:3d}  {mean_err[idx]:8.3f}%  {dists_row}\n")

    print(f"Saved → {OUT_TXT2}")


# ── submission ────────────────────────────────────────────────────────────────

def submit():
    import submitit
    from gp_emulator import load_data
    params, _, _, _ = load_data()
    n = len(params)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder="logs/loo/%j")
    executor.update_parameters(
        slurm_partition="icelake",
        slurm_account="HADZHIYSKA-SL3-CPU",
        cpus_per_task=1,
        mem_gb=8,
        timeout_min=60,
        slurm_array_parallelism=32,
    )

    jobs = executor.map_array(run_loo_worker, list(range(n)))
    print(f"Submitted {len(jobs)} jobs")
    for job in jobs:
        print(f"  job_id={job.job_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true",
                        help="Collect results and make plots (run after jobs finish)")
    args = parser.parse_args()

    if args.collect:
        collect()
    else:
        submit()
