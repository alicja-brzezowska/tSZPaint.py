"""
Interactive 3D scatter of LOO errors in (alpha, beta0, gamma) space.
log10P0 is marginalised over (all points shown, colour = max |frac error|).

Opens an HTML file you can view in a browser.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path

RESULTS_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/loo_results")
OUT_HTML    = Path(__file__).parent / "loo_3d.html"


def load_errors():
    from gp_emulator import load_data
    params, profiles, apertures, _ = load_data()
    n = len(params)

    max_err = np.full(n, np.nan)
    for f in sorted(RESULTS_DIR.glob("loo_*.npy")):
        r = np.load(f, allow_pickle=True).item()
        idx = r["idx"]
        frac = np.abs((r["mu"] - r["truth"]) / r["truth"] * 100)
        max_err[idx] = frac.max()

    return params, max_err


def main():
    params, max_err = load_errors()
    alpha   = params[:, 0]
    beta0   = params[:, 1]
    gamma   = params[:, 2]
    log10P0 = params[:, 3]

    hover = [
        f"#{i}<br>alpha={alpha[i]:.3f}<br>beta0={beta0[i]:.3f}"
        f"<br>gamma={gamma[i]:.3f}<br>log10P0={log10P0[i]:.3f}"
        f"<br>max err={max_err[i]:.2f}%"
        for i in range(len(params))
    ]

    fig = go.Figure(data=go.Scatter3d(
        x=alpha, y=beta0, z=gamma,
        mode="markers",
        marker=dict(
            size=6,
            color=max_err,
            colorscale="YlOrRd",
            colorbar=dict(title="max |frac error| [%]"),
            opacity=0.85,
            line=dict(width=0.5, color="black"),
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="alpha",
            yaxis_title="beta0",
            zaxis_title="gamma",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.write_html(OUT_HTML)
    print(f"Saved → {OUT_HTML}")
    print("Open in your browser to explore interactively.")


if __name__ == "__main__":
    main()
