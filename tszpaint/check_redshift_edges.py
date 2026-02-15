import numpy as np
import asdf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH


# ── 1. Inspect halo catalog header ───────────────────────────────────────

halo_file = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
print(f"\n{'='*70}")
print(f"HALO CATALOG: {halo_file.name}")
print(f"{'='*70}")

with asdf.open(halo_file) as af:
    print(f"Top-level keys: {list(af.keys())}")

    # Single header
    hdr = af["header"]
    print(f"\nheader keys: {list(hdr.keys())}")
    # Print all scalar values from header
    for k in sorted(hdr.keys()):
        v = hdr[k]
        if not hasattr(v, 'shape'):  # scalar or string
            print(f"  {k}: {v}")

    # Show halo data fields
    if "halo_lightcone" in af:
        hl = af["halo_lightcone"]
        print(f"\nhalo_lightcone keys: {list(hl.keys())}")
        for k in hl.keys():
            v = hl[k]
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    # Compute comoving distance range from halo positions
    pos = np.asarray(af["halo_lightcone"]["Interpolated_x_L2com"])
    N_part = np.asarray(af["halo_lightcone"]["Interpolated_N"])
    particle_mass = hdr["ParticleMassHMsun"]
    M_halo = N_part.astype(np.float64) * particle_mass

    # All halos
    r_all = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    print(f"\nAll halos: N={len(r_all)}")
    print(f"  Comoving distance range: {r_all.min():.1f} – {r_all.max():.1f} Mpc/h")

    # M > 1e13 halos (what we actually paint)
    mask = M_halo > 1e13
    r_halo = r_all[mask]
    M_halo = M_halo[mask]
    pos = pos[mask]
    print(f"\nHalos with M > 1e13: N={len(r_halo)}")
    print(f"  Comoving distance range: {r_halo.min():.1f} – {r_halo.max():.1f} Mpc/h")
    print(f"  Mass range: {M_halo.min():.2e} – {M_halo.max():.2e} Msun")


# ── 2. Inspect HEALPix heal-counts headers ───────────────────────────────

# The 3 files currently used + immediate neighbours
files_to_inspect = [
    "LightCone0_halo_heal-counts_Step0653-0658.asdf",
    "LightCone0_halo_heal-counts_Step0659-0664.asdf",
    "LightCone0_halo_heal-counts_Step0665-0670.asdf",  # currently used
    "LightCone0_halo_heal-counts_Step0671-0676.asdf",  # currently used
    "LightCone0_halo_heal-counts_Step0677-0682.asdf",  # currently used
    "LightCone0_halo_heal-counts_Step0683-0688.asdf",
    "LightCone0_halo_heal-counts_Step0689-0695.asdf",
]

print(f"\n{'='*70}")
print("HEALPIX HEAL-COUNTS FILES")
print(f"{'='*70}")

for fname in files_to_inspect:
    fpath = HEALCOUNTS_PATH / fname
    if not fpath.exists():
        print(f"\n{fname}: FILE NOT FOUND")
        continue

    print(f"\n{'-'*50}")
    print(f"{fname}")
    print(f"{'-'*50}")

    with asdf.open(fpath) as af:
        print(f"Top-level keys: {list(af.keys())}")

        # The developer says 'headers' (plural) is a list of dicts
        if "headers" in af:
            headers = af["headers"]
            print(f"\nheaders: list of {len(headers)} dicts")

            # Print keys from first header
            h0 = dict(headers[0])
            print(f"  header[0] keys: {list(h0.keys())}")

            # Print ALL headers with key fields
            for i, h in enumerate(headers):
                h = dict(h)
                # Try common keys that might identify redshift/step/chi
                info_keys = ["Redshift", "redshift", "Step", "step",
                             "ComovedDistance", "ComovingDistance",
                             "TimeSliceRedshift", "ScaleFactor", "a"]
                found = {k: h[k] for k in info_keys if k in h}
                if not found:
                    # Just print all scalar values
                    found = {k: v for k, v in h.items() if not hasattr(v, 'shape')}
                print(f"  header[{i}]: {found}")

        # Also check header_post
        if "header_post" in af:
            hp_dict = dict(af["header_post"])
            print(f"\nheader_post keys: {list(hp_dict.keys())}")
            # Print scalar values
            for k in sorted(hp_dict.keys()):
                v = hp_dict[k]
                if not hasattr(v, 'shape'):
                    print(f"  {k}: {v}")

        # Check data
        if "data" in af:
            d = af["data"]
            print(f"\ndata keys: {list(d.keys())}")
            for k in d.keys():
                v = d[k]
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")


# ── 3. Halo comoving distance histogram ──────────────────────────────────

print(f"\n{'='*70}")
print("HALO COMOVING DISTANCE DISTRIBUTION")
print(f"{'='*70}")

# Simple histogram in text
percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
for p in percentiles:
    print(f"  {p:>3}th percentile: {np.percentile(r_halo, p):.1f} Mpc/h")

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(r_halo, bins=200, alpha=0.7, color='steelblue',
            label=f"Halos (N={len(r_halo)}, M > 1e13)")

    ax.set_xlabel(r"Comoving distance [Mpc/$h$]")
    ax.set_ylabel("N halos")
    ax.set_title(f"Halo comoving distances — catalog z0.542 ({halo_file.name})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("halo_chi_distribution.png", dpi=200)
    print(f"\nSaved: halo_chi_distribution.png")
    print(">>> Once you see this + the headers output, we can compute the shell edges properly.")
except Exception as e:
    print(f"\nCouldn't make plot: {e}")