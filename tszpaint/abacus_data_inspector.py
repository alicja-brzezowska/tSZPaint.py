import asdf
from pathlib import Path
from tszpaint.config import HALO_CATALOGS_PATH, HEALCOUNTS_PATH

def inspect_file(filepath, name):

    print(f"\n{'='*50}")
    print(f"Inspecting: {name}")
    print(f"File path: {filepath}")
    print(f"{'='*50}")

    try:
        with asdf.open(filepath) as f:
            tree = f.tree if hasattr(f, 'tree') else f

            header_post = tree.get("header_post", {})
            if isinstance(header_post, dict) and "healpix_order" in header_post:
                print("  header_post.healpix_order:", header_post["healpix_order"])
            else:
                print("  header_post.healpix_order: (not found)")

            print(f"ASDF file with top-level keys: {list(tree.keys())}")
            for key in tree.keys():
                data = tree[key]
                if key == 'header' and isinstance(data, dict):
                    print(f"  {key}: dict with keys: {list(data.keys())}")
                elif key == 'halo_lightcone' and isinstance(data, dict):
                    print(f"  {key}: dict with keys: {list(data.keys())}")

                    tsi = data["halo_timeslice_index"]
                    print("type:", type(tsi))
                    print("shape:", tsi.shape, "dtype:", tsi.dtype)

                    print("first 20:", tsi[:20])
                    print("min/max (may read full array):", tsi.min(), tsi.max())

                if key == 'halo_timeslice' and isinstance(data, dict):
                    print(f"  {key}: dict with keys: {list(data.keys())}")
                if key == 'header_post' and isinstance(data, dict):
                    print(f"  {key}: dict with keys: {list(data.keys())}")



    except Exception as e:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                print("Text file, full content:")
                for line in lines:
                    print(f"  {line.strip()}")

def main():
    halo_catalog_file = HALO_CATALOGS_PATH / "z0.542" / "lightcone_halo_info_000.asdf"
    healcounts_file = HEALCOUNTS_PATH / "LightCone0_halo_heal-counts_Step0628-0634.asdf"

    inspect_file(halo_catalog_file, "Abacus Halo Catalog File")
    inspect_file(healcounts_file, "Abacus HEALPix Particle Counts File")

if __name__ == "__main__":
    main()