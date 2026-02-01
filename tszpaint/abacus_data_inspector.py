import asdf
from pathlib import Path
from config import HALO_CATALOGS_DATA_PATH, HEALCOUNTS_DATA_PATH

def inspect_file(filepath, name):

    print(f"\n{'='*50}")
    print(f"Inspecting: {name}")
    print(f"File path: {filepath}")
    print(f"{'='*50}")

    try:
        with asdf.open(filepath) as f:
            tree = f.tree if hasattr(f, 'tree') else f
            print(f"ASDF file with top-level keys: {list(tree.keys())}")
            for key in tree.keys():
                data = tree[key]
                if key == 'data' and isinstance(data, dict):
                    print(f"  {key}: dict with keys: {list(data.keys())}")
    except:
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                print("Text file, full content:")
                for line in lines:
                    print(f"  {line.strip()}")
        except:
            print("Could not read file")

def main():
    header_file = HALO_CATALOGS_DATA_PATH.parent / "header"
    halo_catalog_file = HALO_CATALOGS_DATA_PATH / "halo_info_000.asdf"
    healcounts_file = HEALCOUNTS_DATA_PATH / "LightCone0_halo_heal-counts_Step0628-0634.asdf"

    inspect_file(header_file, "Abacus Header File")
    inspect_file(halo_catalog_file, "Abacus Halo Catalog File")
    inspect_file(healcounts_file, "Abacus HEALPix Particle Counts File")

if __name__ == "__main__":
    main()