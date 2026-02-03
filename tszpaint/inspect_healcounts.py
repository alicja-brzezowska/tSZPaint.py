import asdf
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_asdf.py <filepath>")
    sys.exit(1)

filepath = sys.argv[1]

with asdf.open(filepath) as af:
    print(f"\n{'='*70}")
    print(f"File: {filepath}")
    print(f"{'='*70}\n")
    
    def print_tree(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for key in obj.keys():
                val = obj[key]
                print(f"{prefix}{key}: {type(val).__name__}", end="")
                if hasattr(val, 'shape'):
                    print(f" shape={val.shape} dtype={val.dtype}")
                elif isinstance(val, (list, dict)):
                    print(f" (len={len(val)})")
                else:
                    print()
                if isinstance(val, dict) and indent < 3:
                    print_tree(val, indent + 1)
        elif isinstance(obj, list) and obj:
            print(f"{prefix}[0]: {type(obj[0]).__name__}")
            if isinstance(obj[0], dict):
                print_tree(obj[0], indent + 1)
    
    print("Top-level keys:", list(af.keys()))
    print()
    print_tree(af.tree)  