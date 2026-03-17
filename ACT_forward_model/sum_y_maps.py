"""
Sum y_maps from all step directories for each parameter combination.
3 step dirs × 125 param files → 125 summed maps.

Parallelized over parameter files.
Overwrites the target (last) step directory in place.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import asdf


DATA_DIR = Path("/home/ab2927/rds/hpc-work/tSZPaint_data")
STEP_DIRS = [
    DATA_DIR / "Step0605-0610",
    #DATA_DIR / "Step0611-0616",
    DATA_DIR / "Step0677-0682",
]


def sum_one_file(fname: str, overwrite: bool = False) -> str:
    out_path = STEP_DIRS[-1] / fname

    if out_path.exists() and not overwrite:
        return f"SKIP {fname} (already exists)"

    combined = None
    header = None

    for step_dir in STEP_DIRS:
        fpath = step_dir / fname
        if not fpath.exists():
            return f"WARNING {fname}: missing {fpath}"

        with asdf.open(str(fpath), lazy_load=True) as f:
            y = np.asarray(f.tree["data"]["y_map"], dtype=np.float32)

            if combined is None:
                combined = y
                header = dict(f.tree["header"])
            else:
                combined += y

    if combined is None:
        return f"ERROR {fname}: no data found"

    header["summed_steps"] = [d.name for d in STEP_DIRS]
    header.pop("healpix_file", None)

    tree = {"data": {"y_map": combined}, "header": header}

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    af = asdf.AsdfFile(tree)
    af.write_to(str(tmp_path), all_array_compression="blsc")
    os.replace(tmp_path, out_path)

    return f"DONE {fname}"


def sum_y_maps(overwrite: bool = False, nproc: int = 6):
    param_files = sorted(p.name for p in STEP_DIRS[-1].glob("*.asdf"))
    print(f"Using {nproc} workers")
    print(f"Output directory (in-place overwrite): {STEP_DIRS[-1]}")

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = {
            ex.submit(sum_one_file, fname, overwrite): fname
            for fname in param_files
        }

        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                msg = fut.result()
                print(f"[{i}/{len(param_files)}] {msg}", flush=True)

                if msg.startswith("DONE"):
                    done += 1
                elif msg.startswith("SKIP"):
                    skipped += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"[{i}/{len(param_files)}] ERROR {fname}: {e}", flush=True)
                failed += 1

    print("\nFinished.")
    print(f"  done    : {done}")
    print(f"  skipped : {skipped}")
    print(f"  failed  : {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum y_maps across step directories")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in the target step dir",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=6,
        help="Number of worker processes",
    )
    args = parser.parse_args()

    sum_y_maps(overwrite=args.overwrite, nproc=args.nproc)