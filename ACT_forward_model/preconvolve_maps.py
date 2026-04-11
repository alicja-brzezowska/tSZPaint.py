"""
Convolve all 125 y-maps with the ACT beam (once) and save to a separate directory.
"""

import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from time import perf_counter
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tszpaint.logging import setup_logging
from ACT_data_match import load_ymap, convolve_beam

YMAP_DIR   = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/Step0677-0682")
OUT_DIR    = Path("/home/ab2927/rds/hpc-work/tSZPaint_data/preconvolved_Step0677-0682")
BATCH_SIZE = 5


def _worker(src_str: str, dst_str: str) -> tuple[str, int]:
    """Run in a separate process: load, convolve, save, return (dst, nside)."""
    src, dst = Path(src_str), Path(dst_str)
    ymap, nside = load_ymap(str(src))
    ymap_conv   = convolve_beam(ymap).astype(np.float32)
    np.save(str(dst), ymap_conv)
    del ymap, ymap_conv
    gc.collect()
    return dst_str, nside


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None, help="Start index of sorted file list (inclusive)")
    parser.add_argument("--end",   type=int, default=None, help="End index of sorted file list (exclusive)")
    args = parser.parse_args()

    setup_logging("preconvolve_maps")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_fnames = sorted(f.name for f in YMAP_DIR.glob("*.asdf"))
    if args.start is not None or args.end is not None:
        all_fnames = all_fnames[args.start:args.end]
        logger.info(f"Processing slice [{args.start}:{args.end}] = {len(all_fnames)} maps")

    pending = [
        f for f in all_fnames
        if not (OUT_DIR / f.replace(".asdf", ".npy")).exists()
    ]
    logger.info(f"{len(all_fnames)} maps total, {len(pending)} still to process")

    nside_seen = set()
    done = 0

    # Work through pending files in batches of BATCH_SIZE
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]
        jobs  = {
            fname: (
                str(YMAP_DIR / fname),
                str(OUT_DIR / fname.replace(".asdf", ".npy")),
            )
            for fname in batch
        }

        t_batch = perf_counter()
        with ProcessPoolExecutor(max_workers=BATCH_SIZE) as ex:
            futures = {ex.submit(_worker, src, dst): fname
                       for fname, (src, dst) in jobs.items()}
            for fut in as_completed(futures):
                dst_str, nside = fut.result()
                nside_seen.add(nside)
                done += 1
                logger.info(f"[{done}/{len(pending)}] saved: {Path(dst_str).name}")

        logger.info(f"Batch done in {perf_counter()-t_batch:.1f}s — freeing memory")
        gc.collect()

    nside_file = OUT_DIR / "nside.txt"
    if not nside_file.exists() and nside_seen:
        nside_file.write_text(str(nside_seen.pop()))

    logger.info(f"All maps pre-convolved → {OUT_DIR}")


if __name__ == "__main__":
    main()
