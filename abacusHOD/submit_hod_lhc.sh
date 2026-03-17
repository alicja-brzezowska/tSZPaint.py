#!/bin/bash
# Submit the HOD LHC stacking array + auto-chain the merge step.
# Usage: bash submit_hod_lhc.sh

set -e

LOGDIR=/home/ab2927/rds/tSZPaint.py/logs
mkdir -p "$LOGDIR"

# ── 1. preconvolve (1 node, ~1 hr) ──────────────────────────────────────────
PRECONV_ID=$(sbatch --parsable \
  --job-name=hod_preconv \
  --account=HADZHIYSKA-SL3-CPU \
  --partition=icelake \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40GB \
  --time=02:00:00 \
  --output="$LOGDIR/hod_preconv_%j.out" \
  --error="$LOGDIR/hod_preconv_%j.err" \
  --wrap="cd /home/ab2927/rds/tSZPaint.py && .venv/bin/python abacusHOD/run_hod_lhc_stacking.py --preconvolve")
echo "Preconvolve job: $PRECONV_ID"

# ── 2. stacking array (200 tasks, each ~10 min) ─────────────────────────────
ARRAY_ID=$(sbatch --parsable \
  --dependency=afterok:$PRECONV_ID \
  --job-name=hod_lhc \
  --account=HADZHIYSKA-SL3-CPU \
  --partition=icelake \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24GB \
  --time=00:30:00 \
  --array=0-199 \
  --output="$LOGDIR/hod_lhc_%A_%a.out" \
  --error="$LOGDIR/hod_lhc_%A_%a.err" \
  --wrap="cd /home/ab2927/rds/tSZPaint.py && .venv/bin/python abacusHOD/run_hod_lhc_stacking.py --idx \$SLURM_ARRAY_TASK_ID")
echo "Stacking array job: $ARRAY_ID"

# ── 3. merge (runs only if all array tasks succeeded) ───────────────────────
MERGE_ID=$(sbatch --parsable \
  --dependency=afterok:$ARRAY_ID \
  --job-name=hod_merge \
  --account=HADZHIYSKA-SL3-CPU \
  --partition=icelake \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4GB \
  --time=00:05:00 \
  --output="$LOGDIR/hod_merge_%j.out" \
  --error="$LOGDIR/hod_merge_%j.err" \
  --wrap="cd /home/ab2927/rds/tSZPaint.py && .venv/bin/python abacusHOD/run_hod_lhc_stacking.py --merge")
echo "Merge job: $MERGE_ID"
echo "Done. hod_lhc_stacked.npz will appear after job $MERGE_ID completes."
