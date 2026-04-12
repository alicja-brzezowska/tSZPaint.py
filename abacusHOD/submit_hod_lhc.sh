#!/bin/bash
# HOD LHC stacking pipeline — convolves maps in-memory (no preconvolved files saved).
# Usage: bash submit_hod_lhc.sh

set -e

LOGDIR=/home/ab2927/logs
mkdir -p "$LOGDIR"

PYTHON=/home/ab2927/rds/tSZPaint.py/.venv/bin/python
STACK=/home/ab2927/rds/tSZPaint.py/ACT_forward_model/run_stacking.py
MERGE=/home/ab2927/rds/tSZPaint.py/ACT_forward_model/merge_hod_stacks.py

# ── 1. stacking array (200 tasks, each convolves 125 maps in memory) ─────────
ARRAY_ID=$(sbatch --parsable \
  --job-name=hod_lhc \
  --account=HADZHIYSKA-SL3-CPU \
  --partition=icelake-himem \
  --nodes=1 --ntasks=1 --cpus-per-task=76 --mem=350G \
  --time=10:00:00 \
  --array=0-199 \
  --output="$LOGDIR/hod_lhc_%A_%a.out" \
  --error="$LOGDIR/hod_lhc_%A_%a.err" \
  --wrap="$PYTHON $STACK --hod-idx \$SLURM_ARRAY_TASK_ID --nproc 4")
echo "Stacking array job: $ARRAY_ID"

# ── 2. merge (runs only if all array tasks succeeded) ─────────────────────────
MERGE_ID=$(sbatch --parsable \
  --dependency=afterok:$ARRAY_ID \
  --job-name=hod_merge \
  --account=HADZHIYSKA-SL3-CPU \
  --partition=icelake \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
  --time=00:15:00 \
  --output="$LOGDIR/hod_merge_%j.out" \
  --error="$LOGDIR/hod_merge_%j.err" \
  --wrap="$PYTHON $MERGE")
echo "Merge job: $MERGE_ID"
echo "Done. full_grid.npz will appear after job $MERGE_ID completes."
