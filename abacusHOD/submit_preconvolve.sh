#!/bin/bash
#SBATCH --job-name=preconvolve
#SBATCH --account=HADZHIYSKA-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/ab2927/rds/tSZPaint.py/logs/preconvolve_%j.out
#SBATCH --error=/home/ab2927/rds/tSZPaint.py/logs/preconvolve_%j.err

mkdir -p /home/ab2927/rds/tSZPaint.py/logs
cd /home/ab2927/rds/tSZPaint.py
.venv/bin/python abacusHOD/run_hod_lhc_stacking.py --preconvolve