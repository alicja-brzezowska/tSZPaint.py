#!/bin/bash
#SBATCH --job-name=hod_lhc_merge
#SBATCH --account=HADZHIYSKA-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=/home/ab2927/rds/tSZPaint.py/logs/hod_lhc_merge_%j.out
#SBATCH --error=/home/ab2927/rds/tSZPaint.py/logs/hod_lhc_merge_%j.err

cd /home/ab2927/rds/tSZPaint.py
.venv/bin/python abacusHOD/run_hod_lhc_stacking.py --merge
