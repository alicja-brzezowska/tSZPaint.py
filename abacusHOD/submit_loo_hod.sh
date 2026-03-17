#!/bin/bash
#SBATCH --job-name=loo_hod
#SBATCH --account=HADZHIYSKA-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/ab2927/rds/tSZPaint.py/logs/loo_hod_%j.out
#SBATCH --error=/home/ab2927/rds/tSZPaint.py/logs/loo_hod_%j.err

mkdir -p /home/ab2927/rds/tSZPaint.py/logs
cd /home/ab2927/rds/tSZPaint.py
.venv/bin/python abacusHOD/loo_hod.py
