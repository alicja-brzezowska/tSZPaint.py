#!/bin/bash
#SBATCH --job-name=step0628_zoom
#SBATCH --account=HADZHIYSKA-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/ab2927/rds/tSZPaint.py/logs/step0628_zoom_%j.out
#SBATCH --error=/home/ab2927/rds/tSZPaint.py/logs/step0628_zoom_%j.err

cd /home/ab2927/rds/tSZPaint.py
.venv/bin/python -m tszpaint.scripts.check_step0628_zoom
