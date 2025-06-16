#!/bin/bash
#SBATCH --job-name=indexrsetmcar
#SBATCH -c 1
#SBATCH -t 8:00:00
#SBATCH -o /usr/xtmp/ham51/script-outputs/%j.out
#SBATCH -e /usr/xtmp/ham51/script-outputs/%j.err
#SBATCH --mem=50G
#SBATCH --array=0-23

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.count_mcar_rset_index --index $((SLURM_ARRAY_TASK_ID)) --rerun
