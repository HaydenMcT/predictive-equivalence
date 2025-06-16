#!/bin/bash
#SBATCH --job-name=indexrsetmcar
#SBATCH -c 1
#SBATCH -t 9:00:00
#SBATCH -o /usr/xtmp/ham51/script-outputs/%j.out
#SBATCH -e /usr/xtmp/ham51/script-outputs/%j.err
#SBATCH --mem=100G
#SBATCH --array=0-47%50

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.introduce_mcar_rset_index --index $((SLURM_ARRAY_TASK_ID)) --rerun
