#!/bin/bash
#SBATCH --job-name=mcarresults
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH -o /usr/xtmp/ham51/script-outputs/%j.out
#SBATCH -e /usr/xtmp/ham51/script-outputs/%j.err
#SBATCH --mem=100G
#SBATCH --array=0-389%50

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.introduce_mcar_folds_index --index $((SLURM_ARRAY_TASK_ID))
