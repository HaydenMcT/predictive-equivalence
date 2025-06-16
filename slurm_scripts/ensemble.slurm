#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH -c 1
#SBATCH -t 48:00:00
#SBATCH -o slurm_scripts/output/ensemble_%j.out
#SBATCH -e slurm_scripts/output/ensemble_%j.err
#SBATCH --mem=100G

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.run_ensemble
