#!/bin/bash
#SBATCH --job-name=ps_rsets
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH -o slurm_scripts/output/rid_%j.out
#SBATCH -e slurm_scripts/output/rid_%j.err
#SBATCH --mem=100G

# RUN THIS FILE FROM THE BASE DIRECTORY

# run without complete case intersection
python -u -m experiments.run_rid_comparison
