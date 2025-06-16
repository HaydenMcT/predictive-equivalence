#!/bin/bash
#SBATCH --job-name=dnf_vs_ittybitty
#SBATCH -c 1
#SBATCH -t 48:00:00
#SBATCH -o slurm_scripts/output/dnf_vs_ittybitty_%j.out
#SBATCH -e slurm_scripts/output/dnf_vs_ittybitty_%j.err
#SBATCH --mem=100G

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.test_dnfs_vs_ittybitty
