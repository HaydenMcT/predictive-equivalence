# Leveraging Predictive Equivalence in Decision Trees

This repository hosts code to replicate the experiments in the ICML 2025 Paper [Leveraging Predictive Equivalence in Decision Trees](https://arxiv.org/abs/2506.14143).

## Environment
To use the environment specified in `requirements.txt`, run the following commands:
- `python3 -m venv ~/VENVS/predeq`: this creates a virtual environment
- `source ~/VENVS/predeq/bin/activate`: this activates the virtual environment
- `pip install -r requirements.txt`: run this from the base of the repository
Before running any slurm script, make sure to activate your environment.

Code is written in Python module form.
Run files, e.g. `preprocessing/binarize.py`, with
`python3 -m preprocessing.binarize`.
Note the lack of a ".py" at the end.
All unit tests can be run by calling
`python3 -m pytest tests`.

## Replication Guide
### General Structure
The code in this repository is generally split up
according to the sections in the paper, and is written
to run on a slurm batching system. Experiments are launched
using the various scripts in the `slurm_scripts` directory
using the command `sbatch slurm_scripts/<target_script>`.

To run our experiments on a non-slurm system, our slurm scripts
can be converted to bash commands as follows. If the .slurm file
contains a line like this: `#SBATCH --array=0-23`,
copy the last line -- e.g., from `slurm_scripts/rid_test.slurm`,
grab the line `python -u -m experiments.run_rid_comparison $SLURM_ARRAY_TASK_ID`
and run it in a bash for loop to match the specified slurm array,
i.e.,
```
for SLURM_ARRAY_TASK_ID in $(seq 0 23);
do
    python -u -m experiments.run_rid_comparison $SLURM_ARRAY_TASK_ID
done
```

If the script does not include any array reasoning, simply directly
run the python call.

### Replicating Specific Sections
- The experiments for Section 4 are run using `slurm_scripts/index_mcar_rset_count.slurm`,
and aggregated using `experiments/mcar_results/rset_count/get_sizes.ipynb`
- The experiments and plotting for Section 5.1 are
run using the notebook `notebooks/doppleganger_purity_vi.ipynb`
- The experiments for Section 5.2 are run using `slurm_scripts/rid_test.slurm`
and plots are generated using `notebooks/parse_rid_results.ipynb`
- The single tree experiments for Section 6 are run using `slurm_scripts/mcar_folds_index.slurm`,
aggregated using `experiments/mcar_results/single_tree_rerun/merge_individual_results.ipynb`,
and plotted using `experiments/mcar_results/aggregate_by_fold_table.ipynb`.
- The Rashomon set experiments for Section 6 are run using `slurm_scripts/index_mcar_rset.slurm`,
aggregated using `experiments/mcar_results/rset/merge.py`,
and plotted using `experiments/mcar_results/accuracy_vs_coverage.ipynb`.
- The experiments for Section 7 are run using `slurm_scripts/cost_sensitive.slurm`
and plots are generated using `notebooks/cost_sensitive_plots.ipynb`
