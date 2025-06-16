from rid.rashomon_importance_distribution import RashomonImportanceDistribution as RID
from models.treefarms_wrapper import DEFAULT_TF_CONFIG
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from preprocessing.data import nansafe_cut
from utils.model_reliance import get_mr
import os
import sys
from scipy.stats import wasserstein_distance
from scipy import stats

def run_rid_comparison(df, dataset_name, tf_config, binning_map=None, n_resamples=10, internal_bin=False, true_mr_df=None):
    if binning_map is None:
        binning_map = {i : c for i, c in enumerate(df.columns[:-1])}
    print(tf_config)
    rid = RID(
        df, 
        n_resamples=n_resamples, 
        tf_config=tf_config, 
        binning_map=binning_map if not internal_bin else None, 
        binning_fn=lambda x: x, 
        use_test_df=False,
        tree_type="DNF",
        dataset_name=dataset_name,
        allow_binarize_internally=internal_bin,
        verbose=True
    )
    rid_unique = RID(
        df, 
        n_resamples=n_resamples, 
        tf_config=tf_config, 
        binning_map=binning_map if not internal_bin else None,
        binning_fn=lambda x: x, 
        use_test_df=False,
        unique_trees_only=True,
        tree_type="DNF",
        dataset_name=dataset_name,
        allow_binarize_internally=internal_bin,
        verbose=True
    )

    vi_names = []
    vi_estimates = []
    sns.set(font_scale=2.2)
    sns.set_style("white")
    summary_df = {
        "var": [],
        "distance": [],
        "original_rid_max": [],
        "corrected_rid_max": [],
        "ks_test_stat": [],
        "ks_test_p": [],
        "ks_test_loc": [],
        "ks_test_sign": [],
        "pe_rates": []
    }
    all_dupe_rates = []
    for tf in rid_unique.rashomon_sets:
        all_dupe_rates = all_dupe_rates + [v for v in tf.get_duplication_rates().values()]
    print(f"Duplication rates: {all_dupe_rates}")

    for i, v in enumerate(df.columns[:-1]):
        print(f"Plotting for var {v}")
        vi_estimates = list(rid.rid_with_counts[i])
        vi_names = [f"Original RID"] * len(list(rid.rid_with_counts[i]))
        if true_mr_df is not None:
            dist = (abs(true_mr_df.loc[v, 'mr'] - rid.rid_with_counts[i])).mean()
            print(f"Original dist to gt for {v}: {dist}")

        vi_estimates = vi_estimates + list(rid_unique.rid_with_counts[i])
        vi_names = vi_names + [f"PE Corrected RID"] * len(list(rid_unique.rid_with_counts[i]))
        if true_mr_df is not None:
            dist = (abs(true_mr_df.loc[v, 'mr'] - rid_unique.rid_with_counts[i])).mean()
            print(f"Corrected dist to gt for {v}: {dist}")

        dist = wasserstein_distance(rid_unique.rid_with_counts[i], rid.rid_with_counts[i])

        results = pd.DataFrame({
            "variable": vi_names,
            "Model Reliance Value": vi_estimates,
        })

        plt.figure(figsize=(10, 8))
        ax = sns.histplot(
            data=results, 
            x="Model Reliance Value", 
            hue="variable", 
            common_norm=False, 
            bins=20,
            alpha=0.5, multiple='dodge',
            fill=True, stat="percent")
        ax.legend_.set_title(None)
        ax.set_title(f'RID Distribution of {v} Importance')

        
        os.makedirs(f'rid_plots_1_22/{dataset_name}/', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'rid_plots_1_22/{dataset_name}/rid_comparison_{v}_alt.pdf')
        plt.clf()

        ks_test = stats.kstest(rid_unique.rid_with_counts[i], rid.rid_with_counts[i])

        summary_df["var"] = summary_df["var"] + [v]
        summary_df["distance"] = summary_df["distance"] + [dist]
        summary_df["original_rid_max"] = summary_df["original_rid_max"] + [rid.range(i)[1]]
        summary_df["corrected_rid_max"] = summary_df["corrected_rid_max"] + [rid_unique.range(i)[1]]
        summary_df["ks_test_stat"] = summary_df["ks_test_stat"] + [ks_test.statistic]
        summary_df["ks_test_p"] = summary_df["ks_test_p"] + [ks_test.pvalue]
        summary_df["ks_test_loc"] = summary_df["ks_test_loc"] + [ks_test.statistic_location]
        summary_df["ks_test_sign"] = summary_df["ks_test_sign"] + [ks_test.statistic_sign]
        summary_df["pe_rates"] = summary_df["pe_rates"] + [all_dupe_rates]

        pd.DataFrame(summary_df).to_csv(f"rid_plots_1_22/{dataset_name}/summary.csv", index=False)


if __name__ == '__main__':
    np.random.seed(0)
    tf_config = DEFAULT_TF_CONFIG
    tf_config['depth_budget'] = 4
    tf_config['rashomon_ignore_trivial_extensions'] = True
    tf_config['regularization'] = 0.02
    tf_config['rashomon_bound_adder'] = 0.02

    n_resamples = 100
    num_extras = 0
    num_samples = 1000
    x1 = np.random.binomial(1, p=0.5**0.5, size=(num_samples))
    x2 = np.random.binomial(1, p=0.5**0.5, size=(num_samples))
    noise_for_x3 = np.random.binomial(1, p=0.01, size=(num_samples))
    x3 = x1 * x2
    x3 = x3.astype(bool)
    x3[noise_for_x3 == 1] = ~x3[noise_for_x3 == 1]
    x3 = x3.astype(int)
    # x3 = np.random.binomial(1, p=0.5**0.5, size=(num_samples))
    extra_xs = []
    for i in range(num_extras):
        noise_for_x4 = np.random.binomial(1, p=0.001, size=(num_samples))
        x4 = x2 * x3
        x4 = x4.astype(bool)
        x4[noise_for_x4 == 1] = ~x4[noise_for_x4 == 1]
        x4 = x4.astype(int)
        extra_xs.append(x4)

    base_weight = 0.01
    base_prob = 0.5

    prob_vector = x3 #np.logical_xor(x2 * x3, x1) 
    true_model = lambda df_values: df_values[:, 2]#np.logical_xor(df_values[:, 0], df_values[:, 1] * df_values[:, 2])

    y = np.random.binomial(
        1,
        p=base_weight * base_prob + (1 - base_weight) * prob_vector, 
        size=(num_samples)
    )

    vars_dict = {
        "X1": x1,
        "X2": x2,
        "X3": x3
    }
    for i, v in enumerate(extra_xs):
        vars_dict[f"x{i+4}"] = v

    vars_dict["y"] = y

    dummy_data = pd.DataFrame(vars_dict)
    true_mr_df = get_mr(dummy_data, true_model, n_iters=100)

    run_rid_comparison(
        dummy_data, 
        dataset_name=f"dummy_data_alt_mini_triv_ext_{tf_config['rashomon_ignore_trivial_extensions']}_db{tf_config['depth_budget']}_reg_{tf_config['regularization']}_rashomon_bound_adder_{tf_config['rashomon_bound_adder']}_{n_resamples}_bootstraps", 
        n_resamples=n_resamples, 
        true_mr_df=true_mr_df,
        tf_config=tf_config
    )

    possible_dfs = [
        'tic-tac-toe.csv', "wisconsin.csv", "coupon_full.csv", "compas_complete.csv", "wine_quality.csv",
        "broward_general_2y.csv", 'fico_complete.csv', 'iris_setosa.csv', 'iris_versicolor.csv',
        'iris_virginica.csv', 'spiral.csv', 'netherlands.csv'
    ]
    # for df in possible_dfs:
    slurm_id = int(sys.argv[1])
    df = possible_dfs[slurm_id]

    # for reg in [0.04, 0.02, 0.01, 0.005]:
    # tf_config['regularization'] = reg
    df_name = f"{df.split('.')[0]}_{n_resamples}_resamples_triv_ext_{tf_config['rashomon_ignore_trivial_extensions']}_db{tf_config['depth_budget']}_reg_{tf_config['regularization']}_rashomon_bound_adder_{tf_config['rashomon_bound_adder']}"
    original_df = pd.read_csv(f"data/{df}")
    original_df = original_df.dropna(axis=0).reset_index(drop=True)
    # X = original_df.iloc[:, :-1]
    # y = original_df.iloc[:, -1]

    run_rid_comparison(
        original_df, 
        dataset_name=df_name, 
        n_resamples=n_resamples, 
        internal_bin=not ('tic-tac-toe' in df or 'monk' in df), 
        tf_config=tf_config
    )
