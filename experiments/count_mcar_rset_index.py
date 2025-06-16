from preprocessing.data import prepare_binarized_data_kfolds, load_data
from models.tree_classifier_wrapper import DNFTree, TreeClassifierWrapper, ConservativeTree
from models.treefarms_wrapper import TreeFarmsWrapper, construct_tree_rset
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

tf_config = {
    "depth_budget": 4,
    "rashomon_bound_adder": 0.02,
    "rashomon_ignore_trivial_extensions": True,
    "regularization": 0.01,
    "verbose": False
}

DEFAULT_CONFIG = {"num_quantiles": 0, "gosdt_guesses": True, "keep_missingness_after_binarization": True}
CONFIGS = {
    'compas_complete.csv': DEFAULT_CONFIG,
    'wine_quality.csv': DEFAULT_CONFIG,
    'wisconsin.csv': DEFAULT_CONFIG,
    'coupon_full.csv': DEFAULT_CONFIG,
    'fico_complete.csv': DEFAULT_CONFIG,
    'netherlands.csv': DEFAULT_CONFIG,
    'spiral.csv': DEFAULT_CONFIG,
    'tic-tac-toe.csv': DEFAULT_CONFIG,
    'iris_virginica.csv': DEFAULT_CONFIG,
    'iris_versicolor.csv': DEFAULT_CONFIG,
    'iris_setosa.csv': DEFAULT_CONFIG,
    'broward_general_2y.csv': DEFAULT_CONFIG,
}

NAN_HANDLING_METHOD = {'ours': DNFTree, 'path-based': TreeClassifierWrapper, 'used-features': ConservativeTree}

RSET_NAN_HANDLING_METHOD = {'ours': 'DNF', 'path-based': 'Default', 'used-features': 'Conservative'}


def experiment(dataset='fico_full.csv', nan_handling='ours', base_tree='gosdt', balance=False, seed=0):

    X, y = load_data(dataset, balance=balance)
    binarization_config = CONFIGS[dataset]

    folds, binarization_info = prepare_binarized_data_kfolds(X, y, **binarization_config)
    result_rows = []
    for i in range(5):
        result_row = {'dataset': dataset, 'base_tree': base_tree, 'NaN_condition': nan_handling}
        result_row['balanced'] = balance
        result_row['lambda'] = tf_config['regularization']
        result_row['depth'] = tf_config['depth_budget']
        X_train, y_train, X_test, y_test = folds[i]

        # rename columns to be integers
        X_train.columns = range(X_train.shape[1])
        X_test.columns = range(X_test.shape[1])

        # Train model:
        misstree = construct_tree_rset(X_train,
                                       y_train,
                                       tf_config=tf_config,
                                       tree_type=RSET_NAN_HANDLING_METHOD[nan_handling])
        result_row['rset_size_unique'] = misstree.get_unique_tree_count()
        result_row['rset_size'] = misstree.get_tree_count()
        alt_config = tf_config.copy()
        alt_config["rashomon_ignore_trivial_extensions"] = False
        misstree = construct_tree_rset(X_train,
                                       y_train,
                                       tf_config=alt_config,
                                       tree_type=RSET_NAN_HANDLING_METHOD[nan_handling])
        result_row['rset_size_no_trivial_extensions'] = misstree.get_tree_count()
        result_row['rset_size_unique_no_trivial_extensions'] = misstree.get_unique_tree_count()

        result_rows.append(result_row.copy())
    return result_rows


def experiment_loop(index_to_run, rerun):
    rows = []
    index = 0
    for balance in [True, False]:
        for dataset in CONFIGS.keys():
            for nan_handling in ['ours']:
                for base_tree in ['rset']:
                    if index == index_to_run:
                        if not rerun:
                            try:
                                pd.read_csv(f'experiments/mcar_results/rset_count/{index_to_run}.csv')
                                return  # if successfully read data without error, don't rerun
                            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                                pass
                        print('running', dataset, nan_handling, base_tree, balance)
                        rows += experiment(dataset, nan_handling, base_tree, balance)
                        df = pd.DataFrame(rows)
                        df.to_csv(f'experiments/mcar_results/rset_count/{index_to_run}.csv', index=False)
                    index += 1


import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--index', type=int, default=0)
    argparser.add_argument('--rerun', action='store_true', default=False)
    args = argparser.parse_args()
    index_to_run = args.index
    rerun = args.rerun
    experiment_loop(index_to_run, rerun)
