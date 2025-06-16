from preprocessing.data import prepare_binarized_data_kfolds, load_data, nan_preserving_cut
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


def dl85_to_dict(node):
    '''
    Convert a dl85 tree to a dictionary representation.
    This dictionary representation allows constructing
    tree classifier wrappers from the 
    tree_classifier_wrapper.py file.
    '''
    dict_tree = {}
    ####################
    #features needed for dnf_trees:
    dict_tree['reference'] = 'true'  #queries assumed to always correspond to whether a feature is True;
    # since we're expecting to be dealing with binarized data always with
    # thresholds of 0, where left/true corresponds to > 0, this should always hold
    dict_tree['relation'] = '=='
    ####################
    if 'value' in node:
        dict_tree['prediction'] = node['value']
        return dict_tree
    else:
        dict_tree['feature'] = node['feat']
        dict_tree['true'] = dl85_to_dict(node['left'])
        dict_tree['false'] = dl85_to_dict(node['right'])
        return dict_tree


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

# RSET_NAN_HANDLING_METHOD = {'ours': 'DNF', 'path-based': 'Default', 'used-features': 'Conservative'}
RSET_NAN_HANDLING_METHOD = {'path-based': 'Default'}


def experiment(dataset='fico_full.csv',
               nan_handling='ours',
               base_tree='gosdt',
               balance=False,
               prebin_missingness=False,
               seed=0):

    X, y = load_data(dataset, balance=balance)
    binarization_config = CONFIGS[dataset]

    folds, binarization_info = prepare_binarized_data_kfolds(X, y, **binarization_config)
    result_rows = []
    for i in range(5):
        result_row = {'dataset': dataset, 'base_tree': base_tree, 'NaN_condition': nan_handling}
        result_row['balanced'] = balance
        result_row['prebin_missingness'] = prebin_missingness
        X_train, y_train, X_test, y_test = folds[i]
        X_train_pre_bin, X_test_pre_bin, threshold, header = binarization_info[i]

        # rename columns to be integers
        X_train.columns = range(X_train.shape[1])
        X_test.columns = range(X_test.shape[1])

        # Train model:
        misstree = construct_tree_rset(X_train,
                                       y_train,
                                       tf_config=tf_config,
                                       tree_type=RSET_NAN_HANDLING_METHOD[nan_handling])
        if nan_handling == 'ours':
            result_row['rset_size'] = misstree.get_unique_tree_count()
        else:
            result_row['rset_size'] = misstree.get_tree_count()
        predictions_complete_train = misstree.predict(X_train).values
        # predictions_complete_test = misstree.predict(X_test).values
        predictions_complete_test, sparsities_complete_test = misstree.predict(X_test,
                                                                               'best_non_nan',
                                                                               unique_only=(nan_handling == 'ours'),
                                                                               extra_info=True)
        result_row['acc_complete_train'] = np.mean(predictions_complete_train == y_train)
        result_row['acc_complete_test'] = np.mean(
            predictions_complete_test.values ==
            y_test.values)
        result_row['avg_sparsity_complete'] = sparsities_complete_test.mean()

        stable_X_test, stable_X_test_pre_bin = X_test.copy(), X_test_pre_bin.copy()

        # introduce mcar missingness into test set:
        for miss_prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            result_row['miss_prob'] = miss_prob
            np.random.seed(seed)
            if prebin_missingness:
                X_test_pre_bin = stable_X_test_pre_bin.copy()
                X_test_pre_bin[np.random.choice([True, False], size=X_test_pre_bin.shape, p=[miss_prob,
                                                                                             1 - miss_prob])] = pd.NA
                X_test = nan_preserving_cut(X_test_pre_bin, threshold)[header].astype('boolean')
            else:
                X_test = stable_X_test.copy()
                X_test[np.random.choice([True, False], size=X_test.shape, p=[miss_prob, 1 - miss_prob])] = pd.NA

            predictions_inc, sparsities_inc = misstree.predict(X_test,
                                                               'best_non_nan',
                                                               unique_only=(nan_handling == 'ours'),
                                                               extra_info=True)
            result_row['test_set_size'] = X_test.shape[0]
            result_row['n_nan_preds'] = predictions_inc.isna().values.sum()

            mask = predictions_inc.isna()
            assert (sparsities_inc.isna() == predictions_inc.isna()).all()
            correctness = (predictions_inc.values == y_test.values)[~mask]
            if (~mask).sum() == 0:
                result_row['acc_incomplete_when_not_nan'] = np.nan
                result_row['acc_complete_on_points_where_incomplete_not_nan'] = np.nan
                result_row['avg_sparsity_incomplete'] = np.nan
            else:
                result_row['acc_incomplete_when_not_nan'] = correctness.sum() / (~mask).sum()
                result_row['acc_complete_on_points_where_incomplete_not_nan'] = (
                    predictions_complete_test.values == y_test.values)[~mask].sum() / (~mask).sum()
                result_row['avg_sparsity_incomplete'] = sparsities_inc.mean()
            # now impute for the incomplete data
            fill_val = 0 if not prebin_missingness else False
            predictions_inc = misstree.predict(X_test.fillna(fill_val))  #todo add mean imputation, etc
            correctness = (predictions_inc.values == y_test.values)[mask]
            if mask.sum() == 0:
                result_row['acc_when_nan_if_0_impute'] = np.nan
            else:
                result_row['acc_when_nan_if_0_impute'] = correctness.sum() / (mask).sum()

            result_rows.append(result_row.copy())
    return result_rows


def experiment_loop(index_to_run, rerun):
    rows = []
    index = 0
    for prebin in [False, True]:
        for balance in [True, False]:
            for dataset in CONFIGS.keys():
                for nan_handling in RSET_NAN_HANDLING_METHOD.keys():
                    for base_tree in ['rset']:
                        if index == index_to_run:
                            if not rerun:
                                try:
                                    pd.read_csv(f'experiments/mcar_results/rset/{index_to_run}.csv')
                                    return  # if successfully read data without error, don't rerun
                                except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                                    pass
                            print('running', dataset, nan_handling, base_tree, balance, prebin)
                            rows += experiment(dataset, nan_handling, base_tree, balance, prebin)
                            df = pd.DataFrame(rows)
                            df.to_csv(f'experiments/mcar_results/rset/{index_to_run}.csv', index=False)
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
