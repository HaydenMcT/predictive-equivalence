from preprocessing.data import prepare_binarized_data_kfolds, load_data, nan_preserving_cut
from models.tree_classifier_wrapper import DNFTree, TreeClassifierWrapper, ConservativeTree
from sklearn.tree import DecisionTreeClassifier
from utils.tree_to_dict import sklearn_tree_to_dict, dl85_to_dict, gosdt_to_dict
import numpy as np
import pandas as pd
from gosdt import GOSDTClassifier
from pydl85 import DL85Classifier

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
    'higgs.csv': DEFAULT_CONFIG,
}

NAN_HANDLING_METHOD = {'ours': DNFTree, 'path-based': TreeClassifierWrapper, 'used-features': ConservativeTree}

# map from tree description to a tuple containing
# the tree classifier and a method to extract the dict of the tree
BASE_TREES = {
    'gosdt': (GOSDTClassifier(regularization=0.01, depth_budget=4,
                              verbose=False), lambda tree: gosdt_to_dict(tree.trees_[0].tree)),
    'sklearn': (DecisionTreeClassifier(max_depth=3), lambda tree: sklearn_tree_to_dict(tree.tree_)),
    'dl85': (DL85Classifier(max_depth=3), lambda tree: dl85_to_dict(tree.tree_)),
    'sklearn-4': (DecisionTreeClassifier(max_depth=4), lambda tree: sklearn_tree_to_dict(tree.tree_)),
    'sklearn-5': (DecisionTreeClassifier(max_depth=5), lambda tree: sklearn_tree_to_dict(tree.tree_)),
}


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
        result_row = {
            'dataset': dataset,
            'base_tree': base_tree,
            'NaN_condition': nan_handling,
            'balance': balance,
            'prebin_missingness': prebin_missingness,
        }
        X_train, y_train, X_test, y_test = folds[i]
        X_train_pre_bin, X_test_pre_bin, threshold, header = binarization_info[i]

        # rename columns to be integers
        X_train.columns = range(X_train.shape[1])
        X_test.columns = range(X_test.shape[1])

        # Train model:
        (tree, postprocessor) = BASE_TREES[base_tree]  #(postprocessor extracts the dict rep of the tree model)
        tree.fit(X_train, y_train)
        if nan_handling != 'used-features':
            misstree = NAN_HANDLING_METHOD[nan_handling](postprocessor(tree))
        else:
            misstree = NAN_HANDLING_METHOD[nan_handling](postprocessor(tree), X=X_train)

        predictions_complete_train = misstree.predict(X_train).values
        predictions_complete_test = misstree.predict(X_test).values
        result_row['acc_complete_train'] = np.mean(predictions_complete_train == y_train)
        result_row['acc_complete_test'] = np.mean(predictions_complete_test == y_test)

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

            predictions_inc = misstree.predict(X_test)
            result_row['test_set_size'] = X_test.shape[0]
            result_row['n_nan_preds'] = predictions_inc.isna().values.sum()

            mask = predictions_inc.isna()
            correctness = (predictions_inc.values == y_test.values)[~mask]
            if (~mask).sum() == 0:
                result_row['acc_incomplete_when_not_nan'] = np.nan
            else:
                result_row['acc_incomplete_when_not_nan'] = correctness.sum() / (~mask).sum()
            # now impute for the incomplete data
            fill_val = 0 if not prebin_missingness else False
            predictions_inc = misstree.predict(X_test.fillna(fill_val))  #todo add mean imputation, etc
            correctness = (predictions_inc.values == y_test.values)[mask]
            if mask.sum() == 0:
                result_row['acc_when_nan_if_0_impute'] = np.nan
            else:
                result_row['acc_when_nan_if_0_impute'] = correctness.sum() / (mask).sum()

            # When we could have predicted without imputation, we should perform the same when instead we used imputation:
            correctness_subset = (predictions_inc.values == y_test.values)[~mask]
            if (~mask).sum() > 0:
                assert (result_row['acc_incomplete_when_not_nan'] == correctness_subset.sum() / (~mask).sum())
            result_rows.append(result_row.copy())
    return result_rows


def experiment_runner(index_to_run, rerun):
    if not rerun:
        print("rerun checking not implemented; behaving as though rerun is true")

    index = 0
    rows = []
    for prebin_missingness in [False, True]:
        for balance in [False]:
            for dataset in CONFIGS.keys():
                for nan_handling in NAN_HANDLING_METHOD.keys():
                    for base_tree in BASE_TREES.keys():
                        if index == index_to_run:
                            print('running', dataset, nan_handling, base_tree, balance, prebin_missingness)
                            rows = experiment(dataset, nan_handling, base_tree, balance, prebin_missingness)
                            df = pd.DataFrame(rows)
                            df.to_csv(f'experiments/mcar_results/single_tree/{index}.csv', index=False)
                            return
                        index += 1


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    index_to_run = args.index
    rerun = args.rerun
    if not rerun:
        try:
            pd.read_csv(f'experiments/mcar_results/single_tree/{index_to_run}.csv')
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            experiment_runner(index_to_run, rerun)
    else:
        experiment_runner(index_to_run, rerun)
