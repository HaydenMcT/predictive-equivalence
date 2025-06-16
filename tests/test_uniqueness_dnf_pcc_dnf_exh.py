import pandas as pd
import numpy as np
from models.treefarms_wrapper import TreeFarmsWrapper
from models.tree_classifier_wrapper import TreeClassifierWrapper
from preprocessing.data import prepare_binarized_data, load_data
import json

binarization_config = {
    "missing_values": [-7, -8, -9],
    "columns_to_drop": ['ExternalRiskEstimate'],
    "num_quantiles": 0,
    "gosdt_guesses": True,
    "keep_missingness_after_binarization": True
}

def compare_uniqueness(tf_config, X, y):
    tf = TreeFarmsWrapper(tf_config, tree_type='DNF')
    tf_pcc = TreeFarmsWrapper(tf_config, tree_type='PCC')
    tf_dnf_exh = TreeFarmsWrapper(tf_config, tree_type='DNF_exhaustive')
    tf.fit(X.fillna(0), y)
    tf_pcc.fit(X.fillna(0), y)
    tf_dnf_exh.fit(X.fillna(0), y)

    assert(tf.get_tree_count() == tf_pcc.get_tree_count())
    assert(tf.get_unique_tree_count() == tf_pcc.get_unique_tree_count())
    assert(tf.get_tree_count() == tf_dnf_exh.get_tree_count())

    tf_preds = tf.predict(X)
    tf_pcc_preds = tf_pcc.predict(X)
    tf_dnf_exh_preds = tf_dnf_exh.predict(X)
    assert(np.all(tf_preds == tf_pcc_preds))
    assert(np.all(tf_preds.isna() == tf_pcc_preds.isna()))
    assert(np.all(tf_preds == tf_dnf_exh_preds))
    assert(np.all(tf_preds.isna() == tf_dnf_exh_preds.isna()))

X_orig, y_orig = load_data('fico_full.csv', y_name = 'PoorRiskPerformance')
X, y, _, _, _ = prepare_binarized_data(X_orig, y_orig, **binarization_config)

tf_config = {
        "depth_budget": 3,
        "rashomon_bound_adder": 0.01,
        "rashomon_bound_multiplier": 0,
        "rashomon_bound": 0,
        "rashomon_ignore_trivial_extensions": False,
        "regularization": 0.02,
        "verbose": False
    }

def test_uniqueness_simple_config():
    compare_uniqueness(tf_config, X, y)

def test_uniqueness_harder_config():
    tf_config["depth_budget"] = 4
    tf_config["rashomon_ignore_trivial_extensions"] = True
    compare_uniqueness(tf_config, X, y)

def test_uniqueness_hardest_config():
    config = {
        "depth_budget": 4,
        "rashomon_bound_adder": 0.01,
        "rashomon_bound_multiplier": 0,
        "rashomon_bound": 0,
        "rashomon_ignore_trivial_extensions": True,
        "regularization": 0.01,
        "verbose": False
    }
    compare_uniqueness(config, X, y)