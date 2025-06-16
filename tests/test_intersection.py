import numpy as np
import pandas as pd

from preprocessing.binarize import bin_by_quantile
from models.treefarms_wrapper import construct_tree_rset

TF_config = {
    "depth_budget": 5,
    "rashomon_bound_adder": 0.01,
    "rashomon_bound_multiplier": 0,
    "rashomon_bound": 0,
    "rashomon_ignore_trivial_extensions": False,
    "regularization": 0.01,
    "verbose": False
}


def test_single_label_prediction():
    data = pd.DataFrame(np.array([
        [0, 1, 1],
        [3, 2, 1],
        [5, 1, 1],
    ]), columns=["a", "b", "y"])
    X_binned, bin_name_to_original = bin_by_quantile(data.drop(columns=['y']), n_bins=2)
    y = data['y']

    tf = construct_tree_rset(X_binned, y, tf_config=TF_config)

    data = pd.DataFrame(np.array([
        [0, 1, 0],
        [3, 2, 0],
        [5, 1, 0],
    ]), columns=["a", "b", "y"])
    X_binned, bin_name_to_original = bin_by_quantile(data.drop(columns=['y']), n_bins=2)
    y = data['y']

    tf2 = construct_tree_rset(X_binned, y, tf_config=TF_config)

    #these two treefarms objects are trained on pure data with just one label, and the label is different for each,
    # so they should have no overlap for the given epsilon.
    assert len(tf.intersection(tf2)) == 0
