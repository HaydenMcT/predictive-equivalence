import numpy as np
import pandas as pd

from preprocessing.binarize import bin_by_quantile
from models.treefarms_wrapper import construct_tree_rset


def test_single_label_prediction():
    data = pd.DataFrame(np.array([
        [0, 1, 1],
        [3, 2, 1],
        [5, 1, 1],
    ]), columns=["a", "b", "y"])
    X_binned, bin_name_to_original = bin_by_quantile(data.drop(columns=['y']), n_bins=2)
    y = data['y']

    tf = construct_tree_rset(X_binned, y)

    pred = tf.predict_for_all_trees(X_binned)
    assert np.all(pred.values == 1), f"\n\n Preds: {pred}\n\n"

    data = pd.DataFrame(np.array([
        [0, 1, 0],
        [3, 2, 0],
        [5, 1, 0],
    ]), columns=["a", "b", "y"])
    X_binned, bin_name_to_original = bin_by_quantile(data.iloc[:, :-1], n_bins=2)
    y = data.iloc[:, -1]

    tf = construct_tree_rset(X_binned, y)

    pred = tf.predict_for_all_trees(X_binned)
    assert np.all(pred.values == 0), f"\n\n Preds: {pred}\n\n"
