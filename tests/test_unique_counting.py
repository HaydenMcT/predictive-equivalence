from utils.predictive_uniqueness import get_prediction_vector, bits_to_int
from models.treefarms_wrapper import construct_tree_rset, create_tree_classifier, DEFAULT_TF_CONFIG
from preprocessing.binarize import bin_by_quantile

import pandas as pd
import numpy as np
import json


def test_pred_vec_equivalence():
    """
    Test whether our prediction vector extraction method
    always matches the preds of TreeClassifier
    """
    data = pd.DataFrame(np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
                        columns=["a", "b", "c", "y"])
    X_binned, bin_name_to_original = bin_by_quantile(data.drop(columns=['y']), n_bins=2)
    y = data['y']

    model = construct_tree_rset(X_binned, y, DEFAULT_TF_CONFIG, tree_type='IttyBitty')

    for cur_m in list(model.json_set):
        cur_m = json.loads(cur_m)
        sklearnified_model = create_tree_classifier(cur_m, X=X_binned, y=pd.DataFrame(y), tree_type='IttyBitty')
        preds_vector = get_prediction_vector(cur_m, X_binned.shape[1])

        for i in range(100):
            ftrs = np.random.randint(2, size=(1, X_binned.shape[1]))
            sklearn_pred = sklearnified_model.predict(pd.DataFrame(ftrs))
            target_ind = bits_to_int(ftrs[0])
            pred_vector_pred = preds_vector[target_ind]
            assert (sklearn_pred[0] == pred_vector_pred)


def test_treefarms_rep_unique_counting():
    """
    Test whether, for every tree in the Rashomon set, there exists
    a prediction vector that matches its predictions
    """
    data = pd.DataFrame(np.array([
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
        [1, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
                        columns=["a", "b", "c", "y"])
    X_binned = data.drop(columns=['y'])
    y = data['y']

    model = construct_tree_rset(X_binned, y, DEFAULT_TF_CONFIG, tree_type='IttyBitty')

    json_list = [json.loads(t) for t in list(model.json_set)]
    all_pred_vectors = np.stack(
        [get_prediction_vector(model_json, num_ftrs=model.X_train.shape[1]) for model_json in json_list])
    unique_vectors = np.unique(all_pred_vectors, axis=0)
    expensive_count = unique_vectors.shape[0]
    cheap_count = model.get_unique_tree_count()

    assert expensive_count == cheap_count  #, model.get_unique_prediction_vectors(use_itty_bitty_rep=True)


def test_treefarms_rep_unique_counting_fico():
    """
    Test whether, for every tree in the Rashomon set, there exists
    a prediction vector that matches its predictions
    """
    data = pd.read_csv('./data/fico_full.csv')
    X_binned, bin_name_to_original = bin_by_quantile(data.drop(columns=data.columns[10:]), n_bins=2)
    y = data['PoorRiskPerformance']

    model = construct_tree_rset(X_binned, y, DEFAULT_TF_CONFIG, tree_type='IttyBitty')

    json_list = [json.loads(t) for t in list(model.json_set)]
    all_pred_vectors = np.stack(
        [get_prediction_vector(model_json, num_ftrs=model.X_train.shape[1]) for model_json in json_list])
    unique_vectors = np.unique(all_pred_vectors, axis=0)
    expensive_count = unique_vectors.shape[0]
    cheap_count = model.get_unique_tree_count()

    assert expensive_count == cheap_count
