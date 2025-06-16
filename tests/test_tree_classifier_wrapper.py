import pandas as pd
import numpy as np
import json
import time
from treefarms.model.tree_classifier import TreeClassifier
from models.treefarms_wrapper import construct_tree_rset, TreeClassifierWrapper


def json_set_to_model_list_w_classifier_choice(json_set: set,
                                               X_train: pd.DataFrame = None,
                                               y_train: pd.DataFrame = None,
                                               classifier=TreeClassifier) -> list:
    '''
    Converts a set of JSON representations of TreeClassifiers
    to a list of TreeClassifiers that may be easily iterated over
    Args:
        json_set : set -- a set of trees represented as json strings
        X_train : pd.DataFrame -- the X values used to fit these trees.
            If provided, these are used to populate the loss attribute of
            each tree
        y_train : pd.DataFrame -- the y values used to fit these trees.
            If provided, these are used to populate the loss attribute of
            each tree
    '''
    json_list = list(json_set)
    model_list = []
    if X_train is not None and y_train is not None:
        model_list = [classifier(json.loads(cur_dict), X=X_train, y=pd.DataFrame(y_train)) for cur_dict in json_list]
    else:
        model_list = [classifier(json.loads(cur_dict)) for cur_dict in json_list]
    return model_list


def test_tree_wrapper_speed():
    data = pd.DataFrame(np.concatenate([np.array([[0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 0, 1]],),
                                        np.ones((10_000, 4))],
                                       axis=0),
                        columns=["x <= 0", "a <= 1", "b <= 1", "y"],
                        dtype='boolean')
    X_binned = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    config = {
        "depth_budget": 2,
        "rashomon_bound_adder": 0,
        "rashomon_bound_multiplier": 0,
        "rashomon_bound": 0,
        "rashomon_ignore_trivial_extensions": False,
        "regularization": 0.05,
        "verbose": False
    }
    print("About to fit")
    tf_model = construct_tree_rset(X_binned, Y, config)

    slow_trees = json_set_to_model_list_w_classifier_choice(tf_model.json_set, tf_model.X_train, tf_model.y_train,
                                                            TreeClassifier)

    fast_trees = json_set_to_model_list_w_classifier_choice(tf_model.json_set, tf_model.X_train, tf_model.y_train,
                                                            TreeClassifierWrapper)

    for i in range(len(slow_trees)):
        start = time.time()
        preds_slow = slow_trees[i].predict(tf_model.X_train)
        pred_time_slow = time.time() - start

        start = time.time()
        preds_fast = fast_trees[i].predict(tf_model.X_train)
        pred_time_fast = time.time() - start

        assert (preds_slow == preds_fast).all()
        assert pred_time_fast <= pred_time_slow
