import numpy as np
import pandas as pd

from models.treefarms_wrapper import create_tree_classifier


def test_if_correctly_pred_true():
    '''
    This tree predicts 1 if feature 0 is True, 
         or 0 otherwise.
    '''

    tree_json = {
        "feature": 0,
        "relation": "==",
        "reference": "true",
        "true": {
            "prediction": 1,
            "name": "Prediction"
        },
        "false": {
            "prediction": 0,
            "name": "Prediction"
        }
    }
    tree = create_tree_classifier(tree_json, tree_type='DNF')
    X = pd.DataFrame(np.array([[1, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')

    assert (tree.predict(X)[0])
