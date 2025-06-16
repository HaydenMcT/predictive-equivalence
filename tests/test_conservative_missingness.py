from models.tree_classifier_wrapper import ConservativeTree
import numpy as np
import pandas as pd

def test_conservative_missingness():
    '''
    This tree predicts 1 if feature 0 is True, 
    or 0 otherwise.
    But the conservative method will avoid predicting if any missing variable 
    is used in the tree, even if it is irrelevant to the prediction. 
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
            "feature": 1, 
            "relation": "==",
            "reference": "true",
            "true": {
                "prediction": 0,
                "name": "Prediction"
            }, 
            "false": {
                "prediction": 1,
                "name": "Prediction"
            }
        }
    }
    X = pd.DataFrame(np.array([[1, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')
    tree = ConservativeTree(tree_json, X=X)
    assert (tree.predict(X).isna()[0])
    assert (tree.predict(pd.DataFrame(np.array([[1, 0]]), columns=["feature_0", "feature_1"], dtype='boolean'))[0])
    assert (tree.predict(pd.DataFrame(np.array([[1, 1]]), columns=["feature_0", "feature_1"], dtype='boolean'))[0])

def test_basic(): 
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
    X = pd.DataFrame(np.array([[1, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')
    tree = ConservativeTree(tree_json, X=X)
    assert (tree.predict(X)[0])