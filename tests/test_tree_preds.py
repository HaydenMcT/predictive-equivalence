import numpy as np
import pandas as pd

from models.treefarms_wrapper import create_tree_classifier


def correctly_id_nan(tree_type='DNF'):
    '''
    This tree predicts 1 if feature 0 is True, 
    otherwise it splits on feature 1 and: 
         predicts 0 if feature 1 is True, 
         or 1 otherwise.
    If Feature 0 is False and Feature 1 is missing, 
    the tree should evaluate to NA. 
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
    tree = create_tree_classifier(tree_json, tree_type=tree_type)
    X = pd.DataFrame(np.array([[0, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')
    pred = tree.predict(X)[0]
    assert pd.isna(pred)


def correctly_preds_true(tree_type='DNF'):
    '''
    This tree predicts 1 if feature 0 is True, 
    otherwise it splits on feature 1 and: 
         predicts 0 if feature 1 is True, 
         or 1 otherwise.
    If Feature 0 is True and Feature 1 is missing, 
    the tree should evaluate to True. 
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
    tree = create_tree_classifier(tree_json, tree_type=tree_type)
    X = pd.DataFrame(np.array([[1, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')

    assert (tree.predict(X)[0])

def test_correctly_id_nan():
    correctly_id_nan(tree_type='PCC')
    correctly_id_nan(tree_type='DNF')

def test_correctly_preds_true():
    correctly_preds_true(tree_type='PCC')
    correctly_preds_true(tree_type='DNF')
