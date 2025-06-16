from models.tree_classifier_wrapper import DNFTree
import pandas as pd
import numpy as np

def test_stump(): 
    stump = {'prediction': 1, 'name': 'Prediction'}
    tree = DNFTree(stump)
    X = pd.DataFrame(np.array([[1, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')
    assert (tree.predict(X)[0])