from utils.tree_dict_to_sklearn import construct_sklearn_tree_from_dict
from sklearn.tree import DecisionTreeClassifier
from utils.tree_to_dict import sklearn_tree_to_dict, dl85_to_dict, gosdt_to_dict, pystreed_tree_to_dict
from gosdt import GOSDTClassifier as GOSDT
import numpy as np
from pystreed import STreeDClassifier

def test_sklearn_to_sklearn_predictive_validity():
    # Create and fit a dummy tree
    X_dummy = np.random.rand(100, 17)
    y_dummy = np.random.randint(0, 2, size=100)  
    clf = DecisionTreeClassifier()
    clf.fit(X_dummy, y_dummy)

    # Check that we can map from sklearn to a dict and back
    new_clf = construct_sklearn_tree_from_dict(sklearn_tree_to_dict(clf.tree_), clf.tree_.n_features)

    X_dummy = np.random.rand(10_000, 17)
    assert (new_clf.predict(X_dummy) == clf.predict(X_dummy)).all()

def test_gosdt_to_sklearn_predictive_validity():
    # Create and fit a dummy tree
    X_dummy = np.random.randint(0, 2, size=(100, 6))
    y_dummy = np.random.randint(0, 2, size=100)
    config = {
                "regularization": 0.01,
                "depth_budget": 5,
                "time_limit": 60,
                "similar_support": False
            }

    clf = GOSDT(**config)
    clf.fit(X_dummy, y_dummy)

    # Check that we can map from sklearn to a dict and back
    new_clf = construct_sklearn_tree_from_dict(
        gosdt_to_dict(clf.trees_[0].tree), 
        X_dummy.shape[1]
    )

    X_dummy = np.random.randint(0, 2, size=(10_000, 6))
    assert (new_clf.predict(X_dummy) == clf.predict(X_dummy)).all()

def test_pystreed_to_sklearn_predictive_validity():
    # Create and fit a dummy tree
    X_dummy = np.random.rand(100, 17) < 0.5
    y_dummy = np.random.randint(0, 2, size=100)
    clf = STreeDClassifier(max_depth=3, max_num_nodes=3)
    clf.fit(X_dummy, y_dummy)

    dict_tree = pystreed_tree_to_dict(clf.tree_)

    new_clf = construct_sklearn_tree_from_dict(dict_tree, X_dummy.shape[1])

    X_dummy = np.random.rand(10_000, 17)
    assert (new_clf.predict(X_dummy) == clf.predict(X_dummy)).all()