from models.treefarms_wrapper import TreeFarmsWrapper, construct_tree_rset
import pandas as pd
import numpy as np


def test_basic():
    X = pd.DataFrame(np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]]),
                     columns=["x <= 0", "a <= 1", "b <= 1"],
                     dtype='boolean')
    Y = pd.Series([0, 1, 0])

    config = {
        "depth_budget": 2,
        "rashomon_bound_adder": 0,
        "rashomon_bound_multiplier": 0,
        "rashomon_bound": 0,
        "rashomon_ignore_trivial_extensions": True,
        "regularization": 0.05,
        "verbose": False
    }

    model = construct_tree_rset(X, Y, config)

    assert model is not None

    assert (model.predict(X, 'best_non_nan', unique_only=False) == Y).all()

    mask = np.random.rand(*X.shape)
    X[mask < 0.5] = pd.NA

    model.predict(X, 'best_non_nan', unique_only=False)
    assert (np.logical_or(model.predict(X, 'best_non_nan', unique_only=False) == Y, model.predict(X, 'best_non_nan', unique_only=False).isna())).all()

def test_extra_info(): 
    X = pd.DataFrame(np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]]),
                     columns=["x <= 0", "a <= 1", "b <= 1"],
                     dtype='boolean')
    Y = pd.Series([0, 1, 0])

    config = {
        "depth_budget": 2,
        "rashomon_bound_adder": 0,
        "rashomon_bound_multiplier": 0,
        "rashomon_bound": 0,
        "rashomon_ignore_trivial_extensions": True,
        "regularization": 0.05,
        "verbose": False
    }

    model = construct_tree_rset(X, Y, config)

    assert model is not None

    preds, sparsities = model.predict(X, 'best_non_nan', unique_only=False, extra_info=True)
    assert (preds == Y).all()
    assert (sparsities.mean() > 0)

    mask = np.random.rand(*X.shape)
    X[mask < 0.5] = pd.NA

    preds, sparsities = model.predict(X, 'best_non_nan', unique_only=False, extra_info=True)
    assert (np.logical_or(preds == Y, preds.isna())).all()
