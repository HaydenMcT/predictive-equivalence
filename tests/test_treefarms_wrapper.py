import numpy as np
import pandas as pd

from treefarms import TREEFARMS as tf
from models.treefarms_wrapper import construct_tree_rset

#verifies that the wrapper doesn't have issues with index shifting 
# (an old version of treefarms, without the wrapper, hit a bug with this test case)
def test_index_shift():
    data = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 1],
    ])
    data = pd.DataFrame(data, columns=["a <= 1", "b <= 1", "c <= 1", "y"], dtype='boolean')
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
    model = construct_tree_rset(X_binned, Y, config)

    assert (model[0].score(X_binned, Y) == 1)


# The below cases crash treefarms if we don't use our wrapper
# (so the test cases verify that the wrapper doesn't have this problem)
# For some reason, when all rows are the same, treefarms crashes
# This happens regardless of whether all labels are the same


def test_repeat_vals_one_label_wrapper():
    data = pd.DataFrame(
        np.array(
            [[1, 1, 1, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 0]]
        ),
        columns=["x <= 0", "a <= 1", "b <= 1", "y"],
        dtype='boolean'
    )
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
    try:
        model = construct_tree_rset(X_binned, Y, config)
    except:
        print("Error in fitting")
        assert(False)

def test_repeat_vals_dif_label_wrapper():
    data = pd.DataFrame(
        np.array(
            [[1, 1, 1, 0],
             [1, 1, 1, 1],
             [1, 1, 1, 0]]
        ),
        columns=["x <= 0", "a <= 1", "b <= 1", "y"],
        dtype='boolean'
    )
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
    try:
        model = construct_tree_rset(X_binned, Y, config)
    except:
        print("Error in fitting")
        assert(False)