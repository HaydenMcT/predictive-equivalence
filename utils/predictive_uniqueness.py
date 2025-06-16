import numpy as np
import pandas as pd

def get_used_features_recursive(cur_model: dict, features_so_far: set):
    """
    Gets a set containing each feature used in this tree.
    Args:
        cur_model : dict -- The json representation of a (sub)tree
        features_so_for : set -- A running set containing the features
            we've looked at so far
    """
    if "prediction" in cur_model:
        return features_so_far

    cur_ftr = cur_model["feature"]

    # Call for the left_subtree
    left_ftrs= get_used_features_recursive(
        cur_model['false'], features_so_far
    )
    left_ftrs.add(cur_ftr)

    # Call for the right_subtree
    right_ftrs = get_used_features_recursive(
         cur_model['true'], features_so_far
    )

    return left_ftrs | right_ftrs

def get_used_features(model_json: dict):
    """
    Gets a set containing each feature used in this tree.
    Args:
        model_json : dict -- The json representation of a (sub)tree
    """
    return get_used_features_recursive(model_json, set())

def set_positive_predictions_recursive(
    positive_preds_so_far: np.array, 
    running_mask: np.array, 
    cur_model: dict,
    original_features_to_relative_features: dict = None
):
    """
    Gets the set of unique prediction vectors associated with this Rashomon set.
    Here, prediction vectors are defined/structured as bit vectors of size
    2^num_binary_ftrs; the value at each index corresponds to the predicted
    label for the binary representation of that index. So, if index 5 with 4 input
    ftrs is true in this rep, we have 5 = 0101 ==> ftr 0 and ftr 2 are true,
    ftr 1 and ftr3 are false means the prediction is true
    Args: 
        positive_preds_so_far -- the partially constructed bit vector, where each integer index 
            represents the prediction corresponding to its binary form; ie, the following bit_vector
            defined over one variable [0, 1] means the when ftr 0 is 0, predict 0 (because index 0 is 0); 
            when ftr 0 is 1, predict 1 (because index 1 is 1).
        running_mask -- a running mask selecting which indices will be set to true
        cur_model -- the current subtree in dict form
        original_features_to_relative_features -- if we are using feature indices relative to the
            other features used in this tree (as opposed to indices in the overall dataset), 
            this dictionary should map from the original feature indices to the relative 
            feature indices. For example, in a tree using ftrs 1, 17, and 35, this dictionary
            would be: {1: 0, 17: 1, 35: 2}
    Returns:
        positive_preds_so_far -- the binary prediction vector with labels set correctly
            for this subtree
    """
    if "prediction" in cur_model:
        # Base case
        positive_preds_so_far[running_mask] = cur_model['prediction']
        return positive_preds_so_far

    if original_features_to_relative_features is None:
        target_ftr = cur_model["feature"]
    else:
        target_ftr = original_features_to_relative_features[cur_model["feature"]]
    assert cur_model["reference"] == "true"

    # This cursed little line [(i >> c) % 2 for i in range(bit_vector_so_far.shape[0])] checks, for
    # each integer i, whether index target_ftr is true in the binary representation
    # of the integer i. So if we have two ftrs and we want to get the indices for ftr 0, it will return
    # [0 1 0 1]
    mask_for_ftr = np.array(
        [(i >> target_ftr) % 2 for i in range(positive_preds_so_far.shape[0])]
    ).astype(bool)

    running_mask_right = running_mask & mask_for_ftr
    running_mask_left = running_mask & (~mask_for_ftr)

    # Call for the left_subtree
    left_positives_preds_vec = set_positive_predictions_recursive(
        positive_preds_so_far, running_mask_left, cur_model['false'], original_features_to_relative_features
    )

    # Call for the right_subtree
    right_positives_preds_vec = set_positive_predictions_recursive(
        positive_preds_so_far, running_mask_right, cur_model['true'], original_features_to_relative_features
    )

    return left_positives_preds_vec | right_positives_preds_vec

def get_prediction_vector(model_json: dict, num_ftrs: int, original_features_to_relative_features: dict = None):
    """
    A simple wrapper around the recursive bit vector builder.
    We initialize all predictions as 0, and gradually set them
    to 1 where appropriate in the recursive func
    Args: 
        model_json : dict -- the model of interest as a dictionary
        num_ftrs : int -- the number of features over which this model
            was fit
        original_features_to_relative_features -- if we are using feature indices relative to the
            other features used in this tree (as opposed to indices in the overall dataset), 
            this dictionary should map from the original feature indices to the relative 
            feature indices. For example, in a tree using ftrs 1, 17, and 35, this dictionary
            would be: {1: 0, 17: 1, 35: 2}
    """
    return set_positive_predictions_recursive(
        np.zeros(2**num_ftrs).astype(bool), 
        np.ones(2**num_ftrs).astype(bool),
        model_json,
        original_features_to_relative_features=original_features_to_relative_features
    )

def bits_to_int(bit_vec: np.array):
    """
    Converts a bit vector to an integer index
    Args:
        bit_vec -- a binary array indicating
            which features are true
    """
    total = 0
    for bit_ind in range(len(bit_vec)):
        total += bit_vec[bit_ind] * (2**bit_ind)
    return total

def bool2int(x):
    """
    Converts a bit array to an integer value
    Args:
        x : np.array -- a bit array, where index 0 is the lowest
            order bit
    """
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return int(y)