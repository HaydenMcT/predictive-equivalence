import numpy as np
import time
import copy
from collections import deque
import pandas as pd
from sklearn.linear_model import LinearRegression as LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys
from tqdm import tqdm

def perturb_shuffle(data, target_cols):
    """Perturb column target_col of data

    Args:
        data (array): Data set
        target_col (int or list of ints): Index or indices of the column(s) to mess with
    Returns:
        data_perturbed: A copy of X with column target_col perturbed
    """
    data_copy = copy.deepcopy(data)
    try:
        data_copy.loc[:, target_cols] = data[target_cols].sample(frac=1, replace=False).values
    except:
        target_cols = data.columns[target_cols]
        data_copy.loc[:, target_cols] = data[target_cols].sample(frac=1, replace=False).values

    return data_copy.iloc[:, :-1], data_copy.iloc[:, -1]

def get_conditional_model_reliances(
    rset_prediction_function, 
    data_df, 
    unbinned_train_df, 
    unbinned_test_df, 
    var_of_interest=0, 
    eps=1e-6, 
    num_perts=20,
    for_joint=False,
    imputation_model=None,
    var_of_interest_unbinned=0,
    binning_fn=None,
    parameters={'C':[0.01, 0.1, 1, 10, 100], 'penalty': ('l2', 'elasticnet')}
):
    """Computes the sub and div model reliance for each tree in the given trie
    over the given dataset

    Args:
        rset_prediction_function (function): 
            A function that takes a dataframe as input, and returns a 
            [num_samples, num_models] dataframe with the predictions from
            each model in a Rashomon set
        data_df (pd.DataFrame):
            Dataframe of the binarized dataset to compute tree accuracies
        unbinned_train_df (pd.DataFrame):
            The training split of the unbinarized data
        unbinned_test_df (pd.DataFrame):
            The test split of the unbinarized data
        var_of_interest (str or [str]): 
            The column or columns to compute model reliance over
        eps (float):
            A small float value to add for div MR to prevent dividing by 0
        num_perts (int): 
            The number of permutations to consier when computing model reliance
        for_joint (bool): 
            Whether to format our variable importance df for
            joint variable importance or not
        imputation_model (class):
            The SKLearn model to use for imputation
        var_of_interest_unbinned (str or [str]):
            The list of variables corresponding to our target binarized vars
            in the original, unbinarized data
        binning_fn (function):
            A function that takes as input a non-binarized df,
            and outputs a version binarized using the same logic
            as was applied to create our original, binarized df
        imputation_params : dict
            Parameters to pass to the CMR imputation model

    Returns:
        div_model_reliances: Dictionary of div model reliance values, of the form
        {
            means: [mean],
            observed_mr_1: [proportion of R set realizing observed_mr_1],
            observed_mr_2: [proportion of R set realizing observed_mr_2],
            ...
        }
        sub_model_reliances: Dictionary of sub model reliance values, of the form
        {
            means: [mean],
            observed_mr_1: [proportion of R set realizing observed_mr_1],
            observed_mr_2: [proportion of R set realizing observed_mr_2],
            ...
        }
        num_models: The number of models in this trie
    """
    # https://arxiv.org/pdf/1801.01489.pdf
    # Count samples and accuracies of trees in place
    # Extract the x data from the dataframe (here we use all data to evaluate)
    x_all = data_df.iloc[:, :-1]
    y_all = data_df.iloc[:, -1]

    original_preds = rset_prediction_function(x_all)
    original_acc = (
        y_all.values.reshape((-1, 1)) == original_preds.values
    ).mean(axis=0)
    
    if imputation_model is None:
        x_perturbable = data_df[list(set(data_df.columns[:-1]) - set(data_df.columns[var_of_interest]))].to_numpy()
        unique_patterns = np.unique(x_perturbable, axis=0)
        perturbed_acc = 0
        for pattern in unique_patterns:
            mask = (data_df[list(set(data_df.columns[:-1]) - set(data_df.columns[var_of_interest]))].values == 
                            np.repeat(pattern.reshape(1, -1), data_df.shape[0], axis=0)).all(axis=1)
            cur_df = data_df.iloc[mask].copy()
            for _ in range(num_perts):
                x_perturbed, y_perturbed = perturb_shuffle(cur_df, var_of_interest)
                # map from -2, -1 version to 0, 1 labels
                pert_preds = rset_prediction_function(x_perturbed) #-1 * (pert_preds + 1)
                perturbed_acc += (
                    y_perturbed.values.reshape((-1, 1)) == pert_preds.values
                ).sum(axis=0)
                
        perturbed_acc = perturbed_acc.astype(float)
        perturbed_acc = perturbed_acc / num_perts
        perturbed_acc = perturbed_acc/ x_all.shape[0]
    else:
        imputer = imputation_model()
        predictor_cols = list(unbinned_train_df.columns[:var_of_interest_unbinned]) + list(unbinned_train_df.columns[var_of_interest_unbinned+1:-1])
        target_col = unbinned_train_df.columns[var_of_interest_unbinned]

        cv = GridSearchCV(imputer, parameters, cv=min(5, unbinned_train_df.iloc[:, -1].value_counts().min()))
        cv = cv.fit(unbinned_train_df[predictor_cols].values, unbinned_train_df[target_col].values)

        imputed_target_col = cv.predict(unbinned_test_df[predictor_cols].values)
        x_imputed = unbinned_test_df.values[:, :-1].copy()
        x_imputed[:, var_of_interest_unbinned] = imputed_target_col
        x_imputed = binning_fn(pd.DataFrame(x_imputed))

        pert_preds = rset_prediction_function(x_imputed) #-1 * (original_preds + 1)
        perturbed_acc = (
            y_all.values.reshape((-1, 1)) == pert_preds.values
        ).mean(axis=0)
    
    # Misclassification loss is just 1 - accuracy
    div_cmrs = (1-perturbed_acc+eps) / (1-original_acc+eps)
    sub_cmrs = (1-perturbed_acc) - (1-original_acc)

    div_model_reliances = {}
    sub_model_reliances = {}

    num_models = div_cmrs.shape[0]
    
    if for_joint:
        return div_cmrs, sub_cmrs, num_models

    start = time.time()
    '''
    For each unique div mr
    After this loop, div_model_reliances is a dictionary of the form
    {
        means: [mean],
        observed_mr_1: [proportion of R set realizing observed_mr_1],
        observed_mr_2: [proportion of R set realizing observed_mr_2],
        ...
    }
    '''
    running_mean = 0
    for val in np.unique(div_cmrs):
        div_model_reliances[val] = div_cmrs[div_cmrs == val].shape[-1] / num_models
        running_mean += div_model_reliances[val] * val
    div_model_reliances['means'] = [running_mean]

    running_mean = 0
    for val in np.unique(sub_cmrs):
        sub_model_reliances[val] = sub_cmrs[sub_cmrs == val].shape[-1] / num_models
        running_mean += sub_model_reliances[val] * val
    sub_model_reliances['means'] = [running_mean]
    
    return div_model_reliances, sub_model_reliances, num_models
    
def get_model_reliances(
    rset_prediction_function,
    data_df, 
    var_of_interest=0, 
    eps=1e-6, 
    num_perts=10,
    for_joint=False,
    verbose=False
):
    """Computes the sub and div model reliance for each tree in the given trie
    over the given dataset

    Args:
        rset_prediction_function (function): 
            A function that takes a dataframe as input, and returns a 
            [num_samples, num_models] dataframe with the predictions from
            each model in a Rashomon set
        data_df (pd.DataFrame):
            Dataframe of the dataset to compute tree accuracies
        var_of_interest (str or [str]): 
            The column or columns to compute model reliance over
        eps (float):
            A small float value to add for div MR to prevent dividing by 0
        num_perts (int): 
            The number of permutations to consier when computing model reliance
        for_joint (bool): 
            Whether to format our variable importance df for
            joint variable importance or not

    Returns:
        div_model_reliances: Dictionary of div model reliance values, of the form
        {
            means: [mean],
            observed_mr_1: [proportion of R set realizing observed_mr_1],
            observed_mr_2: [proportion of R set realizing observed_mr_2],
            ...
        }
        sub_model_reliances: Dictionary of sub model reliance values, of the form
        {
            means: [mean],
            observed_mr_1: [proportion of R set realizing observed_mr_1],
            observed_mr_2: [proportion of R set realizing observed_mr_2],
            ...
        }
        num_models: The number of models in this trie
    """
    
    # Count samples and accuracies of trees in place
    # Extract the x data from the dataframe (here we use all data to evaluate)
    x_all = data_df.iloc[:, :-1]
    y_all = data_df.iloc[:, -1]

    original_preds = rset_prediction_function(x_all)
    if original_preds.shape[0] == 0:
        return np.array((0,0)), np.array((0,0)), 0

    original_acc = (
        y_all.values.reshape((-1, 1)) == original_preds.values
    ).mean(axis=0)
    
    perturbed_acc = 0
    if verbose:
        cur_iterator = tqdm(range(num_perts), desc="Looping over perturbations")
    else:
        cur_iterator = range(num_perts)
    for _ in range(num_perts):
        x_perturbed, y_perturbed = perturb_shuffle(data_df, var_of_interest)
        pert_preds = rset_prediction_function(x_perturbed)

        perturbed_acc += (
            y_perturbed.values.reshape((-1, 1)) == pert_preds.values
        ).mean(axis=0)

    perturbed_acc /= num_perts
    
    # Misclassification loss is just 1 - accuracy
    div_mrs = (1-perturbed_acc+eps) / (1-original_acc+eps)
    sub_mrs = (1-perturbed_acc) - (1-original_acc)

    div_model_reliances = {}
    sub_model_reliances = {}

    num_models = div_mrs.shape[0]
    
    if for_joint:
        return div_mrs, sub_mrs, num_models

    start = time.time()
    '''
    For each unique div mr
    After this loop, div_model_reliances is a dictionary of the form
    {
        means: [mean],
        observed_mr_1: [proportion of R set realizing observed_mr_1],
        observed_mr_2: [proportion of R set realizing observed_mr_2],
        ...
    }
    '''
    running_mean = 0
    for val in np.unique(div_mrs):
        div_model_reliances[val] = div_mrs[div_mrs == val].shape[-1] / num_models
        running_mean += div_model_reliances[val] * val
    div_model_reliances['means'] = [running_mean]

    running_mean = 0
    for val in np.unique(sub_mrs):
        sub_model_reliances[val] = sub_mrs[sub_mrs == val].shape[-1] / num_models
        running_mean += sub_model_reliances[val] * val
    sub_model_reliances['means'] = [running_mean]
    
    return div_model_reliances, sub_model_reliances, num_models
