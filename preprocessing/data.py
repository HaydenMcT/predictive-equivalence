import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from gosdt import ThresholdGuessBinarizer

from preprocessing.binarize import bin_by_quantile

def load_data(filename: str, y_name: str = None, balance=False) -> tuple[pd.DataFrame, pd.Series]:
    """Load data from a CSV file and return the feature matrix and target vector.

    Args:
        filename (str): The name of the CSV file to load. Data must be in the 'data' directory.
        y_name (str, optional): Name of the target variable. Defaults to None, in which case
        the rightmost column in the dataset is used as the target

    Returns:
        tuple[pd.DataFrame, pd.Series]: The feature matrix and target vector.
    """

    df = pd.read_csv(f'data/{filename}')
    if y_name is None:
        y_name = df.columns[-1]
    y = df[y_name]
    if balance:
        pos_prop = y.mean()
        neg_prop = 1 - pos_prop
        if neg_prop > pos_prop:
            df = pd.concat([
                df[y == 1],
                df[y == 0].sample(int(pos_prop * df.shape[0]))
            ], axis=0).sample(frac=1)
        else:
            df = pd.concat([
                df[y == 1].sample(int(neg_prop * df.shape[0])),
                df[y == 0]
            ], axis=0).sample(frac=1)
        if y_name is None:
            y_name = df.columns[-1]
        y = df[y_name]
    X = df.drop(y_name, axis=1)
    return X, y


def get_complete_subset(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Get the subset of the data with no missing values.
    
    Assumes missing values are encoded with np.nan.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The feature matrix and target vector with no missing values.
    """

    complete_rows = X.dropna(axis=0).index
    X_complete = X.loc[complete_rows]
    y_complete = y.loc[complete_rows]

    return X_complete, y_complete


def get_incomplete_subset(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Get the subset of the data with missing values.

    Assumes missing values are encoded with np.nan.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        tuple[pd.DataFrame, pd.Series]: The feature matrix and target vector with missing values.
    """

    incomplete_rows = X[X.isna().any(axis=1)].index
    X_incomplete = X.loc[incomplete_rows]
    y_incomplete = y.loc[incomplete_rows]

    return X_incomplete, y_incomplete


def get_gosdt_guesses_thresholds_on_complete_subset(X: pd.DataFrame,
                                                    y: pd.Series,
                                                    n_est: int = 40,
                                                    max_depth: int = 1) -> tuple[pd.DataFrame, list, list]:
    """Get the thresholds for binarizing the complete portion of the data using GOSDT+Guesses.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        n_est (int, optional): Number of estimators for GOSDT threshold guessing. Defaults to 40.
        max_depth (int, optional): Maximum depth of GOSDT threshold guessing. Defaults to 1.

    Returns:
        tuple[list, list] : The thresholds and headers for binarizing the data.
    """

    X_complete, y_complete = get_complete_subset(X, y)

    enc = ThresholdGuessBinarizer(n_estimators=n_est, max_depth=max_depth, random_state=42)
    enc.set_output(transform="pandas")
    enc.fit(X_complete, y_complete)
    thresholds = enc.thresholds_
    headers = enc.get_feature_names_out()
    og_ftr_to_binned = enc.feature_map()

    bin_name_to_og = {}
    for k in og_ftr_to_binned:
        for new_c_ind in og_ftr_to_binned[k]:
            bin_name_to_og[headers[new_c_ind]] = X_complete.columns[k]

    return thresholds, headers, bin_name_to_og


def prepare_binarized_data(X: pd.DataFrame,
                           y: pd.Series,
                           missing_values: list = [np.nan],
                           columns_to_drop: list[str] | str = None,
                           num_quantiles: int = 0,
                           gosdt_guesses: bool = True,
                           fix_headers_for_xgboost: bool = False,
                           keep_missingness_after_binarization: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Process the data by replacing missing values and removing the target variable.
    
    Steps of preprocessing:
        - Replace missing_values with np.nan
        - Drop any columns specified in columns_to_drop
        - If num_quantiles > 0, binarize the data by quantiles
        - If gosdt_guesses is True, binarize and do feature selection 
            using GOSDT+Guesses on the complete subset of the data
        
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        missing_values (list, optional): List of values to replace with np.nan. Defaults to [np.nan].
        columns_to_drop (list[str] | str, optional): Column or list of columns to drop. Defaults to None.
        num_quantiles (int, optional): Number of quantiles to binarize the data. Defaults to 0.
        gosdt_guesses (bool, optional): Whether to use GOSDT+Guesses for binarization and feature selection. Defaults to True.
        fix_headers_for_xgboost (bool, optional): Whether to replace special characters in column names for xgboost. Defaults to False.
        keep_missingness_after_binarization (bool, optional): If true, replace indicators with nan after binarization
        
    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Training and testing feature matrices and target vectors.
    """
    if num_quantiles == 0 and not gosdt_guesses:
        raise ValueError("Tried to binarize with num_quantiles=0 and no gosdt guesses; this does nothing.")

    X_ = X.copy()
    X_ = X_.replace(missing_values, np.nan)

    if columns_to_drop:
        if isinstance(columns_to_drop, str):
            columns_to_drop = [columns_to_drop]
        X_ = X_.drop(columns=columns_to_drop)

    if num_quantiles > 0:
        X_, bin_name_to_original = bin_by_quantile(X_, n_bins=num_quantiles, missing_values=[np.nan])

    print(f"num_quantiles: {num_quantiles}")
    print(X_)
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=42)
    X_train_missing_mask = X_train.isna()
    X_test_missing_mask = X_test.isna()
    print(X_train)

    if gosdt_guesses:
        thresholds, headers, bin_name_to_original = get_gosdt_guesses_thresholds_on_complete_subset(X_train, y_train)
        X_train = nansafe_cut(X_train, thresholds)[headers]
        X_test = nansafe_cut(X_test, thresholds)[headers]

    if keep_missingness_after_binarization:
        for col in X_train.columns:
            orig_col_name = col.split(' <= ')[0]
            X_train[col] = X_train[col].astype('Int64')
            X_train.loc[X_train_missing_mask[orig_col_name], col] = pd.NA
            X_test[col] = X_test[col].astype('Int64')
            X_test.loc[X_test_missing_mask[orig_col_name], col] = pd.NA

    if fix_headers_for_xgboost:
        # xgboost does not accept '[', ']', or '<' in column names
        X_train.columns = [col.replace('[', '(').replace(']', ')').replace('<', 'L') for col in X_train.columns]
        X_test.columns = [col.replace('[', '(').replace(']', ')').replace('<', 'L') for col in X_test.columns]

    X_train = X_train.astype('Int64')
    X_test = X_test.astype('Int64')
    y_train = y_train.astype('Int64')
    y_test = y_test.astype('Int64')

    return X_train, y_train, X_test, y_test, bin_name_to_original


def prepare_binarized_data_kfolds(X: pd.DataFrame,
                                  y: pd.Series,
                                  missing_values: list = [np.nan],
                                  columns_to_drop: list[str] | str = None,
                                  num_quantiles: int = 0,
                                  gosdt_guesses: bool = True,
                                  fix_headers_for_xgboost: bool = False,
                                  keep_missingness_after_binarization: bool = False,
                                  num_folds=5) -> tuple[pd.DataFrame, pd.Series]:
    """Process the data by replacing missing values and removing the target variable.
    
    Steps of preprocessing:
        - Replace missing_values with np.nan
        - Drop any columns specified in columns_to_drop
        - If num_quantiles > 0, binarize the data by quantiles
        - If gosdt_guesses is True, binarize and do feature selection 
            using GOSDT+Guesses on the complete subset of the data
        
    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        missing_values (list, optional): List of values to replace with np.nan. Defaults to [np.nan].
        columns_to_drop (list[str] | str, optional): Column or list of columns to drop. Defaults to None.
        num_quantiles (int, optional): Number of quantiles to binarize the data. Defaults to 0.
        gosdt_guesses (bool, optional): Whether to use GOSDT+Guesses for binarization and feature selection. Defaults to True.
        fix_headers_for_xgboost (bool, optional): Whether to replace special characters in column names for xgboost. Defaults to False.
        keep_missingness_after_binarization (bool, optional): If true, replace indicators with nan after binarization
        
    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Training and testing feature matrices and target vectors.
    """

    X_ = X.copy()
    X_ = X_.replace(missing_values, np.nan)

    if columns_to_drop:
        if isinstance(columns_to_drop, str):
            columns_to_drop = [columns_to_drop]
        X_ = X_.drop(columns=columns_to_drop)

    # if num_quantiles > 0:
    #     X_ = bin_by_quantile(X_, n_bins=num_quantiles, missing_values=[np.nan])

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = kf.split(X_)
    retvals = []
    pre_binarization = []
    for train_index, test_index in folds:
        pre_guess_X_train, pre_guess_X_test = X_.iloc[train_index], X_.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_missing_mask = pre_guess_X_train.isna()
        X_test_missing_mask = pre_guess_X_test.isna()

        # if gosdt_guesses:
        thresholds, headers, _ = get_gosdt_guesses_thresholds_on_complete_subset(pre_guess_X_train, y_train)
        X_train = nansafe_cut(pre_guess_X_train.copy(), thresholds)[headers]
        X_test = nansafe_cut(pre_guess_X_test.copy(), thresholds)[headers]

        if keep_missingness_after_binarization:
            for col in X_train.columns:
                orig_col_name = col.split(' <= ')[0]
                X_train[col] = X_train[col].astype('Int64')
                X_train.loc[X_train_missing_mask[orig_col_name], col] = pd.NA
                X_test[col] = X_test[col].astype('Int64')
                X_test.loc[X_test_missing_mask[orig_col_name], col] = pd.NA

        if fix_headers_for_xgboost:
            # xgboost does not accept '[', ']', or '<' in column names
            X_train.columns = [col.replace('[', '(').replace(']', ')').replace('<', 'L') for col in X_train.columns]
            X_test.columns = [col.replace('[', '(').replace(']', ')').replace('<', 'L') for col in X_test.columns]

        X_train = X_train.astype('Int64')
        X_test = X_test.astype('Int64')
        y_train = y_train.astype('Int64')
        y_test = y_test.astype('Int64')

        retvals.append((X_train, y_train, X_test, y_test))
        pre_binarization.append((pre_guess_X_train, pre_guess_X_test, thresholds, headers))
    return retvals, pre_binarization


# This function replicates the cut function from
# GOSDT, but encodes missing values as 0 for each threshold.
# In the function from GOSDT, nan values will be encoded
# as 1 for each indicator
# This is now adapted to work with ts structured as
# [(ind_of_og_col, thresh), (ind_of_og_col, thresh), etc]
def nansafe_cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j, thresh in ts:
        X[colnames[j] + ' <= ' + str(thresh)] = 1
        k = (df[colnames[j]] > thresh) | (df[colnames[j]].isna())
        X.loc[k, colnames[j] + ' <= ' + str(thresh)] = 0
    X = X.drop(colnames, axis=1)
    return X

# This function replicates the cut function from
# GOSDT, but keeps missing values as na for each threshold.
# This is adapted to work with ts structured as
# [(ind_of_og_col, thresh), (ind_of_og_col, thresh), etc]
def nan_preserving_cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j, thresh in ts:
        X[colnames[j] + ' <= ' + str(thresh)] = 1
        k = df[colnames[j]] > thresh
        X.loc[k, colnames[j] + ' <= ' + str(thresh)] = 0
        na_idx = df[colnames[j]].isna()
        X.loc[na_idx, colnames[j] + ' <= ' + str(thresh)] = pd.NA
    X = X.drop(colnames, axis=1)
    return X