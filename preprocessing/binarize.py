import pandas as pd
import numpy as np

# unlimit pandas print for Series
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def bin_by_quantile(X: pd.DataFrame,
                    n_bins: int = 5,
                    continuous_features: list[str] = None,
                    missing_values: list = [np.nan, pd.NA],
                    keep_missingness_indicators=False,
                    separator_str="_<=_") -> pd.DataFrame:
    """Bin continuous data into quantiles.

    Args:
        X (pd.DataFrame): A DataFrame containing continuous features to binarize.
        n_bins (int, optional): Number of quantiles to bin into. Defaults to 5.
        continuous_features (list[str], optional): List of column names to bin. if None (which is the default), then all columns are binned.
        missing_values (list, optional): List of values to treat as missing. Defaults to [np.nan, pd.NA].
        keep_missingness_indicators (bool, optional): Whether to keep missingness indicators or re-introduce missingness back into data. Defaults to False.
        separator_str (str, optional): The string that will be used to separate the base column name
            from the threshold applied for binarization. Defaults to a string that is very unlikely to
            appear in a normal column name.
    Returns:
        pd.DataFrame: The dataset binarized by quantiles. Only contains the binarized columns.
        dict: A mapping from each column name in the binarized dataset to its source column name.
    """
    if not continuous_features:
        continuous_features = [v for v in X.columns if not ((X[v] == 1) | (X[v] == 0)).all()]
        if len(continuous_features) == 0:
            return X, {c: c for c in X.columns}

    X_binary = X[continuous_features].copy()

    # standardize nans and make sure columns are numeric
    for col in X_binary.columns:
        # convert missing values to np.nan
        X_binary[col] = X_binary[col].replace(missing_values, np.nan)
        # convert to numeric
        X_binary[col] = pd.to_numeric(X_binary[col], errors='coerce')

    # binarize the columns by n_bins quantiles
    ## use the nullable Int64 datatype
    X_binary = X_binary.apply(lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop')).replace(np.nan,
                                                                                                     -1).astype('Int64')
    # convert to one-hot encoding
    ## note the nullable integer binary data type
    ## this will one-hot encode missingness indicators also
    X_binary = pd.get_dummies(X_binary, columns=X_binary.columns, prefix_sep=separator_str).astype('Int64')

    if keep_missingness_indicators:
        return X_binary, {c: c.split(separator_str)[0] for c in X_binary.columns}

    # reintroduce missingness in the now-binarized columns
    for col in continuous_features:
        try:
            missing = X_binary[col + f'{separator_str}-1'].astype('bool')
        except KeyError:
            # column had no missing values
            continue
        X_binary = X_binary.drop(col + f'{separator_str}-1', axis=1)
        for bcol in X_binary.columns:
            # drop everything after the first instance of separator_str
            bcol_compare = bcol.split(separator_str)[0]
            if col == bcol_compare:
                # missing values are encoded with pandas' NA
                X_binary.loc[missing, bcol] = pd.NA
    return X_binary, {c: c.split(separator_str)[0] for c in X_binary.columns}
