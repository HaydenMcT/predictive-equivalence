import pandas as pd
from preprocessing.data import get_complete_subset, get_incomplete_subset


def test_get_complete_subset():
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [5, 6, 7, 8],
        'C': [None, 10, None, 12],
        'D': [13, 14, 15, 16],
        'y': [0, 1, 0, 1],
    })
    X = df.drop(columns='y')
    y = df['y']

    X_complete, y_complete = get_complete_subset(X, y)
    assert X_complete.equals(X.iloc[[1, 3]])
    assert y_complete.equals(y.iloc[[1, 3]])


def test_get_incomplete_subset():
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [5, 6, 7, 8],
        'C': [None, 10, None, 12],
        'D': [13, 14, 15, 16],
        'y': [0, 1, 0, 1],
    })
    X = df.drop(columns='y')
    y = df['y']

    X_incomplete, y_incomplete = get_incomplete_subset(X, y)
    assert X_incomplete.equals(X.iloc[[0, 2]])
    assert y_incomplete.equals(y.iloc[[0, 2]])
