import pandas as pd
import numpy as np

from gosdt import GOSDTClassifier as GOSDT


DEFAULT_GOSDT_CONFIG = {
        'regularization': 0.01,
        'max_depth': 5,
        'similar_support': False, 
        'time_limit': 300, 
        'allow_small_reg': True
    }
class GosdtWrapper: 
    '''
    Wrapper class for GOSDT model. Needed to allow predictions 
    to stay in the original space, even if trained on data which does 
    not include all labels
    '''
    def __init__(self, config: dict = DEFAULT_GOSDT_CONFIG):
        '''
        Initialize the model with the given configuration
        config: dict, default=DEFAULT_GOSDT_CONFIG
            The configuration to use for the GOSDT model
        '''
        self.config = config
        self.model = GOSDT(**config)
        self.leaf_pred = None # if not None, we'll just predict this label

    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        Fit the model on the given data. If data only has one label, just predict that 
        one label with a single leaf. 
        X: pd.DataFrame
            The data to fit on
        y: pd.Series
            The labels to fit on
        '''
        observed_labels = list(y.unique()) #TODO: potentially abstract this behaviour into submodels
        if len(observed_labels) == 1:
            # If we only have one label, we should just predict that label
            self.leaf_pred = observed_labels[0]
        else: 
            self.leaf_pred = None
            self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame, prediction_mode: str = 'best'):
        '''
        Predict on the given data. If the model was trained on data with only one label, we
        must predict just that label. 
        X: pd.DataFrame
            The data to predict on
        prediction_mode: str, default='best'
            This is an unused parameter, included for cross-compatibility with rashomon set methods
        '''
        if self.leaf_pred is not None: 
            return np.repeat(self.leaf_pred, X.shape[0])
        else: 
            return self.model.predict(X)