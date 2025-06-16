from __future__ import annotations

import json
import copy
from typing import Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from treefarms import TREEFARMS

from models.tree_classifier_wrapper import TreeClassifierWrapper, create_tree_classifier

DEFAULT_TF_CONFIG = {
    "depth_budget": 5,
    "rashomon_bound_adder": 0.03,
    "rashomon_bound_multiplier": 0,
    "rashomon_bound": 0,
    "rashomon_ignore_trivial_extensions": True,
    "regularization": 0.02,
    "verbose": False
}
TF_PREDICTION_MODES = ["majority", "best", "random"]


def json_set_to_model_list(json_set: set,
                           X_train: pd.DataFrame = None,
                           y_train: pd.DataFrame = None,
                           tree_type='default') -> list:
    '''
    Converts a set of JSON representations of TreeClassifiers
    to a list of TreeClassifiers that may be easily iterated over
    Args:
        json_set : set -- a set of trees represented as json strings
        X_train : pd.DataFrame -- the X values used to fit these trees.
            If provided, these are used to populate the loss attribute of
            each tree
        y_train : pd.DataFrame -- the y values used to fit these trees.
            If provided, these are used to populate the loss attribute of
            each tree
    '''
    json_list = list(json_set)
    model_list = []
    if X_train is not None and y_train is not None:
        model_list = [
            create_tree_classifier(json.loads(cur_dict), X=X_train, y=pd.DataFrame(y_train), tree_type=tree_type)
            for cur_dict in json_list
        ]
    else:
        model_list = [create_tree_classifier(json.loads(cur_dict), tree_type=tree_type) for cur_dict in json_list]
    return model_list


class TreeFarmsWrapper:

    def __init__(self, tf_config: dict = DEFAULT_TF_CONFIG, tree_type='default'):
        self.configuration: dict = tf_config
        self.tf: TREEFARMS = TREEFARMS(self.configuration)
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.X_train_no_nan_cols: pd.DataFrame = None
        self.observed_labels: list[bool] = None
        self.json_set: set[str] = None
        self.model_list: list[TreeClassifierWrapper] = None
        self.model_set: set[TreeClassifierWrapper] = None
        self.unusable_data_prediction: Any = None
        self.fit_on_unusable_data: bool = None
        self.tree_type = tree_type

        if (self.tree_type == 'IttyBitty') and not (tf_config['rashomon_ignore_trivial_extensions']):
            raise ValueError(
                """
                When tree_type is IttyBitty, rashomon_ignore_trivial_extensions should be True, as
                the IttyBitty representation of trees does not recognize trees that use different
                features as equivalent.
                """
            )

    def get_duplication_rates(self):
        """
        Computes a dictionary mapping a dnf form to the number
        of expressions of that dnf found in this rashomon set
        """
        assert self.tree_type == 'DNF'
        dnf_to_count = {}
        for t in self.model_list:
            if t.pos_dnf in dnf_to_count:
                dnf_to_count[t.pos_dnf] = dnf_to_count[t.pos_dnf] + 1
            else:
                dnf_to_count[t.pos_dnf] = 1

        return dnf_to_count

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        '''
        If there are any columns with missing values, we need to drop them to use treefarms.
        If that means dropping all columns, we handle this with a flag.
        We also handle the standing bug with treefarms where, if all rows are equivalent,
        TreeFarms crashes, in that same case.
        '''
        self.X_train_no_nan_cols = self.X_train.dropna(axis=1)
        self.observed_labels = list(self.y_train.unique().astype(bool))
        if self.X_train_no_nan_cols.shape[1] == 0 or (self.X_train_no_nan_cols.nunique(axis=0) == 1).all():
            self.unusable_data_prediction = self.y_train.mode()[0]
            self.fit_on_unusable_data = True
        else:
            self.tf.fit(self.X_train_no_nan_cols, self.y_train)
            '''
            Getting a python set version of the model set.
            We use the JSON representation of each tree to support 
            equality comparisons for set intersect/union, as well
            as to allow for corrections on edge cases treefarms 
            does not handle well (i.e. single label datasets)
            or to remap features back to the original dataset, 
            for the processing we did earlier when dropping nan columns
            '''
            self.json_set = {self._get_corrected_json(i) for i in range(self.tf.get_tree_count())}
            self.model_list = json_set_to_model_list(self.json_set, self.X_train, self.y_train, self.tree_type)
            if 'DNF' in self.tree_type or self.tree_type == 'IttyBitty' or self.tree_type == 'PCC':
                self.model_set = set(self.model_list)
            self.fit_on_unusable_data = False

    def predict(self, X: pd.DataFrame, prediction_mode: str = "best", unique_only=True, extra_info=False) -> pd.Series:
        if self.fit_on_unusable_data:
            # if we were fit on empty data, predict the majority class from the training set.
            return pd.Series([self.unusable_data_prediction] * len(X), index=X.index)

        if prediction_mode == "majority":
            return self.majority_vote(X, unique_only)
        if prediction_mode == "best":
            return self.predict_best(X)
        if prediction_mode == "random":
            return self.predict_random(X)
        if prediction_mode == "best_non_nan":
            return self.predict_best_non_nan(X, unique_only, extra_info)
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")

    def __len__(self):
        return len(self.json_set)

    def _get_corrected_json(self, index: int):
        '''
        Simple wrapper function to make sure our string tree representations
        are consistent and always predict the relevant label.
        '''
        json_dict = json.loads(self.tf[index].json())

        if len(self.observed_labels) == 1:
            self._correct_predicted_label(json_dict, self.observed_labels[0])

        if (self.X_train_no_nan_cols.shape[1] < self.X_train.shape[1] and self.X_train_no_nan_cols.shape[1] > 0):
            self._correct_feature_indices(json_dict)

        return json.dumps(json_dict)

    def _correct_feature_indices(self, tree_rep: dict):
        '''
        Simple wrapper function to make sure our string tree representations
        use indices relevant to the full dataset.

        Args: 
            tree_rep : dict -- the JSON representation of a tree

        Modifies: 
            tree_rep : dict -- the JSON representation of a tree
            (Modified to use the correct feature indices for the full dataset, 
             rather than the subset of columns it was trained on from X_train_no_nan_cols)
        '''
        if 'feature' in tree_rep:
            tree_rep['feature'] = self.X_train.columns.get_loc(self.X_train_no_nan_cols.columns[tree_rep['feature']])
            self._correct_feature_indices(tree_rep['true'])
            self._correct_feature_indices(tree_rep['false'])

    def _correct_predicted_label(self, json_dict: dict, target_label: bool):
        '''
        Simple wrapper function to make sure our string tree representations
        are consistent and always predict the relevant label.
        '''
        if 'prediction' in json_dict:
            json_dict['prediction'] = int(target_label)
        else:
            self._correct_predicted_label(json_dict['true'], target_label)
            self._correct_predicted_label(json_dict['false'], target_label)

    def get_tree_count(self) -> int:
        return len(self.model_list)

    def get_unique_tree_count(self) -> int:
        if self.model_set is None:
            raise UserWarning("Unique tree count is not available for this tree type.")
        return len(self.model_set)

    def intersection(self, other: TreeFarmsWrapper) -> TreeFarmsWrapper:
        new_tf = copy.deepcopy(self)
        new_tf.json_set = self.json_set & other.json_set
        new_tf.model_list = json_set_to_model_list(new_tf.json_set, new_tf.X_train, new_tf.y_train, new_tf.tree_type)
        if 'DNF' in new_tf.tree_type or new_tf.tree_type == 'IttyBitty' or new_tf.tree_type == 'PCC':
            new_tf.model_set = set(new_tf.model_list)

        return new_tf

    def predict_with_tree(self, tree: TreeClassifierWrapper, X: pd.DataFrame) -> pd.Series:
        '''
        Predict every example in X using the given tree.
        '''
        if 'DNF' in self.tree_type or self.tree_type == 'PCC':
            return tree.predict(X)

        preds = tree.predict(X)
        preds[np.isnan(preds)] = pd.NA
        return preds

    def predict_for_all_trees(self, X: pd.DataFrame, unique_only=True) -> pd.DataFrame:
        '''Get predictions from every tree in the Rashomon set.
        '''
        preds = []
        # iterating over the indices of the trees instead of the list
        # because the iterator will run out of bounds for some reason
        if unique_only and self.tree_type in ['DNF', 'DNF_exhaustive', 'IttyBitty', 'PCC']:
            models = self.model_set
        else:
            models = self.model_list
        for model in models:
            preds.append(self.predict_with_tree(model, X))
        return pd.DataFrame(preds).T

    def _vote_fn(self, row):
        try:
            # get the most common prediction among the trees
            # excluding NA values (trees that couldn't make a prediction)
            vote = row.value_counts().idxmax()
        except ValueError:
            # no tree in the Rashomon set was able to make a prediction
            vote = pd.NA
        return vote

    def majority_vote(self, X: pd.DataFrame, unique_only=True) -> pd.Series:
        '''Majority vote among trees in the Rashomon set.
        
        This treats the Rashomon set as an ensemble of trees, and makes a prediction
        for each example in X by taking the most common prediction among the trees.
        '''
        preds = self.predict_for_all_trees(X, unique_only)
        votes = preds.apply(self._vote_fn, axis=1).astype('Int64')
        return votes

    def _find_non_nan(self, row, index_order):
        try:
            # get the first prediction among the trees
            # excluding NA values (trees that couldn't make a prediction)
            ordered_row = row[index_order].reset_index(drop=True)
            vote = ordered_row[np.where(~ordered_row.isna())[0][0]]
        except IndexError:
            # no tree in the Rashomon set was able to make a prediction
            vote = pd.NA
        return vote

    def _find_sparsity_non_nan(self, row, index_order, sparsity=None): 
        try:
            # get the first prediction among the trees
            # excluding NA values (trees that couldn't make a prediction)
            ordered_row = row[index_order].reset_index(drop=True)
            if sparsity is not None:
                vote_sparsity = sparsity[np.where(~ordered_row.isna())[0][0]]
            else: 
                vote_sparsity = pd.NA
        except IndexError:
            # no tree in the Rashomon set was able to make a prediction
            vote_sparsity = pd.NA
        return vote_sparsity

    def predict_best_non_nan(self, X: pd.DataFrame, unique_only=True, extra_info=False) -> pd.Series:
        '''For each sample, predicts from the tree in the Rashomon set with the
        best training loss, among those trees that predict not NaN. 
        If the loss function of each model has not been populated, this will throw
        a KeyError
        '''
        if not unique_only:
            train_losses = [m.loss() for m in self.model_list]
        else:
            train_losses = [m.loss() for m in self.model_set]
        best_to_worst_tree_indices = np.argsort(train_losses)
        preds = self.predict_for_all_trees(X, unique_only)
        votes = preds.apply(lambda x: self._find_non_nan(x, best_to_worst_tree_indices), axis=1).astype('Int64')
        if extra_info and not unique_only: 
            sparsities = pd.Series([len(m) for m in self.model_list])
            sparsities = sparsities[best_to_worst_tree_indices].reset_index(drop=True)
            sparsity_votes = preds.apply(lambda x: self._find_sparsity_non_nan(x, best_to_worst_tree_indices, sparsity=sparsities), axis=1).astype('Int64')
            return votes, sparsity_votes
        return votes

    def predict_best(self, X: pd.DataFrame) -> pd.Series:
        '''Predictions from the tree in the Rashomon set with the
        best training loss. If the loss function of each model
        has not been populated, this will throw a KeyError
        '''
        train_losses = [m.loss() for m in self.model_list]
        tree_ind = np.argmin(train_losses)
        preds = self.predict_with_tree(self.model_list[tree_ind], X)

        return preds

    def predict_random(self, X: pd.DataFrame) -> pd.Series:
        '''Predictions from a random tree in the Rashomon set.
        '''
        tree_ind = np.random.randint(len(self.model_list))
        preds = self.predict_with_tree(self.model_list[tree_ind], X)

        return preds

    def __getitem__(self, index: int) -> TreeClassifierWrapper:
        return self.model_list[index]


def construct_tree_rset(X: pd.DataFrame,
                        y: pd.Series,
                        tf_config: dict = DEFAULT_TF_CONFIG,
                        tree_type='default') -> TreeFarmsWrapper:
    '''
    A simple wrapper function to handle constructing TreeFarms
    for a dataset. This handles a standing ``bug'' in TreeFarms, 
    where when only a single label is present in the dataset,
    TreeFarms will always predict that label as 0.
    '''
    tf = TreeFarmsWrapper(tf_config, tree_type)
    tf.fit(X, y)

    return tf
