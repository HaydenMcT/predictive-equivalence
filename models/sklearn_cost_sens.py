import numpy as np
from scipy.stats import entropy
from collections import Counter

class CostSensitiveDecisionTree:
    def __init__(self, feature_costs, feature_cost_scalar=1.0, max_depth=5, min_samples_split=2):
        """
        A simple greeedy method for optimizing wrt accuaracy and cost
            feature_costs: np.array -- An array where feature_costs[i] is the cost
                of feature i
            feature_cost_scalar: float -- A constant scalar to multiply
                the cost of each feature by when optimizing; lower values place
                more weight on accuracy relative to cost
            max_depth: int -- The maximum depth of the tree
            min_samples_split: int -- The minimum features that need to be in a node
                to split further
        """
        self.feature_costs = np.array(feature_costs)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_cost_scalar = feature_cost_scalar
        self.tree = None

    def fit(self, X, y):
        X = X.values
        y = y.values
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = X.values
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        """ Recursively builds the cost-sensitive decision tree """
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1:
            return np.round(np.median(y))

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.round(np.median(y))

        # Partition data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _find_best_split(self, X, y):
        """ Finds the best split based on cost-sensitive information gain """
        best_gain = -np.inf
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._compute_information_gain(X[:, feature], y, threshold)
                gain -= self.feature_costs[feature] * self.feature_cost_scalar  # Penalize based on test cost
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold

        return best_feature, best_threshold

    def _compute_information_gain(self, feature_values, y, threshold):
        """ Compute entropy-based information gain """
        parent_entropy = entropy(np.bincount(y) / len(y), base=2)

        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        left_entropy = entropy(np.bincount(y[left_mask]) / left_mask.sum(), base=2) if left_mask.sum() > 0 else 0
        right_entropy = entropy(np.bincount(y[right_mask]) / right_mask.sum(), base=2) if right_mask.sum() > 0 else 0

        left_weight = left_mask.sum() / len(y)
        right_weight = right_mask.sum() / len(y)

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _traverse_tree(self, x, node):
        """ Traverse the tree to make predictions """
        if isinstance(node, dict):
            if x[node["feature"]] <= node["threshold"]:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        return node  # Return the leaf class


def correct_cost_sens_dict(tree_dict):
    """
    Converts the internal dictionary structure of CostSensitiveDecisionTree
    to the sklearn-style structure used elsewhere in our code
    """
    def _recurse_to_dict(node_id, source_node):
        node = {}
        node['id'] = node_id
        next_id = node_id + 1
        ####################
        #features needed for dnf_trees: 
        node['reference'] = 'true' #queries assumed to always correspond to whether a feature is True; 
                                   # since we're expecting to be dealing with binarized data always with
                                   # thresholds of 0, where right/true corresponds to > 0, this should always hold
        node['relation'] = '=='
        node['feature'] = source_node['feature']
        ####################
        if type(source_node['left']) is not dict:
            node['false'] = {
                "prediction": source_node['left'],
                "id": next_id,
                "reference": 'true',
                "relation": '=='
            }
            next_id += 1
        else:
            node_false, next_id = _recurse_to_dict(next_id, source_node['left'])
            node['false'] = node_false


        if type(source_node['right']) is not dict:
            node['true'] = {
                "prediction": source_node['right'],
                "id": next_id,
                "reference": 'true',
                "relation": '=='
            }
            next_id += 1
        else:
            node_true, next_id = _recurse_to_dict(next_id, source_node['right'])
            node['true'] = node_true
        
        return node, next_id
    return _recurse_to_dict(0, tree_dict)[0]