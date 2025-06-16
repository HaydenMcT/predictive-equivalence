import numpy as np
import pandas as pd
from sympy import Or, And, Not, to_dnf, simplify_logic

from treefarms.model.tree_classifier import TreeClassifier

from utils.predictive_uniqueness import get_prediction_vector, get_used_features, bool2int
from models.pcc import PCC, pcc_representation


class TreeClassifierWrapper(TreeClassifier):

    def __init__(self, source, encoder=None, X=None, y=None):
        super(TreeClassifierWrapper, self).__init__(source, encoder=encoder, X=X, y=y)

    def __initialize_training_loss__(self, X, y):
        """
        Compares every prediction y_hat against the labels y, then incorporates the misprediction into the stored loss values
        This is used when parsing models from an algorithm that doesn't provide the training loss in the out put
        """
        for node in self.__all_leaves__():
            node["loss"] = 0.0
        (n, m) = X.shape
        X_values = X.values
        y_values = y.values
        for i in range(n):
            node = self.__find_leaf__(X_values[i, :])
            label = y_values[i, -1]
            weight = 1 / n
            if node["prediction"] != label:
                node["loss"] += weight
        return

    def classify(self, x):
        try:
            return super(TreeClassifierWrapper, self).classify(x)
        except TypeError:
            # If the tree is unable to classify the input, return NaN
            return np.nan, np.nan

    def get_variables_for_prediction(self, X):
        """
        Computes the variables that a naive traversal of this tree would check in order
        to form a prediction for each row in X

        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

        Returns
        ---
        list of n lists : a list specifying the list of variables used for each of n rows
        """
        if self.encoder is not None:  # Perform an encoding if an encoding unit is specified
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        all_visited_features = []
        (n, m) = X.shape

        # By only accessing .values once, we avoid duplicating the
        # entire array a ton of times
        data = X.values
        for i in range(n):
            visited_features= self.get_vars_to_leaf(data[i, :])
            all_visited_features.append(visited_features)
        return all_visited_features

    def predict(self, X):
        """
        A faster version of the predict function from TreeClassifier

        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the prediction associated with each row
        """
        if self.encoder is not None:  # Perform an encoding if an encoding unit is specified
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        predictions = []
        (n, m) = X.shape

        # By only accessing .values once, we avoid duplicating the
        # entire array a ton of times
        data = X.values
        for i in range(n):
            prediction, _ = self.classify(data[i, :])
            predictions.append(prediction)
        return pd.Series(predictions)

    def get_vars_to_leaf(self, sample):
        """
        Returns
        ---
        the list of variable indices used to find a leaf
        """
        nodes = [self.source]
        visited_features = []
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                return visited_features
            else:
                visited_features.append(node["feature"])
                value = sample[node["feature"]]
                if value == 1:
                    nodes.append(node["true"])
                else:
                    nodes.append(node["false"])
                # else:
                #     raise "Unsupported relational operator {}".format(node["relation"])
    
class ConservativeTree(TreeClassifierWrapper):
    def __init__(self, source, encoder=None, X=None, y=None):
        if X is None:
            raise ValueError("X must be provided to ConservativeTree")
        super().__init__(source, encoder, X, y)
        used_ftrs = get_used_features(source)
        self.used_ftr_vector = np.zeros(X.shape[1]).astype(bool)
        for ftr in list(used_ftrs):
            self.used_ftr_vector[ftr] = True

    def classify(self, x):
        if pd.Series(x[self.used_ftr_vector]).astype('boolean').isna().any():
            return np.nan, np.nan
        return super().classify(x)      

    def get_vars_to_leaf(self, sample):
        """
        Returns
        ---
        the list of variable indices used to find a leaf
        """
        nodes = [self.source]
        visited_features = []
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                return visited_features
            else:
                visited_features.append(node["feature"])
                value = sample[node["feature"]]
                if value == 1:
                    nodes.append(node["true"])
                else:
                    nodes.append(node["false"])


class IttyBittyTree(TreeClassifierWrapper):

    def __init__(self, source, encoder=None, X=None, y=None):
        super(TreeClassifierWrapper, self).__init__(source, encoder=encoder, X=X, y=y)

        if X is None:
            raise ValueError("X must be provided to IttyBittyTree")

        used_ftrs = get_used_features(source)
        self.used_ftr_vector = np.zeros(X.shape[1]).astype(bool)
        for ftr in list(used_ftrs):
            self.used_ftr_vector[ftr] = True

        # We need to sort our used variables so that we can identify
        # which indices in the short prediction vector correspond to
        # each feature
        used_ftrs_sorted = list(used_ftrs)
        used_ftrs_sorted.sort()

        self.prediction_vector = get_prediction_vector(source, len(used_ftrs_sorted),
                                                       {used_ftrs_sorted[i]: i for i in range(len(used_ftrs_sorted))})

        # This bool2int function will treat index 0 as the lowest order bit
        self.uid = bool2int(np.concatenate([self.used_ftr_vector, self.prediction_vector]))

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return self.uid


class DNFTree(TreeClassifierWrapper):

    def __init__(self, source, encoder=None, X=None, y=None):
        super(TreeClassifierWrapper, self).__init__(source, encoder=encoder, X=X, y=y)
        self.pos_dnf, self.neg_dnf = self.get_simplified_dnfs()

    @staticmethod
    def to_dnf_string(leaf) -> str:
        rules = leaf['rules']
        dnf = []
        for feature, rule in rules.items():
            if rule['type'] == 'Categorical':
                if rule['positive']:
                    dnf.append(f'({feature})')
                else:
                    dnf.append(f'(~{feature})')
        return ' & '.join(dnf)

    def convert_to_dnf(self) -> tuple[Or, Or]:
        """Converts a tree to a DNF of its leaves.

        Returns:
            tuple[Or, Or]: DNFs of the ways to get to 1 and 0 in the tree.
        """
        leaves = self.__groups__()
        
        #otherwise, iterate through leaves
        pos_dnfs = []
        neg_dnfs = []
        for leaf in leaves:
            if leaf['prediction'] == 1:
                pos_dnfs.append(self.to_dnf_string(leaf))
            else:
                neg_dnfs.append(self.to_dnf_string(leaf))

        # cover case where tree simplifies to True or False (like a stump, or a
        # tree with many leaves that all predict the same thing)
        if len(pos_dnfs) == 0: # if no leaf can ever predict positive:
            return False, True
        if len(neg_dnfs) == 0: # if no leaf can ever predict negative:
            return True, False


        return to_dnf(' | '.join(pos_dnfs)), to_dnf(' | '.join(neg_dnfs))

    def get_simplified_dnfs(self) -> tuple[Or, Or]:
        pos_dnf, neg_dnf = self.convert_to_dnf()
        simple_pos_dnf = simplify_logic(pos_dnf, form='dnf', force=True)
        simple_neg_dnf = simplify_logic(neg_dnf, form='dnf', force=True)
        return simple_pos_dnf, simple_neg_dnf

    def _get_definite_trues_from_dnf(self, X, dnf):
        """
        Get the predictions that are definitely True, given a dnf
        and a dataframe X which may have missing values. 
        
        This outputs 1 if the prediction is known to be True,
        and 0 if the prediction is negative or unknown.
        
        Column names need to be in the same format as the atoms in the DNF.
        The atoms are the names of the features in the DNF,
        i.e. in the DNF (f1 & f2) | (~f1 & f3) | f4, 
        the atoms are f1, f2, f3, f4.

        Assumes X is dtype boolean
        """
        if dnf == True:
            return np.ones(X.shape[0], dtype=int)
        elif dnf == False:
            return np.zeros(X.shape[0], dtype=int)

        # start with all predictions as negative
        preds = np.zeros(X.shape[0], dtype=int)

        # if the DNF is an And, we have a single conjunction to evaluate
        # otherwise, we have to iterate through each conjunction in the disjunction
        to_iterate = [dnf] if not isinstance(dnf, Or) else dnf.args
        for arg in to_iterate:
            # sub_preds is True if the prediction is True, False if False, NA if unknown
            sub_preds = X.eval(arg)
            # make the predictions that are definitely True 1
            preds |= np.array(sub_preds.fillna(False))
        return preds

    def _get_definite_positive_predictions(self, X):
        """
        Get the predictions that are definitely positive.
        
        This outputs 1 if the prediction is known to be positive,
        and 0 if the prediction is negative or unknown.
        
        Column names need to be in the same format as the atoms in the DNF.

        Assumes X is dtype boolean
        """
        return self._get_definite_trues_from_dnf(X, self.pos_dnf)

    def _get_definite_negative_predictions(self, X):
        """
        Get the predictions that are definitely negative.
        
        This outputs 0 if the prediction is known to be negative,
        and 1 if the prediction is positive or unknown.
        
        Column names need to be in the same format as the atoms in the DNF.

        Assumes X is dtype boolean
        """
        return 1 - self._get_definite_trues_from_dnf(X, self.neg_dnf)

    def _attempt_to_evaluate_unknown_predictions(self, X, unknown_mask):
        """
        Attempt to correct predictions that are unknown.
        
        This is done by substituting in known feature values and checking
        if the DNF simplifies to True or False.
        
        Column names need to be in the same format as the atoms in the DNF.
        
        Example:
            Consider the positive DNF expression (f1 & f2) | (~f1 & f3) | f4, 
            with corresponding negative DNF (~f1 & ~f3 & ~f4) | (f1 & ~f2 & ~f4).
            If we only know f2 and f3 are true, but f1 and f4 are missing, then this expression is guaranteed to be true. 
            But our code will only see that none of [(f1 & f2), (~f1 & f3), f4] can be evaluated in the positive DNF,
            and none of [(~f1 & ~f3 & ~f4), (f1 & ~f2 & ~f4)] can be evaluated in the negative DNF.
            But if we substitute in f2 = 1 and f3 = 1, we get (f1) | (~f1) | f4 = True.
            The same substitution in the negative DNF simplies to False.
            So we can predict positive, even though no conjunction in either DNF evaluates to True on its own! 
            We have to postprocess to catch this.

        Assumes self.pos_dnf uses at least one feature (it doesn't represent a decision stump)
        (also assumes self.neg_dnf does not use any features unused in pos_dnf, but this is
        always true unless we're using dontcare conditions)
        """
        unknown_preds = pd.Series(np.zeros(X.shape[0]), index=X.index, dtype='Int64') + pd.NA
        used_features = [str(atom) for atom in self.pos_dnf.atoms()]
        # iterate through one copy of each row in the unknown mask
        # with a distinct missingness pattern
        for i, row in X.loc[unknown_mask, used_features].drop_duplicates().iterrows():
            if row.isna().all():
                continue
            # substitute in any feature values we know
            known_feature_values = dict(row[row.notna()])
            pos_simpler_dnf = simplify_logic(self.pos_dnf.subs(known_feature_values), force=True)
            neg_simpler_dnf = simplify_logic(self.neg_dnf.subs(known_feature_values), force=True)
            # check if either the positive or negative DNFs are true
            # and fill in all the rows that match the current pattern
            rows_matching_current_pattern = X.loc[unknown_mask, used_features].query(' & '.join(
                [f'{col} == {val}' for col, val in known_feature_values.items()]))
            if pos_simpler_dnf == True:
                unknown_preds.loc[rows_matching_current_pattern.index] = 1
            elif neg_simpler_dnf == True:
                unknown_preds.loc[rows_matching_current_pattern.index] = 0
        return unknown_preds

    def predict(self, X):
        """Predict using the DNF of the tree.

        Args:
            X (pd.DataFrame): The data to predict on.

        Assumes that X can be cast to a boolean DataFrame.
        """
        X_ = X.copy()
        X_.columns = [f'feature_{i}' for i in range(X_.shape[1])]

        # cast X_ to a boolean DataFrame
        X_ = X_.astype('boolean')

        definite_pos_preds = self._get_definite_positive_predictions(X_)
        preds = pd.Series(definite_pos_preds, index=X_.index, dtype='Int64')

        # we don't actually care about the negative predictions, but we need them to get the unknowns
        definite_neg_preds = self._get_definite_negative_predictions(X_)
        # a prediction is only unknown if the positive pred is not 1 and the negative pred is not 0
        unknown_mask = (definite_pos_preds == 0) & (definite_neg_preds == 1)
        # correct potential cases where we mistakenly did not predict
        if sum(unknown_mask) > 0:
            # only attempt to fill in unknown predictions if there are any. Saves us from
            # some buggy cases when the DNF is deterministically True (or deterministically False)
            unknown_preds = self._attempt_to_evaluate_unknown_predictions(X_, unknown_mask)
            preds[unknown_mask] = unknown_preds
        return preds

    def __eq__(self, other):
        # TODO: verify that this will always work for two trees that are the same
        return self.pos_dnf == other.pos_dnf

    def __hash__(self):
        return hash(self.pos_dnf)


class ExhaustiveDNFTree(DNFTree):

    def __init__(self, source, encoder=None, X=None, y=None, max_simplification_iters=10000):
        super().__init__(source, encoder=encoder, X=X, y=y)
        self.all_pos_terms = self._get_exhaustive_dnf(self.pos_dnf, max_simplification_iters)
        self.all_neg_terms = self._get_exhaustive_dnf(self.neg_dnf, max_simplification_iters)

    def _get_exhaustive_dnf(self, dnf, max_iters=10000):
        """
        Go through all clauses in a dnf, and find potential additional clauses to add, 
        so that our output is in dnf form and satisfies the following invariants: 
        - Logical equivalence:
            A complete sample will satisfy at least one term of the
            exhaustive DNF formula if and only if the sample satisfies at least one term
            of the input DNF formula.
        - Maximally evaluative under missing data: 
            The subset of variable values that are known in this sample will satisfy at least
            one conjunction in the output if and only if all completions of an incomplete sample
            would satisfy the dnf.
        - Minimally sized terms:
            No literal can be removed from a term without changing the logical meaning of the 
            exhaustive DNF formula.
        
        Note that there is no invariant stating that a sample falls into exactly one term
        in the output. It may fall into multiple (this is necessary so that we can keep the 
        terms minimally sized and so that we can handle missing data cases).

        We assume here that all possible combinations of literal values are valid. Additional
        compressions are possible if, say, three features cannot all co-occur as True, 
        but this case is more computationally difficult to handle and out of scope. 
        """
        # if the DNF is an And, we have a single term to evaluate
        # otherwise, we have to iterate through the terms
        if not isinstance(dnf, Or):
            return dnf
        terms = list(dnf.args)
        merge_check = list(dnf.args)  # we'll check for new clauses between merge_check and conjunctions
        for _ in range(max_iters):  # limit to prevent infinite loops
            if _ == max_iters - 1:
                raise ValueError(
                    f'Error hit! Either increase # of iterations allowed above the current # {max_iters},' +
                    " check if your example is too adversarial for our code, or check for a bug")
            new_terms = []
            to_remove = []
            # check for any possible collisions: one disagreeing literal between two conjunctions.
            for i, term in enumerate(terms):
                for j in range(i + 1, len(merge_check)):
                    term_2 = merge_check[j]
                    literals_in_term_1 = set(term.args)
                    negated_literals_in_term_2 = set([~literal for literal in term_2.args])
                    literals_in_term_2 = set(term_2.args)
                    intersection = literals_in_term_1.intersection(negated_literals_in_term_2)
                    if len(intersection) == 1:
                        new_term = And(*[
                            literal for literal in literals_in_term_1.union(literals_in_term_2)
                            if literal not in intersection and ~literal not in intersection
                        ])
                        # if we find a collision, we have to add the logical or
                        # of the two terms to the list of terms
                        # (but first we need to check if that term actually
                        # provides a valid additional truth condition not already
                        # covered by the existing terms)
                        add_new = True
                        for existing_term in terms + new_terms:
                            # if term A being satisfied means term B is always satisfied,
                            # then we can remove term A from our formula.
                            if simplify_logic(existing_term, dontcare=Not(new_term)) == True:
                                # new term is redundant
                                add_new = False
                                break
                            elif simplify_logic(new_term, dontcare=Not(existing_term)) == True:
                                # existing term is redundant
                                to_remove.append(existing_term)
                        if add_new:
                            new_terms.append(new_term)
            if len(new_terms) > 0:
                '''
                If we've found any new terms, they may be able to be merged with 
                existing ones. Repeat the loop, checking for new overlaps between the existing 
                and new terms (or between the new terms).

                We'll add the old merge candidates in with the existing terms, and put the 
                new terms in as merge candidates.
                '''
                for term in to_remove:
                    if term in terms:
                        terms.remove(term)
                    elif term in new_terms:
                        new_terms.remove(term)
                terms += new_terms
                merge_check = new_terms
            else:
                break
        return Or(*terms)

    def _get_definite_positive_predictions(self, X):
        """
        Get the predictions that are definitely positive.
        
        This outputs 1 if the prediction is known to be positive,
        and 0 if the prediction is negative or unknown.
        
        Column names need to be in the same format as the atoms in the DNF.

        Assumes X is dtype boolean
        """
        return self._get_definite_trues_from_dnf(X, self.all_pos_terms)

    def _get_definite_negative_predictions(self, X):
        """
        Get the predictions that are definitely negative.
        
        This outputs 0 if the prediction is known to be negative,
        and 1 if the prediction is positive or unknown.
        
        Column names need to be in the same format as the atoms in the DNF.

        Assumes X is dtype boolean
        """
        return 1 - self._get_definite_trues_from_dnf(X, self.all_neg_terms)

    def predict(self, X):
        """Predict using the DNF of the tree.

        Args:
            X (pd.DataFrame): The data to predict on.

        Assumes that X can be cast to a boolean DataFrame.
        """
        X_ = X.copy()
        X_.columns = [f'feature_{i}' for i in range(X_.shape[1])]

        # cast X_ to a boolean DataFrame
        X_ = X_.astype('boolean')

        definite_pos_preds = self._get_definite_positive_predictions(X_)
        definite_neg_preds = self._get_definite_negative_predictions(X_)
        # a prediction is only unknown if the positive pred is not 1 and the negative pred is not 0
        unknown_mask = (definite_pos_preds == 0) & (definite_neg_preds == 1)

        preds = pd.Series(definite_pos_preds, index=X_.index, dtype='Int64')
        preds[unknown_mask] = pd.NA
        return preds


class PCCTree(TreeClassifierWrapper):
    '''
    Instead of describing each tree in the Rashomon set as a tree, 
    we can describe the Rashomon set by the set of all possible 
    conditions for us to classify a sample with a certain label. 
    We call this form a partial concept class form.

    Example: 
    Consider the following tree. 

    If feature 1 true predict True
    else if feature 2 true predict True
    else predict False

    This tree would be compressed into the following set of conditions:
    Ways to predict true: 
    - (1 true, 2 anything)
    - (2 true, 1 anything)
    Ways to predict false:
    - (1 false, 2 false)

    This allows us to: 
    - assess more fully whether a missing value really needs to affect a tree's prediction
    under an MAR assumption (In the example above, even if 1 is missing, we can still 
    know the tree will predict True if 2 is true)
    - compress the Rashomon set, since the above example tree will be equivalent to another rashomon set member, 
    If feature 2 true predict True
    else if feature 1 true predict True
    else predict False
    and we'll be able to see this because both trees will have the same decision set.
    '''

    def __init__(self, source, encoder=None, X=None, y=None):
        super(TreeClassifierWrapper, self).__init__(source, encoder=encoder, X=X, y=y)
        true, false = pcc_representation(source)
        self.true_pccs = frozenset(true)
        self.false_pccs = frozenset(false)

    def get_true_pccs(self):
        '''
        Method to return the true PCCs of the PCC
        '''
        return self.true_pccs

    def get_false_pccs(self):
        '''
        Method to return the false PCCs of the PCC
        '''
        return self.false_pccs

    def __hash__(self):
        '''
        Method to hash a PCC. 

        Returns: the hash of the PCC
        '''
        return hash((self.true_pccs, self.false_pccs))

    def __eq__(self, other):
        '''
        Method to check if two PCCTrees are equivalent. 

        Args:
        - other: another PCCTree object

        Returns: True if the two PCCs are equivalent, False otherwise

        Note: 
        - assumes both PCCs satisfy standard PCC Tree invariants (see pcc.py, merge method)
        - does not require the order of the PCCs in the list to be the same.
        (TODO: prove that this method always reports equivalence correctly 
               given the PCCs satisfy the standard invariants) 
        '''
        return self.true_pccs == other.true_pccs and self.false_pccs == other.false_pccs

    def predict(self, X: pd.DataFrame):
        '''
        Method to predict the labels of a set of samples. 

        Requires
        ---
        the set of features used should be pre-encoding if an encoder is used

        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

        Returns
        ---
        array-like, shape = [n_samples by 1] : a column where each element is the prediction associated with each row 
        '''
        # cast X_ to a boolean DataFrame
        X_ = X.astype('boolean')

        return X_.apply(self.predict_row, axis=1)

    def predict_row(self, row: pd.Series):
        '''
        Method to predict the label of a single sample.

        Args:
        - row: a pandas series representing a sample
                    (should be boolean)
        
        Returns: the prediction for the sample
        '''
        for pcc in self.true_pccs:
            if pcc.covers(row):
                return 1
        for pcc in self.false_pccs:
            if pcc.covers(row):
                return 0
        return np.nan

    def __str__(self):
        output = 'Ways to predict True:\n'
        for pcc in self.true_pccs:
            output += 'If ' + str(list(pcc.get_true_features())) + ' are all true and ' + str(
                list(pcc.get_false_features())) + ' are all false, predict True\n'

        output += 'Ways to predict False:\n'
        for pcc in self.false_pccs:
            output += 'If ' + str(list(pcc.get_true_features())) + ' are all true and ' + str(
                list(pcc.get_false_features())) + ' are all false, predict False\n'

        return output


def create_tree_classifier(source, encoder=None, X=None, y=None, tree_type='DNF'):
    if tree_type == 'DNF':
        return DNFTree(source, encoder=encoder, X=X, y=y)
    if tree_type == 'DNF_exhaustive':
        return ExhaustiveDNFTree(source, encoder=encoder, X=X, y=y)
    if tree_type == 'IttyBitty':
        return IttyBittyTree(source, encoder=encoder, X=X, y=y)
    if tree_type == 'PCC':
        return PCCTree(source, encoder=encoder, X=X, y=y)
    if tree_type == 'Conservative':
        return ConservativeTree(source, encoder=encoder, X=X, y=y)
    return TreeClassifierWrapper(source, encoder=encoder, X=X, y=y)