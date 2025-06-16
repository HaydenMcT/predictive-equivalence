import copy
import pandas as pd

class PCC(): 
    '''
    A partial concept class is a partial specification of 
    feature values, containing information about 
    two sets: 
        a set of features that need to be true, 
        a set of features that need to be false. 
    It leaves as unspecified the features that can be either true or false.
    '''
    def __init__(self, true_features: set, false_features: set):
        self.true_features = frozenset(true_features)
        self.false_features = frozenset(false_features)
    
    def __eq__(self, other):
        '''
        Two partial concept classes are equal if they have the same true and false sets
        '''
        return self.true_features == other.true_features and self.false_features == other.false_features  

    def __hash__(self):
        return hash((self.true_features, self.false_features))
    
    def get_true_features(self):
        return self.true_features
    
    def get_false_features(self):
        return self.false_features

    def is_subsumed(self, other_PCC): 
        '''
        A PCC is subsumed by another PCC if both the true and false sets
        of the other PCC are subsets of this PCC

        Args:
        - other_PCC: another PCC

        Returns: True if this PCC is subsumed by the other PCC, False otherwise
        '''
        return self.true_features.issuperset(other_PCC.true_features) and self.false_features.issuperset(other_PCC.false_features)

    def covers(self, sample: pd.Series): 
        '''
        Given a training sample, evaluate whether the sample satisfies the PCC
        If we cannot evaluate whether the sample satisfies the PCC, we return False
            (whether or not that's due to NA)

        Args:
        - sample: a pandas series representing a sample
                    (should be boolean)
        '''
        return (sample.iloc[sorted(self.true_features)].fillna(False).all() 
                and not sample.iloc[sorted(self.false_features)].fillna(True).any())

def tree_to_leaf_pccs(tree_json, true_features = set(), false_features = set()):
    '''
    Given a (sub)tree in json format, returns PCCs for each leaf 
    This will be an equivalent classifier to the tree, 
    represented as a set of conditions for us to classify a sample with a certain label, 
    where each condition is a PCC
    
    Output Invariants: 
    - Any possible sample will satisfy at least one condition, and if it satisfies multiple, these conditions
    cannot be contradictory.
    - each condition will correspond one-to-one to a leaf in the tree.

    Args:
    - tree_json: a tree in json format. Assumes the tree predicts binary classes. 
    - true_features: a set of features that must be true for the current subtree to be reached;
        must be included in each pcc's set of true features
    - false_features: a set of features that must be false for the current subtree to be reached;
        must be included in each pcc's set of false features

    Returns: a tuple (true_pccs, false_pccs) including descriptions of the true and false leaves of the tree
    '''
    true_pccs, false_pccs = set(), set()
    if 'feature' in tree_json: 
        feature = tree_json['feature']

        # get PCCs for the true branch, and add them to current pccs
        tpccs_recursive, fpccs_recursive = tree_to_leaf_pccs(tree_json['true'], true_features.union({feature}), false_features)
        true_pccs = true_pccs.union(tpccs_recursive)
        false_pccs = false_pccs.union(fpccs_recursive)

        # get PCCs for the false branch, and add them to current pccs
        tpccs_recursive, fpccs_recursive = tree_to_leaf_pccs(tree_json['false'], true_features, false_features.union({feature}))
        true_pccs = true_pccs.union(tpccs_recursive)
        false_pccs = false_pccs.union(fpccs_recursive)
    else:
        if tree_json['prediction'] == 1:
            true_pccs.add(PCC(true_features, false_features))
        else:
            false_pccs.add(PCC(true_features, false_features))
    return true_pccs, false_pccs

def merge_pccs(pccs):
    '''
    A method to take PCCs (decision sets) which all predict the same label, 
    and return a list of generalized PCCs that predict this label

    Output Invariants: 
    - We should have each condition in minimal form. Equivalently, no element in the set of true features
      can be flipped to false without changing the prediction, and no element in the set of false features 
      can be flipped to true without changing the prediction (otherwise, we could just remove said element
      and get a more general condition). 
    - The output should describe the same space as the input. That means a sample will satisfy at least 
      one condition in the output set of pccs if and only if the same is true of that sample for 
      the original pcc set. 
    
    Note that there is no invariant stating that a sample falls into exactly one condition in the output set.
    It may fall into multiple, (this is necessary so that we can keep the conditions in minimal form).   

    Args:
    - pccs: a set of PCCs

    Returns: a set of PCCs that are the generalized form of the input PCCs

    TODO: improve algorithm efficiency
    '''
    existing_pccs = copy.deepcopy(pccs)
    merge_candidate_pccs = copy.deepcopy(pccs)

    first = True # flag to differentiate behaviour for the first iteration
    update = True # flag to indicate whether we need to keep updating
    while update == True: 
        update = False
        new_pccs = []
        for pcc in existing_pccs:
            for pcc2 in merge_candidate_pccs: 
                '''
                check for overlap between existing pccs and merge candidate pccs.
                When exactly one feature is in the true set of one and the false 
                set of the other (or vice versa), we know that we can create a 
                pcc based on the union of the two pccs, minus that feature. 
                
                This is because, when the union of those two pccs is true, a sample 
                will fall into one of the two pccs regardless of the value of that 
                feature.
                '''
                overlaps = pcc.get_true_features().intersection(pcc2.get_false_features()).union(pcc.get_false_features().intersection(pcc2.get_true_features()))
                if len(overlaps) == 1:
                    #TODO: abstract this to a PCC function
                    new_pcc = PCC(pcc.get_true_features().union(pcc2.get_true_features()) - overlaps, 
                                  pcc.get_false_features().union(pcc2.get_false_features()) - overlaps)
                    if new_pcc not in existing_pccs:
                        new_pccs.append(new_pcc)
        if len(new_pccs) > 0:
            '''
            If we've found any new pccs, they may be able to be merged with existing pccs.
            So we'll want to repeat the loop, checking for new overlaps between the 
            existing pccs and the new pccs.

            We'll add the old merge candidates in with the existing pccs, and put the 
            new pccs in as merge candidates.

            (We have to do something slightly different on the first iteration, because
            we initialized the merge candidates and existing pccs as the same set, to start
            by looking for all overlaps between pccs - so the merge candidates don't have
            to be added in). 
            '''
            update = True
            if not first: 
                existing_pccs = existing_pccs.union(merge_candidate_pccs)
            merge_candidate_pccs = new_pccs
        first = False

    # at the end, we'll want to add in the last set of merge candidates
    existing_pccs = existing_pccs.union(merge_candidate_pccs)

    '''
    Now, we post-process to remove unneeded elements.

    We now have a set of pccs which includes the most general possible forms, 
    but may also include some redundant forms. To satisfy output invariant 1, 
    we'll remove any pccs that are subsumed by another pcc.

    Pcc j is subsumed by pcc i if both the true and false sets of i are subsets of j
    '''
    update = True
    while update: 
        update = False
        for i, pcc in enumerate(existing_pccs):
            for j, pcc2 in enumerate(existing_pccs):
                if (i != j and 
                    pcc.is_subsumed(pcc2)):
                        # as soon as we find one instance of a pcc being subsumed by another,
                        # we remove the subsumed pcc and break the loop, 
                        # so that we can start over with the updated list of pccs
                        # (TODO: fix algorithmic inefficiency here)
                        existing_pccs.remove(pcc)
                        update = True
                        break
            if update: 
                break
    
    return existing_pccs


def pcc_representation(tree_json: dict):
    '''
    Given a tree in json format, returns the decision set representation. 
    This will be an equivalent classifier to the tree, represented in standard
    form (see earlier documentation)

    Args:
    - tree_json: a tree in json/dict format. Assumes the tree predicts binary classes. 

    Returns: a tuple (true_pccs, false_pccs), describing all conditions for true/false 
    predictions under tree_json and satisfying the above output invariants
    '''
    true_pccs, false_pccs = tree_to_leaf_pccs(tree_json)
    true_pccs = merge_pccs(true_pccs)
    false_pccs = merge_pccs(false_pccs)
    return true_pccs, false_pccs