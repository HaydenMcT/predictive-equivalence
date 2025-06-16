import numpy as np
import pandas as pd

from models.tree_classifier_wrapper import create_tree_classifier, PCCTree, DNFTree, ExhaustiveDNFTree
from models.pcc import PCC
from sympy import to_dnf

tree = {'feature': 10,
        'relation': '==',
        'reference': 'true',
        'true': {'prediction': 1, 'name': 'Prediction'},
        'false': {'feature': 30,
                    'relation': '==',
                    'reference': 'true',
        'true': {'prediction': 1, 'name': 'Prediction'},
        'false': {'feature': 62,
        'relation': '==',
        'reference': 'true',
        'true': {'prediction': 0, 'name': 'Prediction'},
        'false': {'prediction': 1, 'name': 'Prediction'}}}}

tree_two = {'feature': 10,
            'relation': '==',
            'reference': 'true',
            'true': {'prediction': 1, 'name': 'Prediction'},
            'false': {'feature': 62,
            'relation': '==',
            'reference': 'true',
            'true': {'feature': 30,
            'relation': '==',
            'reference': 'true',
            'true': {'prediction': 1, 'name': 'Prediction'},
            'false': {'prediction': 0, 'name': 'Prediction'}},
            'false': {'prediction': 1, 'name': 'Prediction'}}}
            

true_rep = to_dnf('feature_10 | feature_30 | ~feature_62')
false_rep = to_dnf('feature_62 & ~feature_10 & ~feature_30')

pcc_rep = ExhaustiveDNFTree(tree)
pcc_rep_two = ExhaustiveDNFTree(tree_two)


def test_construction():
    assert(pcc_rep.all_pos_terms == true_rep)
    assert(pcc_rep.all_neg_terms == false_rep)
    assert(pcc_rep_two.all_pos_terms == true_rep)
    assert(pcc_rep_two.all_neg_terms == false_rep)

def test_equivalence(): 
    assert(pcc_rep == pcc_rep_two)

def test_nonequivalence(): 
    tree_three = {'feature': 11,
                    'relation': '==',
                    'reference': 'true',
                    'true': {'prediction': 1, 'name': 'Prediction'},
                    'false': {'feature': 34,
                    'relation': '==',
                    'reference': 'true',
                    'true': {'prediction': 1, 'name': 'Prediction'},
                    'false': {'feature': 58,
                    'relation': '==',
                    'reference': 'true',
                    'true': {'prediction': 0, 'name': 'Prediction'},
                    'false': {'prediction': 1, 'name': 'Prediction'}}}}
    pcc_rep_three = ExhaustiveDNFTree(tree_three)
    assert(not pcc_rep == pcc_rep_three)
    assert(not pcc_rep_two == pcc_rep_three)

    true_rep_three = to_dnf('feature_11 | feature_34 | ~feature_58')
    false_rep_three = to_dnf('feature_58 & ~feature_11 & ~feature_34')

    assert(pcc_rep_three.all_pos_terms == true_rep_three)
    assert(pcc_rep_three.all_neg_terms == false_rep_three)

def test_correct_redundant_clause(): 
    tree = {'feature': 11,
                    'relation': '==',
                    'reference': 'true',
                    'true': {'feature': 58,
                            'relation': '==',
                            'reference': 'true',
                            'true': {'prediction': 0, 'name': 'Prediction'},
                            'false': {'prediction': 1, 'name': 'Prediction'}},
                    'false': {'feature': 34,
                    'relation': '==',
                    'reference': 'true',
                    'true': {'prediction': 1, 'name': 'Prediction'},
                    'false': {'prediction': 0, 'name': 'Prediction'}}}
    dnf_rep = ExhaustiveDNFTree(tree)
    true_rep = to_dnf('feature_11 & ~feature_58 | ~feature_11 & feature_34 | ~feature_58 & feature_34')
    false_rep = to_dnf('feature_11 & feature_58 | ~feature_11 & ~feature_34 | feature_58 & ~feature_34')

    assert(dnf_rep.all_pos_terms == true_rep)
    assert(dnf_rep.all_neg_terms == false_rep)