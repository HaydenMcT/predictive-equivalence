import numpy as np
import pandas as pd

from models.tree_classifier_wrapper import create_tree_classifier, PCCTree
from models.pcc import PCC

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
            

true_rep = set([PCC(set([10]), set()), PCC(set([30]), set()), PCC(set(), set([62]))])
false_rep = set([PCC({62}, {10, 30})])

pcc_rep = PCCTree(tree)
pcc_rep_two = PCCTree(tree_two)


def test_pcc_construction():
    assert(pcc_rep.get_true_pccs() == true_rep)
    assert(pcc_rep.get_false_pccs() == false_rep)
    assert(pcc_rep_two.get_true_pccs() == true_rep)
    assert(pcc_rep_two.get_false_pccs() == false_rep)

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
    pcc_rep_three = PCCTree(tree_three)
    assert(not pcc_rep == pcc_rep_three)
    assert(not pcc_rep_two == pcc_rep_three)

    true_rep_three = set([PCC(set([11]), set()), PCC(set([34]), set()), PCC(set(), set([58]))])
    false_rep_three = set([PCC({58}, {11, 34})])
    assert(pcc_rep_three.get_true_pccs() == true_rep_three)
    assert(pcc_rep_three.get_false_pccs() == false_rep_three)