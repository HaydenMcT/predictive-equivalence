import numpy as np
from gosdt._tree import Leaf

def sklearn_tree_to_dict(tree) -> dict:
    def _recurse_to_dict(node_id):
        node = {}
        node['id'] = node_id
        ####################
        #features needed for dnf_trees: 
        node['reference'] = 'true' #queries assumed to always correspond to whether a feature is True; 
                                   # since we're expecting to be dealing with binarized data always with
                                   # thresholds of 0, where right/true corresponds to > 0, this should always hold
        node['relation'] = '=='
        ####################
        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        if left_id == -1 and right_id == -1:
            node['prediction'] = tree.value[node_id].argmax()
        else: 
            node['feature'] = tree.feature[node_id]
            node['threshold'] = tree.threshold[node_id] # threshold should always be 0 for binarized data; TODO verify
            if left_id != -1:
                node['false'] = _recurse_to_dict(left_id)
            if right_id != -1:
                node['true'] = _recurse_to_dict(right_id)
        return node

    return _recurse_to_dict(0)


def dl85_to_dict(node):
    '''
    Convert a dl85 tree to a dictionary representation.
    This dictionary representation allows constructing
    tree classifier wrappers from the 
    tree_classifier_wrapper.py file.
    '''
    dict_tree = {}
    ####################
    #features needed for dnf_trees:
    dict_tree['reference'] = 'true'  #queries assumed to always correspond to whether a feature is True;
    # since we're expecting to be dealing with binarized data always with
    # thresholds of 0, where left/true corresponds to > 0, this should always hold
    dict_tree['relation'] = '=='
    ####################
    if 'value' in node:
        dict_tree['prediction'] = node['value']
        return dict_tree
    else:
        dict_tree['threshold'] = 0.5
        dict_tree['feature'] = node['feat']
        dict_tree['true'] = dl85_to_dict(node['left'])
        dict_tree['false'] = dl85_to_dict(node['right'])
        return dict_tree
        
def _gosdt_to_dict(node, next_node_id=0):
    '''
    Convert a GOSDT tree to a dictionary representation.
    This dictionary representation allows constructing
    tree classifier wrappers from the 
    tree_classifier_wrapper.py file.
    '''
    dict_tree = {}
    ####################
    #features needed for dnf_trees:
    dict_tree['reference'] = 'true'  #queries assumed to always correspond to whether a feature is True;
    # since we're expecting to be dealing with binarized data always with
    # thresholds of 0, where left/true corresponds to > 0, this should always hold
    dict_tree['relation'] = '<='
    dict_tree['id'] = next_node_id
    ####################
    if isinstance(node, Leaf):
        dict_tree['prediction'] = node.prediction
        return dict_tree, next_node_id
    else:
        dict_tree['threshold'] = 0.5
        dict_tree['feature'] = node.feature
        dict_tree['false'], right_node_id = _gosdt_to_dict(node.right_child, next_node_id + 1)
        dict_tree['true'], left_node_id = _gosdt_to_dict(node.left_child, right_node_id + 1)
        return dict_tree, left_node_id

def gosdt_to_dict(tree):
    dict_tree, _ = _gosdt_to_dict(tree)
    return dict_tree

def pystreed_tree_to_dict(tree) -> dict:
    def _recurse_to_dict(node_obj, node_id):
        node = {}
        node['id'] = node_id
        ####################
        #features needed for dnf_trees:
        node['reference'] = 'true' #queries assumed to always correspond to whether a feature is True;
                                   # since we're expecting to be dealing with binarized data always with
                                   # thresholds of 0, where right/true corresponds to > 0, this should always hold
        node['relation'] = '=='
        ####################
        if node_obj.is_leaf_node():
            node['prediction'] = node_obj.label
            largest_id_in_subtree = node_id
        else:
            node['feature'] = node_obj.feature
            node['threshold'] = 0 # threshold should always be 0 for binarized data; TODO verify
            node['false'], largest_id_in_right_child = _recurse_to_dict(node_obj.right_child, node_id+1)
            node['true'], largest_id_in_subtree = _recurse_to_dict(node_obj.left_child, largest_id_in_right_child + 1) #TODO verify left/right
        return node, largest_id_in_subtree

    return _recurse_to_dict(tree, 0)[0]