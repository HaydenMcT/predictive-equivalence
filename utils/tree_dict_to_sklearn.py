from sklearn.tree import DecisionTreeClassifier
import numpy as np

def construct_sklearn_tree_from_dict(tree_dict, n_features):
    n_nodes = [0]
    n_internal_nodes = [0]
    depth = [1]
    children_left = np.array([], dtype=np.int64)
    children_right = np.array([], dtype=np.int64)
    feature = np.array([], dtype=np.int64) 
    threshold = np.array([], dtype=np.float64)
    value = np.zeros((0, 1, 2), dtype=np.float64)
    order_nodes_were_visited = {}

    result_info = {
        'children_left': children_left,
        'children_right': children_right,
        'feature': feature,
        'threshold': threshold,
        'value': value
    }

    def _recurse_to_dict(node):
        order_nodes_were_visited[n_nodes[0]] = node['id']
        n_nodes[0] = n_nodes[0] + 1
        
        # If we're at an internal node
        if 'true' in node:
            depth[0] += 1
            # Grab the ID of right child
            result_info['children_right'] = np.concatenate([result_info['children_right'], np.array([node['true']['id']])])
            # For internal nodes, predicted value is kinda meaningless, so use 0.5
            result_info['value'] = np.concatenate([result_info['value'], 0.5 + np.zeros((1, 1, 2))])
        else:
            # If we're at a leaf, no right child -- indicate with -1
            result_info['children_right'] = np.concatenate([result_info['children_right'], np.array([-1])])
            new_value = np.zeros((1, 1, 2))
            new_value[:, :, node['prediction']] = 1
            result_info['value'] = np.concatenate([result_info['value'], new_value], axis=0)

        if 'false' in node:
            print(f"Adding {np.array([node['true']['id']])} to left")
            result_info['children_left'] = np.concatenate([result_info['children_left'], np.array([node['false']['id']])])
        else:
            result_info['children_left'] = np.concatenate([result_info['children_left'], np.array([-1])])

        if 'true' in node:
            result_info['feature'] = np.concatenate([result_info['feature'], np.array([node['feature']])])
            result_info['threshold'] = np.concatenate([result_info['threshold'], np.array([node['threshold']])])
            n_internal_nodes[0] += 1
            _recurse_to_dict(node['false'])
            _recurse_to_dict(node['true'])
        else:
            result_info['feature'] = np.concatenate([result_info['feature'], np.array([-2])])
            result_info['threshold'] = np.concatenate([result_info['threshold'], np.array([-2])])

    _recurse_to_dict(tree_dict)


    # As far as I can find, the safest way to customize the internals of
    # an SKlearn decision tree is to fit a tree that has the right number 
    # of nodes, then just fiddle with the description arrays
    num_leaves_produced = -1
    num_iters = 0
    n_samples = 10
    while num_leaves_produced != n_internal_nodes[0] + 1:
        X_dummy = np.random.rand(n_samples, n_features)
        y_dummy = np.random.randint(0, 2, size=n_samples)  
        clf = DecisionTreeClassifier(max_leaf_nodes=n_internal_nodes[0] + 1)
        clf.fit(X_dummy, y_dummy)
        num_leaves_produced = clf.get_n_leaves()
        num_iters += 1
        if num_iters % 1000 == 0:
            print(f"tried {num_iters} iters")
            n_samples *= 2
        
    tree = clf.tree_

    # In case we visited nodes in a weird order, correct this
    corrected_ordering = [None] * len(order_nodes_were_visited)
    for k in order_nodes_were_visited:
        corrected_ordering[order_nodes_were_visited[k]] = k

    tree.children_left[:] = result_info['children_left'][corrected_ordering]
    tree.children_right[:] = result_info['children_right'][corrected_ordering]
    tree.feature[:] = result_info['feature'][corrected_ordering]
    tree.threshold[:] = result_info['threshold'][corrected_ordering]
    tree.value[:] = result_info['value'][corrected_ordering]
    tree.impurity[:] = np.zeros_like(result_info['threshold']) + 0.5
    tree.n_node_samples[:] = np.zeros_like(result_info['feature'])
    tree.weighted_n_node_samples[:] = np.zeros_like(result_info['threshold']) + 0.5
    tree.n_features = n_features

    print(tree.children_left)

    clf.tree_ = tree
    clf.n_features_in_ = n_features

    return clf