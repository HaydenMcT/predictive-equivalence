# import random forest
from models.qlearner import *
from models.tree_classifier_wrapper import DNFTree
from models.treefarms_wrapper import DEFAULT_TF_CONFIG, construct_tree_rset
import sys

from preprocessing.data import load_data, prepare_binarized_data, get_complete_subset
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from models.ensemble_wrapper import sklearn_tree_to_dict
from models.tree_classifier_wrapper import create_tree_classifier
from models.sklearn_cost_sens import CostSensitiveDecisionTree, correct_cost_sens_dict
from pystreed import STreeDCostSensitiveClassifier, STreeDClassifier
from pystreed.data import CostSpecifier
from tqdm import tqdm
import time
import warnings

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
        else: 
            node['feature'] = node_obj.feature
            node['threshold'] = 0 # threshold should always be 0 for binarized data; TODO verify
            node['false'] = _recurse_to_dict(node_obj.left_child, node_id+1)
            node['true'] = _recurse_to_dict(node_obj.right_child, 2*(node_id+1))
        return node

    return _recurse_to_dict(tree, 0)


def generate_tmp_cost_csv(feature_to_cost, column_names, bin_to_original, cost_sens_scalar=0, filename="tmp.csv"):
    """
    ftr_to_bin_range should be {ftr: (begin, end), ftr: (begin, end)...}
    """
    pd.DataFrame({
        "Attribute-name": [ftr.replace(" ", "_") for ftr in column_names],
        "test-cost": [feature_to_cost[bin_to_original(ftr)] * cost_sens_scalar for ftr in column_names],
        "discount": [feature_to_cost[bin_to_original(ftr)] * cost_sens_scalar for ftr in column_names], # only charge for each continuous variable once
        "group": ['a'] * len(column_names),
        "binarize-begin": [i for i in range(len(column_names))],
        "binarize-end": [i for i in range(len(column_names))],
    }).to_csv(filename, index=False, sep=" ")
    
    return filename


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def eval_cost_on_dataset(model, X, trained_qlearner):
    column_names = X.columns
    n_vars = len(column_names)

    # Calculate the mean cost over rows of always purchasing
    # every single variable used by the tree
    all_purchased_vars = model.get_variables_for_prediction(X)
    all_purchased_vars_flattened = set()
    max_cost_for_tree = 0
    for r in range(X.shape[0]):
        all_purchased_vars_flattened = all_purchased_vars_flattened.union(set(all_purchased_vars[r]))

    ternary_state = ''.join(['2'] * n_vars)
    for var in tqdm(all_purchased_vars_flattened, desc="Max cost"):
        max_cost_for_tree += trained_qlearner.cost_by_var(var, cur_ternary_state=ternary_state)

    # Calculate the mean cost over rows of naively purchasing variables
    # according to the tree
    all_purchased_vars = model.get_variables_for_prediction(X)
    mean_cost_for_base = 0
    x_vals = X.values
    for r in tqdm(range(X.shape[0]), desc="Mean cost"):
        ternary_state = ''.join(['2'] * n_vars)
        for var in all_purchased_vars[r]:
            mean_cost_for_base += trained_qlearner.cost_by_var(var, cur_ternary_state=ternary_state)
            ternary_state_list = list(ternary_state)
            ternary_state_list[var] = str(x_vals[r][var])
            ternary_state = ''.join(ternary_state_list)
    mean_cost_for_base /= X.shape[0]

    # Given our trained RL model, iterate over each row in the dataset and
    # iteratively purchase variables until we arrive at a prediction
    all_costs = []
    for r in tqdm(range(X.shape[0]), desc="Optimized"):
        row = x_vals[r]
        ternary_state = ''.join(['2'] * n_vars)
        total_cost = 0
        if trained_qlearner.is_terminal(ternary_state):
            break
        for _ in range(100):
            state = from_ternary(ternary_state)
            # Decide which variable to buy next
            recommended_action = trained_qlearner.recommend_action_for_state(state)
            # Record the cost of that action
            total_cost += trained_qlearner.cost_by_var(recommended_action, ternary_state)
            # And decide on the resulting state
            observed_val = row[recommended_action]
            ternary_state = ternary_state[:recommended_action] + f'{observed_val}' + ternary_state[recommended_action+1:]
            if trained_qlearner.is_terminal(ternary_state):
                break
        all_costs.append(total_cost)
    mean_cost_optimized = sum(all_costs) / X.shape[0]

    # Evaluate the cost of naively purchasing cheapest first
    all_costs = []
    action_by_cost = sorted(
        trained_qlearner.reasonable_actions,
        key=lambda x: trained_qlearner.cost_by_var(x, ''.join(['2'] * n_vars))
    )
    for r in tqdm(range(X.shape[0]), desc="Greedy"):
        row = x_vals[r]
        ternary_state = ''.join(['2'] * n_vars)
        total_cost = 0
        if trained_qlearner.is_terminal(ternary_state):
            break
        for var_ind in range(100):
            state = from_ternary(ternary_state)
            # Decide which variable to buy next
            recommended_action = action_by_cost[var_ind]
            # Record the cost of that action
            total_cost += trained_qlearner.cost_by_var(recommended_action, ternary_state)
            # And decide on the resulting state
            observed_val = row[recommended_action]
            ternary_state = ternary_state[:recommended_action] + f'{observed_val}' + ternary_state[recommended_action+1:]
            if trained_qlearner.is_terminal(ternary_state):
                break
        all_costs.append(total_cost)
    mean_cost_greedy = sum(all_costs) / X.shape[0]

    return mean_cost_for_base, mean_cost_optimized, max_cost_for_tree, mean_cost_greedy

def run_cost_sensitive_comparison(i, use_cost_sens=False, use_streed=False, cost_sens_scalar=1):
    """
    Run the full cost optimization experiment.
    Args:
        use_cost_sens (bool): If true, use a greedy cost sensitive decision tree
            as our base decision tree
        use_streed (bool): If true, use a pystreed to fit an optimal cost sensitive decision tree
            as our base decision tree
        cost_sens_scalar (float): The weight to assign to variable cost relative to classification
            error
    """
    if use_cost_sens and use_streed:
        warnings.warn("Both use_cost_sens and use_streed are true -- pystreed will be used.")

    # NOTE: For hyperparameter exploration, add these terms as outer loops
    # for exploration_rate in [0.3, 0.5, 0.7]:
    #     for alpha in [0.05, 0.1, 0.2]:
    #         for gamma in [0.75, 0.9, 0.95]:
    results = pd.DataFrame()
    for dataset in [
        "higgs.csv", "coupon_full.csv", 'wine_quality.csv', 'compas_complete.csv', 'wisconsin.csv',
        "broward_general_2y.csv", 'fico_complete.csv', 'iris_setosa.csv', 'iris_versicolor.csv',
        'iris_virginica.csv', 'spiral.csv', 'tic-tac-toe.csv', 'netherlands.csv'
    ]:
        binarization_config = {
            "missing_values": [-7, -8, -9] if 'fico' in dataset else np.nan,
            "columns_to_drop": ['ExternalRiskEstimate'] if 'fico' in dataset else None,
            "num_quantiles": 2,
            "gosdt_guesses": False,
            "keep_missingness_after_binarization": False
        }
        X_all, y_all = load_data(dataset, balance=True)
        X_train, y_train, X_test, y_test, bin_to_original_dict = prepare_binarized_data(
            X_all, 
            y_all, 
            **binarization_config
        )
        X_train, y_train = get_complete_subset(X_train, y_train)
        X_test, y_test = get_complete_subset(X_test, y_test)
        bin_to_original = lambda x: bin_to_original_dict[x]

        # For each model in our Rashomon set, check our gain in cost efficacy
        column_names = X_train.columns
        n_vars = len(column_names)
        original_col_names = [bin_to_original(c) for c in column_names]
        cost_by_var_unbinned = {c: np.random.randint(1, 10) for c in original_col_names}
        
        if use_streed:
            filename = generate_tmp_cost_csv(
                cost_by_var_unbinned, 
                column_names, 
                bin_to_original, 
                cost_sens_scalar=cost_sens_scalar,
                filename=f"/home/users/jcd97/code/rashomon-submodels/experiments/configs/tmp_{i}_{cost_sens_scalar}.csv"
            )
            misclass_cost = 1
            line_prepender(filename, f"{int(misclass_cost)} 0")
            line_prepender(filename, f"0 {int(misclass_cost)}")

            cs = CostSpecifier(filename, 2)
            m = STreeDCostSensitiveClassifier(max_depth = 3, random_seed=i)
        elif use_cost_sens:
            m = CostSensitiveDecisionTree(
                [cost_by_var_unbinned[bin_to_original(c)] for c in column_names],
                feature_cost_scalar=cost_sens_scalar,
                max_depth=3
            )
        else:
            m = DecisionTreeClassifier(
                random_state=i,
                max_depth=3
            )
        cur_X_train, cur_y_train = resample(X_train, y_train, random_state=i)
        if use_streed:
            m.fit(cur_X_train, cur_y_train, cs)
            tree_dict = pystreed_tree_to_dict(m.tree_)
        else:
            m.fit(cur_X_train, cur_y_train)
            if type(m) is DecisionTreeClassifier:
                tree_dict = sklearn_tree_to_dict(m.tree_)
            else:
                tree_dict = correct_cost_sens_dict(m.tree)
        m = create_tree_classifier(tree_dict, tree_type='DNF')

        saved_states = {}
        
        def is_terminal(state):
            """
            A function specifying which states have completed a DNF clause
            """
            if type(state) is int:
                state = to_ternary(state)
                # Prepend our ternary state with 0's if necessary
                if len(state) < n_vars:
                    state = ''.join(['0'] * (n_vars - len(state))) + state

            if state in saved_states:
                return saved_states[state]

            def get_val_at_index(i):
                if state[i] == '2':
                    return np.nan
                elif state[i] == '1':
                    return 1
                elif state[i] == '0':
                    return 0

            # Create a sample specifying only the values we've purchased
            X_tmp = pd.DataFrame(
                np.array(
                    [[get_val_at_index(i) for i in range(len(column_names))]]
                ), 
                columns=column_names, 
                dtype='boolean'
            )
            # If we can form a prediction with the stuff we've bought, this is
            # a terminal state
            saved_states[state] = not m.predict(X_tmp).isna()[0]
            return saved_states[state]

        saved_state_costs = {}

        def cost_by_var(target_var, cur_ternary_state):
            target_original_var = bin_to_original(column_names[target_var])
            if target_var in saved_state_costs:
                if cur_ternary_state in saved_state_costs[target_var]:
                    return saved_state_costs[target_var][cur_ternary_state]

            for idx, v in enumerate(cur_ternary_state):
                # If there exists another bin on this variable that we already know, this condition
                # makes it free to purchase any other bins, since presumably we would measure an input
                # variable rather than an individual bin
                if v != '2' and target_original_var == bin_to_original(column_names[idx]):
                    if target_var in saved_state_costs:
                        saved_state_costs[target_var][cur_ternary_state] = 0
                    else:
                        saved_state_costs[target_var] = {cur_ternary_state: 0}
                    return 0
            
            if target_var in saved_state_costs:
                saved_state_costs[target_var][cur_ternary_state] = cost_by_var_unbinned[target_original_var]
            else:
                saved_state_costs[target_var] = {cur_ternary_state: cost_by_var_unbinned[target_original_var]}
            return cost_by_var_unbinned[target_original_var]

        print("About to init learner")

        all_purchased_vars = m.get_variables_for_prediction(cur_X_train)
        all_purchased_vars_flattened = set()
        for r in tqdm(range(cur_X_train.shape[0]), desc="Purchased vars"):
            all_purchased_vars_flattened = all_purchased_vars_flattened.union(set(all_purchased_vars[r]))

        all_purchased_vars_flattened = np.array(list(all_purchased_vars_flattened))

        qlearner = QLearner(
            cost_by_var=cost_by_var,
            is_terminal=is_terminal,
            reasonable_actions=np.unique(all_purchased_vars_flattened),
            n_vars=n_vars,
            # NOTE: For hyperparameter exploration, uncomment the following 3 lines
            # alpha=alpha,
            # gamma=gamma,
            # exploration_rate=exploration_rate,
            max_cost=sum([cost_by_var_unbinned[cur_key] for cur_key in cost_by_var_unbinned])
        )

        print("Warm starting Q")
        qlearner.prefill_based_on_tree(tree_dict, ''.join(['2'] * qlearner.n_vars), cur_X_train)

        # Fit our RL model
        print("About to fit")
        qlearner.fit(10000, 2*len(np.unique(all_purchased_vars_flattened)), X_train=cur_X_train)
        print("Done fitting")

        mean_cost_for_base, mean_cost_optimized, max_cost_for_tree, mean_cost_greedy = eval_cost_on_dataset(m, cur_X_train, qlearner)
        print(f"TRAIN Cost with base tree: {mean_cost_for_base}; Cost with optimized: {mean_cost_optimized}; max would be {max_cost_for_tree}")
        results = pd.concat([
            results,
            pd.DataFrame({
                "dataset": [dataset.split('.')[0]],
                "naive_cost": [mean_cost_for_base],
                "max_cost_for_tree": [max_cost_for_tree],
                "greedy_cost_for_tree": [mean_cost_greedy],
                "optimized_cost": [mean_cost_optimized],
                "split": ["train"],
                "cost_by_var": [cost_by_var_unbinned],
                "tree_accuracy": [(m.predict(X_test) == y_test).mean()],
                "use_cost_sens": [use_cost_sens],
                "cost_sens_scalar": [cost_sens_scalar]
            })
        ])

        mean_cost_for_base, mean_cost_optimized, max_cost_for_tree, mean_cost_greedy = eval_cost_on_dataset(m, X_test, qlearner)
        print(f"TEST Cost with base tree: {mean_cost_for_base}; Cost with optimized: {mean_cost_optimized}; max would be {max_cost_for_tree}")
        results = pd.concat([
            results,
            pd.DataFrame({
                "iter": [i],
                "dataset": [dataset.split('.')[0]],
                "naive_cost": [mean_cost_for_base],
                "max_cost_for_tree": [max_cost_for_tree],
                "greedy_cost_for_tree": [mean_cost_greedy],
                "optimized_cost": [mean_cost_optimized],
                "split": ["test"],
                "cost_by_var": [cost_by_var_unbinned],
                "tree_accuracy": [(m.predict(X_test) == y_test).mean()],
                "use_cost_sens": [use_cost_sens],
                "cost_sens_scalar": [cost_sens_scalar]
            })
        ])

        del saved_states, m, qlearner, tree_dict
        results.to_csv(f'experiments/output/cost_optimization_{i}_warm_start_cost_{cost_sens_scalar}_6_25.csv', index=False)


if __name__ == "__main__":
    np.random.seed(0)
    slurm_id = int(sys.argv[1])
    # for i in range(50):
    for cost_sens_scalar in [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        run_cost_sensitive_comparison(slurm_id, use_streed=True, cost_sens_scalar=cost_sens_scalar)
