from models.qlearner import *
from models.treefarms_wrapper import create_tree_classifier
import numpy as np
import pandas as pd

def test_learning_sanity():
    def is_terminal(state, n_vars=3):
        """
        A function specifying which states have completed a DNF clause
        """
        if type(state) is int:
            state = to_ternary(state)
            if len(state) < n_vars:
                state = ''.join(['0'] * (n_vars - len(state))) + state
        
        if state[0] == '1':
            return True
        elif state[1] == '0' and state[2] == '0':
            return True
        elif '2' in state:
            return False
        else:
            return True

    np.random.seed(0)
    n_vars = 3
    probability_by_var = [0.5] * n_vars
    cost_by_var = lambda x, _: [5, 1, 1][x]
    qlearner = QLearner(
        cost_by_var,
        lambda x: is_terminal(x, n_vars),
        probability_by_var=probability_by_var,
        n_vars=n_vars,
        max_cost=7,
        reasonable_actions=[0, 1, 2]
    )

    qlearner.fit(10000, 10)
    
    # This state represents knowing nothing
    ternary_state = '222'
    state = from_ternary(ternary_state)
    # Because we made x0 so expensive, it should never be the first purchase
    recommended_action = qlearner.recommend_action_for_state(state)
    assert recommended_action != 0, qlearner.Q

    # This state represents knowing only that x1 is 0, so we should buy x2
    ternary_state = '202'
    state = from_ternary(ternary_state)
    # Because we made x0 so expensive, it should never be the first purchase
    recommended_action = qlearner.recommend_action_for_state(state)
    assert recommended_action == 2, qlearner.Q

    # This state represents knowing only that x1 is 1, so we should buy x0
    # because the only way to avoid buying all variables is if x0 = 1
    ternary_state = '212'
    state = from_ternary(ternary_state)
    recommended_action = qlearner.recommend_action_for_state(state)
    assert recommended_action == 0, qlearner.Q


def test_qlearning_with_dnf():
    '''
    This tree predicts 1 if feature 0 is True, 
         or 0 otherwise.
    '''
    tree_json = {
        "feature": 0,
        "relation": "==",
        "reference": "true",
        "true": {
            "prediction": 1,
            "name": "Prediction"
        },
        "false": {
            "prediction": 0,
            "name": "Prediction"
        }
    }
    tree = create_tree_classifier(tree_json, tree_type='DNF')

    def is_terminal(state, column_names=['feature_0', 'feature_1']):
        """
        A function specifying which states have completed a DNF clause
        """
        if type(state) is int:
            state = to_ternary(state)
            if len(state) < n_vars:
                state = ''.join(['0'] * (n_vars - len(state))) + state

        def get_val_at_index(i):
            if state[i] == '2':
                return np.nan
            elif state[i] == '1':
                return 1
            elif state[i] == '0':
                return 0

        # X = pd.DataFrame(np.array([[1, np.nan]]), columns=["feature_0", "feature_1"], dtype='boolean')
        X = pd.DataFrame(
            np.array(
                [[get_val_at_index(0), get_val_at_index(1)]]
            ), 
            columns=column_names, 
            dtype='boolean'
        )
        return not tree.predict(X).isna()[0]

    np.random.seed(0)
    n_vars = 2
    probability_by_var = [0.5] * n_vars
    cost_by_var = lambda x, _: [1, 1][x]
    qlearner = QLearner(
        cost_by_var,
        lambda x: is_terminal(x),
        probability_by_var=probability_by_var,
        n_vars=n_vars,
        max_cost=2
    )

    qlearner.fit(1000, 10)
    
    # This state represents knowing nothing
    ternary_state = '22'
    state = from_ternary(ternary_state)
    # Given this dnf, the first purchase should always be feature 0
    recommended_action = qlearner.recommend_action_for_state(state)
    assert recommended_action == 0, f"Q table: {qlearner.Q}"

    # This state represents knowing onle feature 1
    ternary_state = '21'
    state = from_ternary(ternary_state)
    # Given this dnf, we should still purchase ftr 0 here
    recommended_action = qlearner.recommend_action_for_state(state)
    assert recommended_action == 0, f"Q table: {qlearner.Q}"

    # This state represents knowing onle feature 1
    ternary_state = '20'
    state = from_ternary(ternary_state)
    # Given this dnf, we should still purchase ftr 0 here
    recommended_action = qlearner.recommend_action_for_state(state)
    assert recommended_action == 0, f"Q table: {qlearner.Q}"