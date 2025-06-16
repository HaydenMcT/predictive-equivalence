import numpy as np
import random
from tqdm import tqdm

def to_ternary(n):
    """Convert an integer to its ternary (base-3) representation."""
    if n == 0:
        return "0"
    
    ternary = []
    is_negative = n < 0
    n = abs(n)
    
    while n > 0:
        ternary.append(str(n % 3))
        n //= 3
    
    if is_negative:
        ternary.append('-')
    
    return ''.join(reversed(ternary))

def from_ternary(s):
    """Convert a ternary (base-3) representation to an integer."""
    is_negative = s[0] == '-'
    if is_negative:
        s = s[1:]
    
    decimal = 0
    for digit in s:
        decimal = decimal * 3 + int(digit)
    
    return -decimal if is_negative else decimal

class QLearner:
    """
    A simple Q-learning approach to solving our variable purchase
    problem. NOTE: Throughout this framework, I am using a ternary
    encoding where 0 means a variable is known 0, 1 means the variable
    is known 1, and 2 means the variable is unknown. So, 2102 represents
    the state X1=?, X2=1, X3=0, X4=?
        cost_by_var: A function that returns the cost of a variable given
            our current ternary state, where cost_by_var(i, state) is the 
            cost of the i-th variable given a state
        is_terminal: A function that determines whether a given state
            is terminal or not
        probability_by_var: Optional; the marginal probability of each
            variable being 1
        n_vars: The number of input variables
        alpha: The alpha used in the Q-learning update
        gamma: The gamma used in the Q-learning update
        exploration_rate: The chance of drawing a random action rather
            than the predicted best action
        eps_cost: A small value to the max reward to make buying all variables
            better than not terminating
        max_cost: The maximum possible cost in for this setting (i.e., the
            cost of purchasing every single input variable)
        reasonable_actions: A list of integers indicating which actions the 
            agent should be allowed to take
    """
    def __init__(
        self,
        cost_by_var,
        is_terminal,
        probability_by_var=None,
        n_vars=3,
        alpha=0.1,
        gamma=0.9,
        exploration_rate=0.5,
        eps_cost=1e-5,
        max_cost=1e3,
        reasonable_actions=None
    ):
        self.is_terminal = is_terminal
        self.n_vars = n_vars
        self.n_states = 3 ** n_vars
        self.n_actions = n_vars
        self.reasonable_actions = reasonable_actions
        self.probability_by_var = probability_by_var
        self.cost_by_var = cost_by_var
        # We need buying all variables to be slightly better than
        # never terminating, so add a small epsilon
        self.max_cost = max_cost + eps_cost

        print(self.n_states, self.n_actions)
        self.Q = {} # np.zeros((self.n_states, self.n_actions))  # Q-table

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.exploration_rate = exploration_rate  # Exploration rate

    def transitions(self, state, action, sample_row):
        """
        Transitions from the current state to another state
        based on a given action
        it is (max_cost - cost_to_reach_terminal)
            state: The current state we're in
            action: The variable we'd like to buy next
            sample_row: The row of hidden values that are revealed
                upon purchase
        """
        ternary_state = to_ternary(state)
        if len(ternary_state) < self.n_vars:
          ternary_state = ''.join(['0'] * (self.n_vars - len(ternary_state))) + ternary_state
        
        if self.is_terminal(state):
            return state
        else:
            # If we don't know this variable, we are allowed to buy it; otherwise,
            # ignore this action and stay in our current state
            if ternary_state[action] == '2':
                ternary_state = ternary_state[:action] + f'{sample_row[action]}' + ternary_state[action+1:]
            return from_ternary(ternary_state)

    def rewards(self, state, action):
        """
        Computes the reward from taking a given action in a given state.
        In our case, the reward is 0 if the state is not terminal; else,
        it is (max_cost - cost_to_reach_terminal)
            state: The current state we're in
            action: The variable we'd like to buy next
        """
        ternary_state = to_ternary(state)
        if len(ternary_state) < self.n_vars:
          ternary_state = ''.join(['0'] * (self.n_vars - len(ternary_state))) + ternary_state
        var_vals_dict = {
            i: int(ternary_state[i]) for i in range(self.n_vars)
        }
        # Check whether we're in a terminal state; if we are,
        # our reward is gonna be (max cost - cost_of_vars_we_bought)
        if self.is_terminal(state):
            overall_reward = self.max_cost
            return overall_reward
        else:
            return -self.cost_by_var(action, ternary_state)

    def prefill_based_on_tree(self, tree_node, running_state, running_data):
        if 'prediction' in tree_node:
            # If this node is terminal, return the full reward
            self.Q[running_state] = np.ones(self.n_actions) * self.max_cost
            return self.max_cost
        else:
            action = tree_node['feature']

            left_state = list(running_state)
            left_state[action] = '0'
            left_state = ''.join(left_state)
            left_data = running_data[running_data[running_data.columns[action]] == 0]
            l_prob = left_data.shape[0] / running_data.shape[0]
            if left_data.shape[0] > 0:
                l_cost = self.prefill_based_on_tree(tree_node['false'], left_state, left_data)
            else:
                # If no data reaches this leaf, simply set to the max possible cost
                l_cost = self.max_cost

            right_state = list(running_state)
            right_state[action] = '1'
            right_state = ''.join(right_state)
            right_data = running_data[running_data[running_data.columns[action]] == 1]
            r_prob = right_data.shape[0] / running_data.shape[0]
            if right_data.shape[0] > 0:
                r_cost = self.prefill_based_on_tree(tree_node['true'], right_state, right_data)
            else:
                r_cost = self.max_cost

            expected_reward = l_prob * l_cost + r_prob * r_cost - self.cost_by_var(action, running_state)
            self.Q[running_state] = np.ones(self.n_actions) * -self.max_cost
            self.Q[running_state][self.reasonable_actions] = 0
            self.Q[running_state][action] = expected_reward
            return expected_reward


    def fit(self, num_episodes=1000, max_episode_length=1000, X_train=None):
        """
        Runs a simple Q-learning algorithm to estimate the value of each state.
            num_episodes: The number of re-initializations to run random traversal with
            max_episode_length: If a terminal state isn't hit after max_episode_length actions,
                end the episode early
            X_train: Optionally, we can pass in a training dataset to directly infer conditional
                probabilities rather than directly using probabilities. If given, in each episode
                we pick a random row of X_train to be our 'hidden' variable values to explore
        """
        # Q-learning algorithm
        if X_train is not None:
            x_vals = X_train.values
        for episode in tqdm(range(num_episodes)):
            if X_train is None:
                # If not given a training set, build a random row of hidden values based on marginal
                # probabilities
                sample_row = [1 if np.random.rand() <= self.probability_by_var[i] else 0 for i in range(self.n_vars)]
            else:
                target_row = np.random.randint(0, X_train.shape[0])
                sample_row = x_vals[target_row, :]
            
            if X_train is None:
                state = random.randint(0, self.n_states - 1)
            else:
                state = from_ternary(''.join(['2'] * self.n_vars))
            for _ in range(max_episode_length):  # Limit episode length
                if random.random() < self.exploration_rate:
                    if self.reasonable_actions is None:
                        action = random.randint(0, self.n_actions - 1)  # Explore
                    else:
                        action = random.randint(0, len(self.reasonable_actions) - 1)  # Explore
                        action = self.reasonable_actions[action]
                else:
                    if state not in self.Q:
                        self.Q[state] = np.ones(self.n_actions) * -self.max_cost
                        self.Q[state][self.reasonable_actions] = 0
                    action = np.argmax(self.Q[state])  # Exploit

                next_state = self.transitions(state, action, sample_row)
                reward = self.rewards(state, action)

                # Q-learning update
                if self.is_terminal(next_state):
                    # Break this loop and deterministically set value for sink states
                    if next_state not in self.Q:
                        self.Q[next_state] = np.ones(self.n_actions) * -self.max_cost
                        self.Q[next_state][self.reasonable_actions] = 0
                    if state not in self.Q:
                        self.Q[state] = np.ones(self.n_actions) * -self.max_cost
                        self.Q[state][self.reasonable_actions] = 0
                    self.Q[next_state] = np.ones(self.n_actions) * self.rewards(next_state, 0)
                    self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
                    break
                else:
                    if next_state not in self.Q:
                        self.Q[next_state] = np.ones(self.n_actions) * -self.max_cost
                        self.Q[next_state][self.reasonable_actions] = 0
                    if state not in self.Q:
                        self.Q[state] = np.ones(self.n_actions) * -self.max_cost
                        self.Q[state][self.reasonable_actions] = 0
                    self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
                
                state = next_state

    def recommend_action_for_state(self, state):
        """
        Given either the ternary or integer representation of a state,
        estimate the next best purchase
            state: The state to work from
        """
        if type(state) is str:
            state = from_ternary(state)
        if state not in self.Q:
            action = random.randint(0, len(self.reasonable_actions) - 1)  # Explore
            return self.reasonable_actions[action]
        return np.argmax(self.Q[state])

if __name__ == '__main__':
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
        if state[1] == '0' and state[2] == '0':
            return True
        elif '2' in state:
            return False
        else:
            return True

    n_vars = 3
    probability_by_var = [0.5] * n_vars
    cost_by_var = lambda x, ternary_state: [100, 1, 1][x]
    qlearner = QLearner(
        probability_by_var,
        cost_by_var,
        lambda x: is_terminal(x, n_vars),
        n_vars=n_vars
    )

    qlearner.fit()

    ternary_state = '222'
    state = from_ternary(ternary_state)
    print(f"Starting state is {ternary_state}")
    for _ in range(5):
        recommended_action = qlearner.recommend_action_for_state(state)
        observed_val = 1 if np.random.rand() <= probability_by_var[recommended_action] else 0
        ternary_state = ternary_state[:recommended_action] + f'{observed_val}' + ternary_state[recommended_action+1:]
        print(f"I recommend checking {recommended_action}, which costed {cost_by_var(recommended_action, ternary_state)} and yielded {observed_val} giving state {ternary_state}")
        state = from_ternary(ternary_state)
        if is_terminal(state):
            break