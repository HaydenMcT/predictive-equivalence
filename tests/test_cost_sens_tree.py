from models.sklearn_cost_sens import CostSensitiveDecisionTree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

def test_cost_sens_equivalence():
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(100, 5), columns = ['a', 'b', 'c', 'd', 'e'])
    y = pd.Series(1 * ((X.iloc[:, 0] + X.iloc[:, 1]) > 1))


    t1 = DecisionTreeClassifier()
    t2 = CostSensitiveDecisionTree(feature_costs=np.zeros(5), feature_cost_scalar=0)

    t1.fit(X, y)
    t2.fit(X, y)
    assert (t1.predict(X) == t2.predict(X)).all()


def test_cost_sens_nonequivalence():
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(100, 5), columns = ['a', 'b', 'c', 'd', 'e'])
    y = pd.Series(1 * ((X.iloc[:, 0] + X.iloc[:, 1]) > 1))
    costs = np.zeros(5)
    costs[0] = 1e5
    costs[1] = 1e5

    t1 = DecisionTreeClassifier()
    t2 = CostSensitiveDecisionTree(feature_costs=costs, feature_cost_scalar=1.0)

    t1.fit(X, y)
    t2.fit(X, y)

    assert not (t1.predict(X) == t2.predict(X)).all()