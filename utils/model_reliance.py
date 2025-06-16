import pandas as pd
from rid.rashomon_vi_utils import perturb_shuffle
from sklearn.metrics import accuracy_score

def get_mr(dataset, predictor, n_iters=30, metric=accuracy_score):
    res = {
        'var': [],
        'mr': [],
        'mr_og_metric': [],
        'mr_pert_metric': [],
        'iter': []
    }
    og_metric = metric(dataset.values[:, -1], predictor(dataset.values[:, :-1]))
    
    all_vars = dataset.columns[:-1]
    
    for i in range(n_iters):
        for col_ind, overall_col in enumerate(all_vars):
            col_name = overall_col
            col = col_ind

            perturbed_X, perturbed_Y = perturb_shuffle(dataset.copy(), target_cols=[col_name])
            pert_metric = metric(perturbed_Y.values, predictor(perturbed_X.values))

            res['var'] = res['var'] + [col_name]
            res['mr_og_metric'] = res['mr_og_metric'] + [og_metric]
            res['mr_pert_metric'] = res['mr_pert_metric'] + [pert_metric]
            res['mr'] = res['mr'] + [og_metric - pert_metric]
            res['iter'] = res['iter'] + [i]

    return pd.DataFrame(res).groupby(['var']).mean()