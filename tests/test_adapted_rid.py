from rid.rashomon_importance_distribution import RashomonImportanceDistribution as RID
from models.treefarms_wrapper import DEFAULT_TF_CONFIG
import pandas as pd
import numpy as np

def test_rid_construction():
    dummy_data = pd.DataFrame({
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 1, 0],
        "y": [0, 0, 1, 1]
    })
    tf_config = DEFAULT_TF_CONFIG
    tf_config['depth_budget'] = 3

    np.random.seed(0)
    rid = RID(
        dummy_data, 
        n_resamples=5, 
        tf_config=tf_config, 
        binning_map={0:['x1'], 1:['x2']}, 
        binning_fn=lambda x: x, 
        use_test_df=False
    )

    assert rid.mean(0) > rid.mean(1)

    rid = RID(
        dummy_data, 
        vi_metric='sub_cmr',
        n_resamples=2, 
        tf_config=tf_config, 
        binning_map={0:['x1'], 1:['x2']}, 
        binning_fn=lambda x: x, 
        use_test_df=False
    )

    assert rid.mean(0) > rid.mean(1)