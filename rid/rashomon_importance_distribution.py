import os
import numpy as np
import shutil
import pickle
import time
import warnings
from gosdt import GOSDTClassifier as GOSDT
from gosdt import ThresholdGuessBinarizer
from models.treefarms_wrapper import DEFAULT_TF_CONFIG, TreeFarmsWrapper, construct_tree_rset
import pandas as pd
from .rashomon_vi_utils import get_model_reliances, get_conditional_model_reliances
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

class RashomonImportanceDistribution:
    '''
    A class to compute and interface with the
    Rashomon Importance Distribution. Note that this
    implementation makes heavy use of caching.

    Attributes
    ----------
    input_df : pd.DataFrame
        A pandas DataFrame containing a binarized
        version of the dataset we seek to explain
    tf_config : dict
        The arguments to pass to treefarms
    binning_map : dict
        A dictionary of the form
        {0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7, 8]}
        describing which variables in the unbinned version of
        input_df map to which columns of the binarized version
    dataset_name : string
        The name of the datset being analyzed. Used to determine
        where to cache various files
    n_resamples : int
        The number of bootstrap samples to compute. If None, do not bootstrap
    cache_dir_root : string
        The root file path at which all cached files should be stored
    verbose : bool
        Whether to produce extra logging
    vi_metric : string
        The VI metric to use for this RID; should be one of 
        ['sub_mr', 'div_mr', 'sub_cmr', 'div_cmr']
    max_par_for_gosdt : int
        The maximum number of instances of GOSDT to run
        in parallell; reduce this number if memory issues
        occur
    original_df : pd.DataFrame
        The original, unbinarized version of the input
        dataset. Used for computing CMR, where we need
        to perform imputation
    holdout_prop : float
        The proportion of the dataset to use for our test
        set, if using a separate test set
    use_joint : bool
        Whether to format our variable importance df for
        joint variable importance or not
    binning_fn : function
        A function that takes as input a non-binarized df,
        and outputs a version binarized using the same logic
        as was applied to create our original, binarized df
    impute_cmr : bool
        Whether to learn an imputation model to compute CMR
        or to use direct conditional probability estimates
    impute_cmr : class
        The SKLearn model to use for imputation in CMR,
        if applicable
    use_abs_eps : bool
        If true, use a constant, shared maximum loss
        across all of our Rashomon sets
    use_test_df : bool
        Whether to use a held out test set for VI estimation
        (ie, separate from the set used for Rashomon set 
        estimation)
    imputation_params : dict
        Parameters to pass to the CMR imputation model
    allow_binarize_internally : bool
        Whether to let RID use GOSDT binarization internally
    tree_type : str
        The tree type to use, as passed to the treefarms 
        wrapper
    unique_trees_only : bool
        If true, compute RID with respect to only predictively
        unique trees from TreeFarms
    flush_cache : bool
        If true, flush all cached files for this config
        before computing RID
    '''
    def __init__(self, 
            input_df,
            tf_config=DEFAULT_TF_CONFIG,
            binning_map=None,
            dataset_name='dataset_1', 
            n_resamples=100,
            cache_dir_root='./cached_files',
            verbose=False,
            vi_metric='sub_mr',
            max_par_for_gosdt=5,
            original_df=None,
            holdout_prop=0.2,
            use_joint=True,
            binning_fn=None,
            impute_cmr=True,
            imputation_model=RandomForestClassifier,
            use_abs_eps=False,
            use_test_df=True,
            imputation_params={'n_estimators':[25, 50, 100], 'max_depth': [None, 3, 5, 7]},
            allow_binarize_internally=False,
            tree_type='default',
            unique_trees_only=False,
            flush_cache=True
        ):

        assert tree_type != 'default' or not unique_trees_only, \
            "Error: We can't actually do unique only with default trees"

        supported_vis = ['sub_mr', 'div_mr', 'sub_cmr', 'div_cmr']
        assert vi_metric in supported_vis, \
            f"Error: VI metric {vi_metric} not recognized. Supported VI metrics are {supported_vis}"

        if input_df.isin([0, 1]).all().all():
            assert binning_map, "Error: Binning map must not be None if binary data is given."
            self.input_df = input_df
            self.binning_map = binning_map
        elif allow_binarize_internally:
            warnings.warn("Non-binarized data detected, binarizing internally using guesses.")
            binarized_df, binning_map = self._binarize_data_guesses(input_df)
            self.input_df = binarized_df
            self.binning_map = binning_map
        else:
            raise Exception(
                """
                Non-binarized data was given, but allow_binarize_internally is set to False. 
                If you would like RID to binarize data internally, you can specify allow_binarize_internally=True.
                Otherwise, binarize your data before passing it to RID.
                """
            )
        if original_df is None:
            original_df = self.input_df
        self.input_df = self.input_df.astype(int)

        self.vi_metric = vi_metric
        train_df, test_df, train_unbinned, test_unbinned = train_test_split(self.input_df, original_df, test_size=holdout_prop)
        self.input_df = train_df
        self.test_df = test_df
        self.original_train_df = train_unbinned
        self.original_test_df = test_unbinned

        self.use_test_df = use_test_df
        if not use_test_df:
            self.input_df = input_df
            self.original_train_df = original_df

        self.use_abs_eps = use_abs_eps
        self.use_joint = use_joint
        if n_resamples is None:
            self.model_counts = [0]
        else:
            self.model_counts = [0] * n_resamples
        self.n_vars = len(binning_map)
        self.n_resamples = n_resamples
        self.tf_config = tf_config
        self.tree_type = tree_type
        self.unique_trees_only = unique_trees_only
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.max_par_for_gosdt = max_par_for_gosdt
        
        self.binning_fn  = binning_fn
        self.imputation_params = imputation_params
        if impute_cmr:
            self.imputation_model = imputation_model
        else:
            self.imputation_model = None

        if self.use_abs_eps:
            opt_loss = self._get_opt_loss()
            self.tf_config['rashomon_bound_adder'] = 0
            self.tf_config['rashomon_bound'] = self.tf_config['rashomon_bound_adder'] + opt_loss

        try:
            self.num_cpus = os.cpu_count()
        except:
            self.num_cpus = 1

        # Create the cache directory if necessary
        if not os.path.exists(os.path.join(cache_dir_root, dataset_name)):
            os.makedirs(os.path.join(cache_dir_root, dataset_name))
        elif flush_cache:
            shutil.rmtree(os.path.join(cache_dir_root, dataset_name))
            os.makedirs(os.path.join(cache_dir_root, dataset_name))
            
        self.cache_dir = os.path.join(cache_dir_root, dataset_name)

        # First, compute each necessary bootstrapped dataset --------------
        if self.n_resamples is None:
            self._construct_bootstrap_datasets(None)
        else:
            for i in range(self.n_resamples):
                self._construct_bootstrap_datasets(i)

        # Second, compute each necessary Rashomon set ---------------------
        if self.n_resamples is not None:
            with Pool(min(self.num_cpus, self.max_par_for_gosdt)) as p:
                self.rashomon_sets = p.map(RashomonImportanceDistribution._construct_rashomon_sets, 
                    [self]*self.n_resamples,
                    [i for i in range(self.n_resamples)])
        else:
            self.rashomon_sets = [self._construct_rashomon_sets(None)]

        # Third, compute the variable importance for each ----------------
        # model in each bootstrap Rashomon set ---------------------------
        self._compute_and_aggregate_vis()

        if (self.imputation_model is not None) and ('cmr' in self.vi_metric):
            metric_str = f'{self.vi_metric}_impute'
        else:
            metric_str = self.vi_metric

        if self.n_resamples is not None:
            self.vi_dataframe = self._read_vis_to_construct_rid(
                file_paths=[os.path.join(self.cache_dir, f'{metric_str}s_{"for_joint_" if self.use_joint else ""}bootstrap_{i}.pickle') for i in range(n_resamples)],
                n_vars=self.n_vars
            )
        else:
            self.vi_dataframe = self._read_vis_to_construct_rid(
                file_paths=[os.path.join(self.cache_dir, f'{metric_str}s_{"for_joint_" if self.use_joint else ""}bootstrap_{None}.pickle')],
                n_vars=self.n_vars
            )
        self.rid_with_counts = self._get_df_with_counts()

    def _binarize_data_guesses(self, df_unbinned):
        '''
        Converts a non-binarized dataset to a binarized version
        that is compliant with RID
        Parameters
        ----------
            df_unbinned : pd.DataFrame
                The original, non-binarized dataset provided
        '''
        n_est = 40
        max_depth = 1
        X_all = df_unbinned.iloc[:, :-1]
        y = df_unbinned.iloc[:, -1]
        enc = ThresholdGuessBinarizer(n_estimators=n_est, max_depth=max_depth, random_state=42)
        enc.set_output(transform="pandas")
        X_binned_full = enc.fit_transform(X_binned_full, y)
        bin_map = enc.feature_map()

        df = pd.concat((X_binned_full, y), axis=1)

        return df, bin_map

    def _construct_bootstrap_datasets(self, bootstrap_ind):
        '''
        Constructs and stores the bootstrapped dataset for the given index

        Parameters
        ----------
            bootstrap_ind : int
                The index of the current bootstrap
        '''
        if (not os.path.isfile(os.path.join(self.cache_dir, f'tmp_bootstrap_unbinned_{bootstrap_ind}.csv'))) and self.original_train_df is not None:
            if bootstrap_ind is not None:
                resampled_df = resample(self.original_train_df, random_state=bootstrap_ind)
            else:
                resampled_df = self.original_train_df
            
            resampled_df.to_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_unbinned_{bootstrap_ind}.csv'), index=False)
            binarized_df = self.binning_fn(resampled_df.reset_index(drop=True))
            binarized_df.to_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'), index=False)

        elif not os.path.isfile(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv')):
            if bootstrap_ind is not None:
                resampled_df = resample(self.input_df, random_state=bootstrap_ind)
            else:
                resampled_df = self.input_df
            
            resampled_df.to_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'), index=False)

    def _construct_rashomon_sets(self, bootstrap_ind):
        '''
        Constructs and stores the Rashomon set for the given index

        Parameters
        ----------
            bootstrap_ind : int
                The index of the current bootstrap
        '''
        if self.verbose:
            print("Generating Rashomon set")

        df = pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'))
        tf = construct_tree_rset(
            X=df.iloc[:, :-1],
            y=df.iloc[:, -1],
            tf_config=self.tf_config,
            tree_type=self.tree_type
        )
        if self.unique_trees_only:
            print(f"Produced rset with {tf.get_unique_tree_count()} of {tf.get_tree_count()} unique")
            print(f"Found duplication rates {[v for v in tf.get_duplication_rates().values()]}")
        return tf

    def _compute_and_aggregate_vis(self):
        '''
        Computes and stores the variable importance metric
        for this RID for each model in each Rashomon set
        '''
        if 'cmr' in self.vi_metric:
            self._compute_and_aggregate_cmrs()
        elif 'mr' in self.vi_metric:
            self._compute_and_aggregate_mrs()
        elif 'shap' in self.vi_metric:
            self._compute_and_aggregate_shaps()
    
    def _compute_and_aggregate_cmrs(self):
        '''
        Constructs and stores the CMR for all models. 
        Not currently implemented.
        '''

        if self.n_resamples is not None:
            with Pool(self.num_cpus) as p:
                results_list = p.map(RashomonImportanceDistribution._get_cmrs_for_dataset, 
                    [self]*self.n_resamples,
                    [i for i in range(self.n_resamples)])
        else:
            results_list = [self._get_cmrs_for_dataset(None)]
        if self.imputation_model is not None:
            self._process_and_save_results(results_list, 'cmr_impute')
        else:
            self._process_and_save_results(results_list, 'cmr')

    def _compute_and_aggregate_mrs(self):
        '''
        Computes and stores model reliance (sub and div) for
        each model in all bootstrapped Rashomon sets
        '''
        if self.n_resamples is not None:
            with Pool(self.num_cpus) as p:
                results_list = p.map(RashomonImportanceDistribution._get_mrs_for_dataset, 
                    [self]*self.n_resamples,
                    [i for i in range(self.n_resamples)])
        else:
            results_list = [self._get_mrs_for_dataset(None)]
        self._process_and_save_results(results_list, 'mr')

    def _process_and_save_results(self, results_list, result_type='mr'):
        target_div_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        target_sub_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        
        start = time.time()
        for bootstrap_ind, val in enumerate(results_list):
            # A bit hacky, but this allows us to skip datasets for which
            # we've already found our VIs
            if val is None:
                continue

            cur_model_reliance = val[0]
            for var in range(self.n_vars):
                if self.use_joint:
                    target_div_model_reliances[var][bootstrap_ind] = cur_model_reliance[var]
                else:
                    for mr in cur_model_reliance[var].keys():
                        if mr == 'mean':
                            target_div_model_reliances[var]['means'].append(cur_model_reliance[mr])
                        elif mr in target_div_model_reliances[var].keys():
                            target_div_model_reliances[var][mr] += cur_model_reliance[var][mr]
                        else:
                            target_div_model_reliances[var][mr] = cur_model_reliance[var][mr]

            with open(os.path.join(self.cache_dir, f'div_{result_type}s_{"for_joint_" if self.use_joint else ""}bootstrap_{bootstrap_ind if self.n_resamples is not None else self.n_resamples}.pickle'), 'wb') as f:
                pickle.dump(target_div_model_reliances, f, protocol=pickle.HIGHEST_PROTOCOL)

            cur_model_reliance = val[1]
            for var in range(self.n_vars):
                if self.use_joint:
                    target_sub_model_reliances[var][bootstrap_ind] = cur_model_reliance[var]
                else:
                    for mr in cur_model_reliance[var].keys():
                        if mr == 'mean':
                            target_sub_model_reliances[var]['means'].append(cur_model_reliance[mr])
                        elif mr in target_sub_model_reliances[var].keys():
                            target_sub_model_reliances[var][mr] += cur_model_reliance[var][mr]
                        else:
                            target_sub_model_reliances[var][mr] = cur_model_reliance[var][mr]

            with open(os.path.join(self.cache_dir, f'sub_{result_type}s_{"for_joint_" if self.use_joint else ""}bootstrap_{bootstrap_ind if self.n_resamples is not None else self.n_resamples}.pickle'), 'wb') as f:
                pickle.dump(target_sub_model_reliances, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        if self.verbose:
            print("Completed final processing in {} seconds".format(time.time() - start))
    
    def _compute_and_aggregate_shaps(self):
        '''
        Computes and stores SHAP values for
        each model in all bootstrapped Rashomon sets
        Not currently implemented.
        '''
        assert False, "Error: SHAP not yet implemented"
    
    def _get_cmrs_for_dataset(self, bootstrap_ind):
        '''
        Computes and stores the conditional model reliance (sub and div) for
        each model in the Rashomon set corresponding to the given index.

        Parameters
        ----------
            bootstrap_ind : int
                The index of the current bootstrap
        '''
        assert self.original_train_df is not None, \
            "Error: CMR computation requires the original, unbinned dataset"
        if self.verbose:
            print(f"Beginning to compute CMRs for iter {bootstrap_ind}")
        
        if os.path.isfile(os.path.join(self.cache_dir, f'div_cmrs_{"for_joint_" if self.use_joint else ""}bootstrap_{bootstrap_ind}.pickle'))\
            and os.path.isfile(os.path.join(self.cache_dir, f'sub_cmrs_{"for_joint_" if self.use_joint else ""}bootstrap_{bootstrap_ind}.pickle')):
            return None

        div_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        sub_model_reliances = [{'means':[]} for i in range(self.n_vars)]

        resampled_df = pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'))
        unbinned_resampled_df = pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_unbinned_{bootstrap_ind}.csv'))
        #unbinned_resampled_df = resampled_df#pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_unbinned_{bootstrap_ind}.csv'))
        if self.verbose:
            cur_iterator = tqdm(self.binning_map)
        else:
            cur_iterator = self.binning_map
        
        for var in cur_iterator:
            def rset_pred_function(x_all):
                if self.rashomon_sets[bootstrap_ind].fit_on_unusable_data:
                    return pd.Series([self.rashomon_sets[bootstrap_ind].unusable_data_prediction] * x_all.shape[0])
                else:
                    return self.rashomon_sets[bootstrap_ind].predict_for_all_trees(x_all, self.unique_trees_only)

            tmp_div_model_reliances, tmp_sub_model_reliances, num_models = get_conditional_model_reliances(rset_pred_function, 
                        self.test_df if self.use_test_df else resampled_df, 
                        unbinned_train_df=self.original_train_df, 
                        unbinned_test_df=self.original_test_df if self.use_test_df else unbinned_resampled_df, 
                        var_of_interest=self.binning_map[var], 
                        for_joint=self.use_joint,
                        imputation_model=self.imputation_model,
                        var_of_interest_unbinned=var,
                        binning_fn=self.binning_fn,
                        parameters=self.imputation_params)
            div_model_reliances[var] = tmp_div_model_reliances
            sub_model_reliances[var] = tmp_sub_model_reliances
            
        self.model_counts[bootstrap_ind] = num_models
        return (div_model_reliances, sub_model_reliances)
        
    def _get_mrs_for_dataset(self, bootstrap_ind):
        '''
        Computes and stores the model reliance (sub and div) for
        each model in the Rashomon set corresponding to the given index.

        Parameters
        ----------
            bootstrap_ind : int
                The index of the current bootstrap
        '''
        if self.verbose:
            print(f"Beginning to compute MRs for iter {bootstrap_ind}")

        if os.path.isfile(os.path.join(self.cache_dir, f'div_mrs_{"for_joint_" if self.use_joint else ""}bootstrap_{bootstrap_ind}.pickle'))\
            and os.path.isfile(os.path.join(self.cache_dir, f'sub_mrs_{"for_joint_" if self.use_joint else ""}bootstrap_{bootstrap_ind}.pickle')):
            return None

        div_model_reliances = [{'means':[]} for i in range(self.n_vars)]
        sub_model_reliances = [{'means':[]} for i in range(self.n_vars)]

        resampled_df = pd.read_csv(os.path.join(self.cache_dir, f'tmp_bootstrap_{bootstrap_ind}.csv'))
        print("resampled_df.shape", resampled_df.shape)
        print("test_df.shape", self.test_df.shape)

        if self.verbose:
            cur_iterator = tqdm(self.binning_map, desc="Finding MR for each var: ")
        else:
            cur_iterator = self.binning_map

        for var in cur_iterator:
            def rset_pred_function(x_all):
                if self.rashomon_sets[bootstrap_ind].fit_on_unusable_data:
                    return pd.Series([self.rashomon_sets[bootstrap_ind].unusable_data_prediction] * x_all.shape[0])
                else:
                    return self.rashomon_sets[bootstrap_ind].predict_for_all_trees(x_all, self.unique_trees_only)
            tmp_div_model_reliances, tmp_sub_model_reliances, num_models = get_model_reliances(
                rset_pred_function, 
                self.test_df if self.use_test_df else resampled_df, 
                var_of_interest=self.binning_map[var], 
                for_joint=self.use_joint, 
                verbose=self.verbose,
                num_perts=5
            )
            div_model_reliances[var] = tmp_div_model_reliances
            sub_model_reliances[var] = tmp_sub_model_reliances
        
        if bootstrap_ind is not None:
            self.model_counts[bootstrap_ind] = num_models
        else:
            self.model_counts[0] = num_models
        return (div_model_reliances, sub_model_reliances)
    

    def _read_vis_to_construct_rid(self, file_paths, n_vars):
        '''
        Reads through each set of variable importance values specified by
        file_paths, combining them into the RID for this dataset

        Parameters
        ----------
            file_paths : list(string)
                A list of file paths pointing to each stored set of variable
                importance values
            n_vars : int
                The number of variables in the original version of the
                relevant dataset
        '''
        n_bootstraps = 0
        if not self.use_joint:
            combined_mrs = [{'means':[]} for i in range(n_vars)]
        else:
            combined_mrs = [{} for i in range(n_vars)]
        skips = []
        for file_path in file_paths:
            #try:
            with open(file_path, 'rb') as f:
            
                model_reliances = pickle.load(f)
                # Add the information from this R-set to one mega trie
                # For each variable
                for var in range(n_vars):
                    # For each value in ['mean', observed_mr_1, observed_mr_2, ...]
                    if not self.use_joint:
                        for key in model_reliances[var].keys():
                            if key == 'means':
                                if 'cmr' in file_path:
                                    combined_mrs[var]['means'].append(np.mean(model_reliances[var]['means']))
                                else:
                                    combined_mrs[var]['means'] = combined_mrs[var]['means'] + np.mean(model_reliances[var]['means'])
                                continue
                            # If we've already seen this MR value, add the probability in it's R-set
                            # to our running list
                            elif key in combined_mrs[var].keys():
                                if 'cmr' in file_path:
                                    combined_mrs[var][key] = combined_mrs[var][key] + model_reliances[var][key]
                                else:
                                    combined_mrs[var][key].append(model_reliances[var][key])
                            # Otherwise, start a new running list for it
                            else:
                                if 'cmr' in file_path:
                                    combined_mrs[var][key] = model_reliances[var][key]
                                else:
                                    combined_mrs[var][key] = [model_reliances[var][key]]
                    else:
                        # Each combined_mrs[var] should be a list of 500 n_models arrays
                        combined_mrs[var][n_bootstraps] = model_reliances[var][n_bootstraps]
                        # TODO: Figure out the right way to aggregate joint VIs
            # Track how many successful rashomon sets we loaded
            n_bootstraps += 1
            '''except:
                if self.verbose:
                    print(f"Skipping {file_path}")
                skips.append(file_path)
                continue'''

        '''
        combined_mrs is now a dict of the form
        {
            var_1: {
                "means": [mean_1, mean_2, ...],
                observe_mr_1: [p_1, p_2, ...],
                ...
            }
            var_2: {
                "means": [mean_1, mean_2, ...],
                observe_mr_1: [p_1, p_2, ...],
                ...
            }
        }
        '''

        # The overall probability of observing each MR is then the mean over all
        # observed datasets
        model_reliance_df = pd.DataFrame()

        if not self.use_joint:
            vars = []
            values = []
            probabilities = []
            # For each variable
            for var in range(n_vars):
                # For each observed MR
                for key in combined_mrs[var].keys():
                    if key == 'means':
                        continue
                    vars.append(var)
                    values.append(key)
                    probabilities.append(combined_mrs[var][key])

            # We now have vars, a list, values, a list, and probabilities,
            # a jagged list of lists
            model_reliance_df['var'] = vars
            model_reliance_df['val'] = values

            # Take the mean across rashomon sets for each probability
            true_probabilities = []
            for p in probabilities:
                if type(p) is not float:
                    true_probabilities.append(sum(p) / n_bootstraps)
                else:
                    true_probabilities.append(p / n_bootstraps)
            model_reliance_df['prob'] = true_probabilities

            model_reliance_df['count'] = 0
            for var in range(n_vars):
                cur_prob = model_reliance_df[model_reliance_df['var'] == var]['prob']
                model_reliance_df.loc[model_reliance_df['var'] == var, 'count'] = (cur_prob / cur_prob.min()).round().astype(int)
        else:
            model_reliance_df = combined_mrs
            
        return model_reliance_df

    def _get_df_with_counts(self):
        '''
        Gathers all computed sets of variable importances to compute
        a easily interfaced with datafram
        '''
        if not self.use_joint:
            rid_with_counts = {}
            if self.verbose:
                print("Processing ours with counts")
            for var in range(self.n_vars):
                if self.verbose:
                    print(f"Starting var {var}")
                rid_with_counts[var] = self.vi_dataframe[self.vi_dataframe['var'] == var]['val'].values
                rid_with_counts[var] = np.repeat(self.vi_dataframe[self.vi_dataframe['var'] == var]['val'].values,
                                                            self.vi_dataframe[self.vi_dataframe['var'] == var]['count'].values)
        else:
            rid_with_counts = [None] * self.n_vars
            for v in range(self.n_vars):
                if self.n_resamples is not None:
                    rid_with_counts[v] = np.concatenate([self.vi_dataframe[v][b] for b in range(self.n_resamples)])
                else:
                    rid_with_counts[v] = np.concatenate([self.vi_dataframe[v][b] for b in range(1)])
            
        return rid_with_counts


    def eval_cdf(self, k, var, mask_fn=None):
        '''
        Computes the value of the CDF for var at k

        Parameters
        ----------
            k : float
                The point in the CDF to evaluate
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        if mask_fn is not None:
            return (self.rid_with_counts[var][mask_fn(self.rid_with_counts)] <= k).mean()
        else:
            return (self.rid_with_counts[var] <= k).mean()

    def mean(self, var, mask_fn=None):
        '''
        Computes the mean variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        if mask_fn is not None:
            return self.rid_with_counts[var][mask_fn(self.rid_with_counts)].mean()
        else:
            return self.rid_with_counts[var].mean()

    def median(self, var, mask_fn=None):
        '''
        Computes the median variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        if mask_fn is None:
            return np.median(self.rid_with_counts[var])
        else:
            return np.median(self.rid_with_counts[var][mask_fn(self.rid_with_counts)])

    def range(self, var, mask_fn=None):
        '''
        Computes the median variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        if mask_fn is None:
            return (np.min(self.rid_with_counts[var]), np.max(self.rid_with_counts[var]))
        else:
            return (np.min(self.rid_with_counts[var][mask_fn(self.rid_with_counts)]), 
                    np.max(self.rid_with_counts[var][mask_fn(self.rid_with_counts)]))

    def bwr(self, var, mask_fn=None):
        '''
        Computes the box and whiskers range of variable importance for var
        Parameters
        ----------
            var : int
                The variable for which we want to
                evaluate the CDF
        '''
        if mask_fn is None:
            lower_q = np.quantile(self.rid_with_counts[var], 0.25)
            upper_q = np.quantile(self.rid_with_counts[var], 0.75)
        else:
            lower_q = np.quantile(self.rid_with_counts[var][mask_fn(self.rid_with_counts)], 0.25)
            upper_q = np.quantile(self.rid_with_counts[var][mask_fn(self.rid_with_counts)], 0.75)
            
        iqr = upper_q - lower_q
        return (lower_q - 1.5*iqr, upper_q + 1.5*iqr)

    def _get_opt_loss(self):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(self.input_df.values[:, :-1], self.input_df.values[:, -1])
        warm_labels = clf.predict(self.input_df.values[:, :-1])

        # train GOSDT model
        config = {
                    "regularization": self.lam,
                    "depth_budget": self.db,
                    "time_limit": 60,
                    "similar_support": False
                }

        model = GOSDT(**config)

        model.fit(self.input_df.iloc[:, :-1], self.input_df.iloc[:, -1], y_ref=warm_labels)

        print("evaluate the model, extracting tree and scores", flush=True)

        # get the results
        train_acc = model.score(self.input_df.iloc[:, :-1], self.input_df.iloc[:, -1])
        n_leaves = model.leaves()
        print(f"Returning loss {(1 - train_acc) + self.lam * n_leaves}")
        return (1 - train_acc) + self.lam * n_leaves