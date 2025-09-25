import os 

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["NUMBA_NUM_THREADS"] = "20"

from idealpestimation.src.testlinate import IdeologicalEmbedding
import ipdb
import pathlib
import jsonlines
import numpy as np
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from idealpestimation.src.utils import pickle, \
                                        time, timedelta, \
                                            load_matrix, fix_plot_layout_and_save
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

def ridge_cv_estimate(X, y, dirout, alphas=None, seed_value=None, selected_coords_names=None, country=None, year=None):
    """
    Ridge regression with 10-fold CV to select lambda    
    -----------
    X : np.ndarray
        Feature matrix (n_samples x n_features)
    y : np.ndarray
        Target vector (n_samples,)
    alphas : array-like, optional
        Candidate regularization strengths (λ). Default: logspace(-4, 4, 50)

    Returns:
    --------
    beta_matrix : np.ndarray
        Estimated coefficients including intercept (shape: n_features+1,)
        Format: [intercept, coef_1, coef_2, ..., coef_p]
    best_alpha : float
        The selected regularization coefficient
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 50)
    
    # RidgeCV with LOOCV
    ridgecv = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, store_cv_results=True, cv=None)
    )

    # Fit model
    ridgecv.fit(X, y)
    # Extract best alpha
    best_alpha = ridgecv.named_steps['ridgecv'].alpha_
    # Extract coefficients
    intercept = ridgecv.named_steps['ridgecv'].intercept_
    coefs = ridgecv.named_steps['ridgecv'].coef_
    beta_matrix = coefs
    beta_matrix[:, -1] = intercept
    
    cv_values = ridgecv.named_steps['ridgecv'].cv_results_
    mean_loocv_mse = cv_values.mean(axis=0).sum(axis=0)
    mean_loocv_rmse = np.sqrt(mean_loocv_mse)
    plt.figure(figsize=(8,5))
    plt.semilogx(alphas, mean_loocv_rmse, marker="o")
    plt.axvline(best_alpha, color="r", linestyle="--", 
                label=f"Best α = {best_alpha:.4f}")
    plt.xlabel("Alpha (λ)")
    plt.ylabel("LOOCV RMSE")
    plt.title("Ridge Regression: LOOCV Error vs Regularization Strength")
    plt.legend()
    outpath = "{}/plots_upd_fixeddims_ridgecv/{}".format(dirout, country)
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    if country == "us":
        savename = "{}/{}_{}_{}_ridgecv.png".format(outpath, country, year, selected_coords_names[0])
    else:
        savename = "{}/{}_{}_{}_{}_ridgecv.png".format(outpath, country, year, selected_coords_names[0], selected_coords_names[1])
    print('About to save the figure...')
    plt.savefig(savename,
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    plt.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    print('✅ Figure saved to {}'.format(savename))
    # plt.show()
    
    return beta_matrix, best_alpha

def allocate_followers(N: List[int], n_h: int, cap: int, threshold: Optional[int] = None) -> List[int]:
    """
    Allocate how many followers to sample from *each followee*, subject to caps and optional census rule.

    Important:
        - All followees are included.
        - We only decide how many of their followers to keep (m_i <= N_i).
        - Later, you will actually draw m_i followers at random from each followee.

    Args:
        N        : list of follower counts per followee (N_i)
        n_h      : total number of followers to allocate across all followees
        cap      : maximum followers to take from any one followee (c)
        threshold: if not None, any followee with N_i <= threshold is taken fully (census)

    Returns:
        m        : list of integers (allocations m_i per followee)
    """
    N = np.array(N, dtype=float)
    caps = np.minimum(N, cap)
    
    # Census rule: pre-assign all followers for small accounts
    m = np.zeros_like(N)
    if threshold is not None:
        census_mask = N <= threshold
        m[census_mask] = N[census_mask]
        n_h -= m[census_mask].sum()
        N = N.copy()  # avoid modifying original
        caps = np.minimum(N, cap)
        caps[census_mask] = 0  # exclude from proportional allocation


    if n_h <= 0:
        return m.astype(int).tolist()

    # initial proportional allocation (can be fractional)
    raw = n_h * caps / caps.sum()
    m = np.maximum(m, np.minimum(raw, N)).astype(float)

    # iterative redistribution of leftover quota
    while True:
        leftover = n_h - m.sum()
        if leftover < 1e-6:
            break

        unsat = (m + 1e-9) < N  # followees not yet filled
        if not np.any(unsat):
            break

        remaining_capacity = N[unsat] - m[unsat]
        share = remaining_capacity / remaining_capacity.sum()
        delta = leftover * share

        m[unsat] = np.minimum(m[unsat] + delta, N[unsat])

    # final rounding to integers
    m_int = np.floor(m).astype(int)
    remainder = int(n_h - m_int.sum())
    if remainder > 0:
        # distribute remainder to those with largest fractional parts
        frac = m - m_int
        order = np.argsort(-frac)
        for i in order[:remainder]:
            m_int[i] += 1


    return m_int.tolist()

def diagnostics(N: List[int], m: List[int], cap: int, n_h: int, buckets: Optional[List[int]] = None, dirout: str = None):
    """
    Run diagnostics comparing allocations to original population.

    - Distribution of follower counts (original vs allocated)
    - Fraction of accounts fully taken (census)
    - Effective sample size (ESS) overall and by bucket if provided
    - Distribution of weights (inverse inclusion probabilities)
    
    Args:
        N       : list of follower counts per followee
        m       : list of allocated counts per followee
        cap     : per-followee maximum
        n_h     : target number of samples
        buckets : optional list of bucket labels (same length as N)
    """
    
    N = np.array(N)
    m = np.array(m)

    print("Total allocated:", m.sum(), "(target:", n_h, ")")
    print("Number of accounts with census (all followers taken):", np.sum(m == N))
    print("Number of accounts capped:", np.sum(m == cap))

    # Plot distribution of allocated vs original
    plt.figure(figsize=(8, 5))
    plt.hist(N, bins=30, alpha=0.5, label="Original follower counts")
    plt.xlabel("Followers per followee")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of original followers")
    # plt.show()
    savename = "{}/{}_{}_distribution_orig_followers.png".format(dirout, country, year)    
    plt.savefig(savename,
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    plt.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    
    plt.figure(figsize=(8, 5))  
    plt.hist(m, bins=30, alpha=0.5, label="Allocated followers")
    plt.xlabel("Followers per followee")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of allocated followers")
    # plt.show()
    savename = "{}/{}_{}_distribution_alloc_followers.png".format(dirout, country, year)    
    # plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(N, bins=30, alpha=0.5, label="Original follower counts")
    plt.hist(m, bins=30, alpha=0.5, label="Allocated followers")
    plt.xlabel("Followers per followee")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Distribution of original vs allocated followers")
    savename = "{}/{}_{}_distribution_orig_vs_allocated_followers.png".format(dirout, country, year)    
    plt.savefig(savename,
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    plt.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 

    # Coverage ratio
    coverage = m / np.maximum(N, 1)
    plt.figure(figsize=(8, 5))
    plt.scatter(N, coverage, alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Original follower count (log scale)")
    plt.ylabel("Coverage ratio (allocated / original)")
    plt.title("Coverage ratio by followee size")
    # plt.show()
    savename = "{}/{}_{}_coverage_ratio_by_followee_size.png".format(dirout, country, year)    
    plt.savefig(savename,
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    plt.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 

    # Weights (approximate inclusion probabilities)
    inclusion_probs = np.where(N > 0, m / N, 0)
    weights = np.where(inclusion_probs > 0, 1.0 / inclusion_probs, 0)

    print("Inclusion probability stats:")
    if np.any(inclusion_probs > 0):
        print(" min:", inclusion_probs[inclusion_probs > 0].min(),
              " max:", inclusion_probs.max(),
              " mean:", inclusion_probs.mean())

    print("Weight stats:")
    if np.any(weights > 0):
        print(" min:", weights[weights > 0].min(),
              " max:", weights.max(),
              " mean:", weights[weights > 0].mean())

    plt.figure(figsize=(8, 5))
    plt.hist(weights[weights > 0], bins=50)
    plt.xlabel("Sampling weight (1/π)")
    plt.ylabel("Frequency")
    plt.title("Distribution of follower weights")
    # plt.show()
    savename = "{}/{}_{}_distribution_follower_weights.png".format(dirout, country, year)    
    plt.savefig(savename,
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    plt.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 

    # Effective sample size (ESS)
    if np.any(weights > 0):
        ess_overall = (weights.sum())**2 / (weights**2).sum()
        print("Overall effective sample size (ESS):", ess_overall)

        if buckets is not None:
            buckets = np.array(buckets)
            for b in np.unique(buckets):
                mask = buckets == b
                w_b = weights[mask]
                if np.any(w_b > 0):
                    ess_b = (w_b.sum())**2 / (w_b**2).sum()
                    print(f"  ESS for bucket {b}: {ess_b}")

class AttitudinalEmbedding(BaseEstimator, TransformerMixin): 

    def __init__(self, N = None, random_state = None):

        self.random_state = random_state

        # number of latent ideological dimensions to be considered
        self.N = N # default : None --> P (number of groups) - 1

    def fit(self, X, Y, dirout, seed_value, selected_coords_names=None, country=None, year=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('\'X\' parameter must be a pandas dataframe')

        if 'entity' not in X.columns:
            raise ValueError('\'X\' has to have an \'entity\' column')

        if not isinstance(Y, pd.DataFrame):
            raise ValueError('\'Y\' parameter must be a pandas dataframe')

        if 'entity' not in Y.columns:
            raise ValueError('\'Y\' has to have an \'entity\' column')
        
        Y['entity'] = pd.to_numeric(Y['entity'])

        # also keep only the groups that exist in both datasets
        ga_merge_df = pd.merge(X, Y, on = 'entity', how = 'inner')
        X = X[X['entity'].isin(ga_merge_df.entity.unique())]
        Y = Y[Y['entity'].isin(ga_merge_df.entity.unique())]

        print('Groups: ', Y['entity'].values)
        print('Y columns: ', len(Y.columns), Y.columns)

        # finally fit an affine transformation to map X --> Y

        # first sort X and Y by entity (so as to have the corresponding mapping in the same rows)
        X = X.sort_values('entity', ascending = True)
        Y = Y.sort_values('entity', ascending = True)

        # convert Y to Y_tilda
        Y_df = Y.drop('entity', axis = 1, inplace = False)
        self.Y_columns = Y_df.columns.tolist()
        Y_np = Y_df.to_numpy().T
        ones_np = np.ones((Y_np.shape[1],), dtype = float)
        Y_tilda_np = np.append(Y_np, [ones_np], axis = 0)
        #print(Y_np.shape, Y_tilda_np.shape)

        # convert X to X_tilda
        X_df = X.drop('entity', axis = 1, inplace = False)
        X_np = X_df.to_numpy()
        print('Number of political parties: ', X_np.shape[0])
        if self.N is None:
            self.employed_N_ = X_np.shape[0] - 1
        else:
            self.employed_N_ = self.N
        X_np = X_np[:, :self.employed_N_]
        self.X_columns = X_df.columns.tolist() 
        if self.employed_N_ < len(self.X_columns):
            self.X_columns = self.X_columns[:self.employed_N_]
        X_np = X_np.T
        ones_np = np.ones((X_np.shape[1],), dtype = float)
        X_tilda_np = np.append(X_np, [ones_np], axis = 0)
        #print(X_tilda_np.shape)

        # finally compute T_tilda_aff
        # T_tilda_aff_np_1 = np.matmul(Y_tilda_np, X_tilda_np.T)
        # T_tilda_aff_np_2 = np.matmul(X_tilda_np, X_tilda_np.T)
        # T_tilda_aff_np_3 = np.linalg.inv(T_tilda_aff_np_2)
        # self.T_tilda_aff_np_ = np.matmul(T_tilda_aff_np_1, T_tilda_aff_np_3)
        # #print(self.T_tilda_aff_np.shape)
        # self.T_tilda_aff_np_[-1, :-1] = 0.0

        beta_matrix, best_alpha = ridge_cv_estimate(X_tilda_np.T, Y_tilda_np.T, dirout=dirout, alphas=None, seed_value=seed_value,
                                                    selected_coords_names=selected_coords_names, country=country, year=year)
        self.T_tilda_aff_np_ = beta_matrix
        
        return self

    def transform(self, X):

        entitiy_col = None
        if isinstance(X, pd.DataFrame): # check input and convert to matrix
            if 'entity' not in X.columns:
                raise ValueError('Input dataframe has to have an \'entity\' column.')

            entitiy_col = X['entity'].values
            X.drop('entity', axis = 1, inplace = True)

            for c in X.columns:
                X[c] = X[c].astype(float)

            X = X.to_numpy()

        try:
            

            X_np = X[:, :self.employed_N_]
            X_np = X_np.T
            ones_np = np.ones((X_np.shape[1],), dtype = float)
            X_tilda_np = np.append(X_np, [ones_np], axis = 0)

            if self.T_tilda_aff_np_.shape[1] != X_tilda_np.shape[0]:
                raise ValueError('Wrong input dimensions')

            Y_tilda_np = np.matmul(self.T_tilda_aff_np_, X_tilda_np)
            #print(Y_tilda_np.shape)
            Y_tilda_np = Y_tilda_np[:-1]
            Y_tilda_np = Y_tilda_np.T

            Y = pd.DataFrame(Y_tilda_np, columns = self.Y_columns)
            if entitiy_col is not None:
                cols = Y.columns
                cols = cols.insert(0, 'entity')
                Y['entity'] = entitiy_col
                Y = Y[cols]

        except AttributeError:
            raise AttributeError('Transformation parameters have not been computed.')

        return (Y)

    def get_params(self, deep = True):
        return {'random_state': self.random_state, 
                'N': self.N}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

    def score(self, X, y):
        return 1

    def load_attitudinal_referential_coordinates_from_file(self, path_attitudinal_reference_data,
            attitudinal_reference_data_header_names = None):

        # check if attitudinal reference data file exists
        if not os.path.isfile(path_attitudinal_reference_data):
            raise ValueError('Attitudinal reference data file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_attitudinal_reference_data, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Attitudinal reference data file has to have at least two columns.')

        if attitudinal_reference_data_header_names is not None:
            if attitudinal_reference_data_header_names['entity'] not in header_df.columns:
                raise ValueError('Attitudinal reference data file has to have a '
                        + attitudinal_reference_data_header_names['entity'] + ' column.')

        # load attitudinal reference data
        attitudinal_reference_data_df = None
        if attitudinal_reference_data_header_names is None:
            attitudinal_reference_data_df = pd.read_csv(path_attitudinal_reference_data,
                    header = None).rename(columns = {0:'entity'})
        else:
            attitudinal_reference_data_df = pd.read_csv(path_attitudinal_reference_data).rename(columns
                    = {attitudinal_reference_data_header_names['entity']:'entity'})
            if 'dimensions' in attitudinal_reference_data_header_names.keys():
                cols = attitudinal_reference_data_header_names['dimensions'].copy()
                cols.append('entity')
                attitudinal_reference_data_df = attitudinal_reference_data_df[cols]

        # exclude groups with a NaN in any of the dimensions (or group)
        attitudinal_reference_data_df.dropna(inplace = True)
        attitudinal_reference_data_df['entity'] = attitudinal_reference_data_df['entity'].astype(str)

        return (attitudinal_reference_data_df)

    def load_ideological_embedding_from_file(self, path_ideological_embedding, ideological_embedding_header_names = None):

        # check if ideological embedding file exists
        if not os.path.isfile(path_ideological_embedding):
            raise ValueError('Ideological embedding data file does not exist.')

        # handles files with or without header
        header_df = pd.read_csv(path_ideological_embedding, nrows = 0)
        column_no = len(header_df.columns)
        if column_no < 2:
            raise ValueError('Ideological embedding data file has to have at least two columns.')
        #
        if ideological_embedding_header_names is not None:
            if ideological_embedding_header_names['entity'] not in header_df.columns:
                raise ValueError('Ideological embedding data file has to have a '
                        + ideological_embedding_header_names['entity'] + ' column.')

        # load ideological embeddings
        ideological_embedding_df = None
        if ideological_embedding_header_names is None:
            ideological_embedding_df = pd.read_csv(path_ideological_embedding,
                    header = None).rename(columns = {0:'entity'})
        else:
            ideological_embedding_df = pd.read_csv(path_ideological_embedding).rename(columns =
                    {ideological_embedding_header_names['entity']:'entity'})

        # exclude nodes with a NaN in any of the dimensions (or group)
        ideological_embedding_df.dropna(inplace = True)
        ideological_embedding_df['entity'] = ideological_embedding_df['entity'].astype(str)

        if ideological_embedding_header_names is not None:
            if 'dimensions' in ideological_embedding_header_names.keys():
                ideological_embedding_df = ideological_embedding_df[ideological_embedding_header_names['dimensions']]

        return(ideological_embedding_df)

    def load_entity_to_group_mapping_from_file(self, path_entity_to_group_mapping,
            entity_to_group_mapping_header_names = None):

            # check if entity to group file exists
            if not os.path.isfile(path_entity_to_group_mapping):
                raise ValueError('Entity to group data file does not exist.')

            # handles entity to group files with or without header
            header_df = pd.read_csv(path_entity_to_group_mapping, nrows = 0)
            column_no = len(header_df.columns)
            if column_no < 2:
                raise ValueError('Entity to group data file has to have at least two columns.')

            if entity_to_group_mapping_header_names is not None:
                if entity_to_group_mapping_header_names['group'] not in header_df.columns:
                    raise ValueError('Entity to group data file has to have a '
                            + entity_to_group_mapping_header_names['group'] + ' column.')

                    if entity_to_group_mapping_header_names['entity'] not in header_df.columns:
                        raise ValueError('Entity to group data file has to have a '
                                + entity_to_group_mapping_header_names['entity'] + ' column.')

            # load entity to group data
            entity_to_group_data_df = None
            if entity_to_group_mapping_header_names is None:
                entity_to_group_data_df = pd.read_csv(path_entity_to_group_mapping, header
                        = None).rename(columns = {0:'entity', 1:'group'})
            else:
                entity_to_group_data_df = pd.read_csv(path_entity_to_group_mapping).rename(columns =
                        {entity_to_group_mapping_header_names['group']:'group',
                            entity_to_group_mapping_header_names['entity']:'entity'})

            # maintain only entity and group columns
            entity_to_group_data_df = entity_to_group_data_df[['entity', 'group']]
            entity_to_group_data_df.dropna(inplace = True)
            entity_to_group_data_df['entity'] = entity_to_group_data_df['entity'].astype(str)
            entity_to_group_data_df['group'] = entity_to_group_data_df['group'].astype(str)

            # exclude rows with a NaN in any of the columns
            entity_to_group_data_df.dropna(inplace = True)

            # check that each entity belongs to only 1 group
            has_entities_in_more_than_one_group = entity_to_group_data_df.groupby(['entity']).size().max() > 1
            if has_entities_in_more_than_one_group:
                raise ValueError('Entities should belong to a single group.')

            return(entity_to_group_data_df)

    def convert_to_group_ideological_embedding(self, ideological_embedding_df, entity_to_group_data_df,
            entity_to_group_agg_fun = None):

        if not isinstance(ideological_embedding_df, pd.DataFrame):
            raise ValueError('\'ideological_embedding_df\' parameter must be a pandas dataframe')

        if not isinstance(entity_to_group_data_df, pd.DataFrame):
            raise ValueError('\'entity_to_group_data_df\' parameter must be a pandas dataframe')

        if 'entity' not in ideological_embedding_df.columns:
            raise ValueError('\'ideological_embedding_df\' has to have an \'entity\' column')

        if 'entity' not in entity_to_group_data_df.columns:
            raise ValueError('\'entity_to_group_data_df\' has to have an \'entity\' column')

        if 'group' not in entity_to_group_data_df.columns:
            raise ValueError('\'entity_to_group_data_df\' has to have an \'group\' column')

        # add group information to the ideological embeddings
        entity_group_ideological_embedding_df = pd.merge(ideological_embedding_df, entity_to_group_data_df, on = 'entity')
        entity_group_ideological_embedding_df.drop('entity', axis = 1, inplace = True)

        entity_group_ideological_embedding_df['k'] = pd.to_numeric(entity_group_ideological_embedding_df['k'])
        entity_group_ideological_embedding_df['group'] = pd.to_numeric(entity_group_ideological_embedding_df['group'])
        # create ideological embeddings aggregates : user can define custom (columnwise) aggregate
        entity_ideological_embedding_df = None
        if entity_to_group_agg_fun is None:
            entity_ideological_embedding_df = entity_group_ideological_embedding_df.groupby(['group']).agg('mean').reset_index()
        else:
            entity_ideological_embedding_df = \
                    entity_group_ideological_embedding_df.groupby(['group']) .agg(entity_to_group_agg_fun).reset_index()

        entity_ideological_embedding_df.rename(columns = {'group': 'entity'}, inplace = True)

        return(entity_ideological_embedding_df)

    def save_transformation_parameters(self, path_to_transformation_parameters_file):
        try:
            at_df_index = self.Y_columns.copy()  # metadata
            at_df_index.append('plus_one_column')
            at_df_columns = self.X_columns.copy()
            at_df_columns.append('plus_one_row')

            at_df = pd.DataFrame(self.T_tilda_aff_np_, columns = at_df_columns)
            at_df.index = at_df_index
            #at_df.index.name = ''
            at_df.to_csv(path_to_transformation_parameters_file)
        except AttributeError:
            raise AttributeError('Transformation parameters have not been computed.')

def adjacency_to_edge_list(Y):
    
    # Y: K x J

    adj_matrix = np.array(Y)

    # Find indices where an edge exists
    k_indices, j_indices = np.where(adj_matrix != 0)
    
    df = pd.DataFrame({
        'i': j_indices,
        'j': k_indices
    })
    
    return df

def plot_hexhist(target_coords, source_coords, df_ref_group, group_attitudes, selected_coords_names, selected_coords_names_att, partylabels, country, dirout, vis="users", seed_value=None):
    
    color_dic = {'0':'blue','1':'red','2':'gold','3':'orange','4':'green',
                 '5':'violet','6':'cyan','7':'magenta','8':'brown','9':'gray',
                 '10':'olive','11':'yellow','12':'lime','13':'navy','14':'coral', 
                 '15': "turquoise", "16": "indigo", "17": "teal", "18": "purple", 
                 "19" : "pink", "20": "khaki"}
    
    target_coords['k'] = target_coords.index.map(df_ref_group.set_index('i')['k'])
    if vis == "users":
        g = sn.jointplot(data=source_coords.drop_duplicates(), x=selected_coords_names[0], y=selected_coords_names[1], kind="hex", gridsize=100)
    else:
        g = sn.jointplot(data=target_coords.drop_duplicates(), x=selected_coords_names[0], y=selected_coords_names[1], kind="hex", gridsize=100)

    ax = g.ax_joint
    for k in target_coords['k'].unique():
        df_k = target_coords[target_coords['k']==k]        
        ax.scatter(df_k[selected_coords_names[0]],df_k[selected_coords_names[1]],
            marker='+',s=30,alpha=0.5,color=color_dic[k])

    if country == 'us':
        ax.axvline(x=1, color='red', linestyle='--', label='GOP')
        ax.axvline(x=-1, color='blue', linestyle='--', label='Dem')
        ax.set_xlabel(selected_coords_names_att[0])
        ax.set_ylabel("")
        custom_lines = [
            Line2D([0], [0], color='red', linestyle='--'),
            Line2D([0], [0], color='blue', linestyle='--')
        ]

        plt.legend(
            handles=[plt.Line2D([], [], color='black'), *custom_lines],
            labels=["Dem", "GOP"],
        )
        fig = g.figure
        outpath = "{}/plots_upd_fixeddims_ridgecv/{}".format(dirout, country)
        pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
        savename = "{}/{}_{}_{}_{}_att_{}.png".format(outpath, country, year, selected_coords_names[0], selected_coords_names[1], vis)
        print('About to save the figure...')
        fig.savefig(savename,
                dpi=300,
                bbox_inches='tight',
                transparent=False) 
        print('✅ Figure saved to {}'.format(savename))
        fig.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
        return

    fig = g.figure
    # plt.show()
    outpath = "{}/plots_upd_fixeddims_ridgecv/{}".format(dirout, country)
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    savename = "{}/{}_{}_{}_{}_{}.png".format(outpath, country, year, selected_coords_names[0], selected_coords_names[1], vis)
    print('About to save the first figure...')
    fig.savefig(savename,
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    fig.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
    print('✅ Figure saved to {}'.format(savename))
    
    
    group_attitudes['k'] = group_attitudes['k'].astype(str)  
    group_ideologies = target_coords.groupby('k').mean()
    fig = plt.figure(figsize=(10,4))# width, height inches
    ax = {1:fig.add_subplot(1,2,1), 2:fig.add_subplot(1,2,2)}
    for k,row in group_ideologies[group_ideologies.index.isin(group_attitudes['k'])].iterrows():
        ax[1].plot(row[selected_coords_names[0]],row[selected_coords_names[1]],'o',mec='k',color=color_dic[k])
    ax[1].set_xlabel(selected_coords_names[0])
    ax[1].set_ylabel(selected_coords_names[1])
    ax[1].set_title('Group positions in ideological space')
    
    for k,row in group_attitudes.iterrows():
        ax[2].plot(row[selected_coords_names_att[0]], row[selected_coords_names_att[1]],'o',mec='k',color=color_dic[row['k']])
    if selected_coords_names_att[0] == 'lrecon': 
        ax[2].set_xlabel("Left-Right Economic")
    elif selected_coords_names_att[0] == 'galtan':
        ax[2].set_xlabel("Liberal-Conservative")
    elif selected_coords_names_att[0] == 'antielite_salience':
        ax[2].set_xlabel("Anti-Elite Salience")
    if selected_coords_names_att[1] == 'lrecon': 
        ax[2].set_ylabel("Left-Right Economic")
    elif selected_coords_names_att[1] == 'galtan':
        ax[2].set_ylabel("Liberal-Conservative")
    elif selected_coords_names_att[1] == 'antielite_salience':
        ax[2].set_ylabel("Anti-Elite Salience")
    # ax[2].set_xlabel(selected_coords_names_att[0])
    # ax[2].set_ylabel(selected_coords_names_att[1])
    ax[2].set_title('Group positions in attitudinal space')
    attiembedding_model = AttitudinalEmbedding(N = 2)

    target_coords['entity'] = target_coords.index 
    X = attiembedding_model.convert_to_group_ideological_embedding(target_coords, df_ref_group.rename(columns={'i':'entity','k':'group'}))
    Y = group_attitudes.rename(columns={'k':'entity'})
    attiembedding_model.fit(X, Y, dirout=dirout, seed_value=seed_value, selected_coords_names=selected_coords_names_att, country=country, year=year)
    target_coords['entity'] = target_coords.index
    target_attitudinal = attiembedding_model.transform(target_coords)
    source_coords['entity'] = source_coords.index
    source_attitudinal = attiembedding_model.transform(source_coords)
    target_attitudinal['k'] = target_attitudinal['entity'].map(pd.Series(index=df_ref_group['i'].values,data=df_ref_group['k'].values))

    if vis == "users":
        g = sn.jointplot(data=source_attitudinal.drop_duplicates(),x=selected_coords_names_att[0],y=selected_coords_names_att[1], kind="hex", gridsize=100)
    else:
        g = sn.jointplot(data=target_attitudinal.drop_duplicates(),x=selected_coords_names_att[0],y=selected_coords_names_att[1], kind="hex", gridsize=100)

    print('jointplot created – axes:', g.ax_joint)
    ax = g.ax_joint
    print('axes object:', ax)
    for k in target_attitudinal['k'].unique():
        df_k = target_attitudinal[target_attitudinal['k']==k]       
        df_k_mean = df_k[[selected_coords_names_att[0],selected_coords_names_att[1]]].mean()
        ax.scatter(df_k[selected_coords_names_att[0]],df_k[selected_coords_names_att[1]],marker='+',s=30,alpha=0.5,color=color_dic[k])
        ax.plot(df_k_mean[selected_coords_names_att[0]],df_k_mean[selected_coords_names_att[1]],'o',mec='k',color=color_dic[k],ms=7)
        if int(k) % 2 ==0:
            ax.annotate(
                partylabels[int(k)].replace(";", "/"),
                xy=(df_k_mean[selected_coords_names_att[0]],df_k_mean[selected_coords_names_att[1]]),         # Point to annotate
                xytext=(df_k_mean[selected_coords_names_att[0]]-0.1,df_k_mean[selected_coords_names_att[1]]+0.2),   # Text location
                # arrowprops=dict(arrowstyle="->", color="red")
            )
        else:
            ax.annotate(
                partylabels[int(k)],
                xy=(df_k_mean[selected_coords_names_att[0]],df_k_mean[selected_coords_names_att[1]]),         # Point to annotate
                xytext=(df_k_mean[selected_coords_names_att[0]]+0.1,df_k_mean[selected_coords_names_att[1]]-0.2),   # Text location
                # arrowprops=dict(arrowstyle="->", color="red")
            )
    if selected_coords_names_att[0] == 'lrecon': 
        ax.set_xlabel("Left-Right Economic")
    elif selected_coords_names_att[0] == 'galtan':
        ax.set_xlabel("Liberal-Conservative")
    elif selected_coords_names_att[0] == 'antielite_salience':
        ax.set_xlabel("Anti-Elite Salience")
    if selected_coords_names_att[1] == 'lrecon': 
        ax.set_ylabel("Left-Right Economic")
    elif selected_coords_names_att[1] == 'galtan':
        ax.set_ylabel("Liberal-Conservative")
    elif selected_coords_names_att[1] == 'antielite_salience':
        ax.set_ylabel("Anti-Elite Salience")
    fig = g.figure
    outpath = "{}/plots_upd_fixeddims_ridgecv/{}".format(dirout, country)
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    savename = "{}/{}_{}_{}_{}_att_{}.png".format(outpath, country, year, selected_coords_names_att[0], selected_coords_names_att[1], vis)
    print('About to save the second figure...')
    try:
        fig.savefig(savename,
                dpi=100,
                bbox_inches='tight',
                transparent=False) 
        fig.savefig(savename.replace("png","pdf"),
            dpi=300,
            bbox_inches='tight',
            transparent=False) 
        print('✅ Figure saved to {}'.format(savename))
    except:
        return

def plot_single_coord(axe, axe_att, coord_idx, target_coords, source_coords, df_ref_group, group_attitudes, 
                    selected_coords_names, selected_coords_names_att, partylabels, country, dirout, year, coordtargetplot, vis="users", seed_value=None):
    
    color_dic = {'0':'blue','1':'red','2':'gold','3':'orange','4':'green',
                 '5':'violet','6':'cyan','7':'magenta','8':'brown','9':'gray',
                 '10':'olive','11':'yellow','12':'lime','13':'navy','14':'coral', 
                 '15': "turquoise", "16": "indigo", "17": "teal", "18": "purple", 
                 "19" : "pink", "20": "khaki"}

    target_coords['k'] = target_coords.index.map(df_ref_group.set_index('i')['k'])
    if vis == "users":
        df = source_coords.drop_duplicates()
    else:
        df = target_coords.drop_duplicates()
    series = df[selected_coords_names[coord_idx]]
    if axe is not None:
        axe.hist(series.dropna(), color=color_dic[str(coord_idx+2+coordtargetplot)], bins=20, edgecolor='black', alpha=0.7, density=True)
        if coord_idx == 0:
            axe.set_title("{} - {}".format(country, year))
        axe.set_xlabel(series.name)
        kde = gaussian_kde(series.dropna().values)
        x_vals = np.linspace(series.dropna().values.min(), series.dropna().values.max(), 200)
        axe.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="")
       
    group_attitudes['k'] = group_attitudes['k'].astype(str)  
    group_ideologies = target_coords.groupby('k').mean()
    attiembedding_model = AttitudinalEmbedding(N = 2)

    target_coords['entity'] = target_coords.index 
    X = attiembedding_model.convert_to_group_ideological_embedding(target_coords, df_ref_group.rename(columns={'i':'entity','k':'group'}))
    Y = group_attitudes.rename(columns={'k':'entity'})
    attiembedding_model.fit(X, Y, dirout=dirout, seed_value=seed_value, selected_coords_names=selected_coords_names_att, country=country, year=year)
    target_coords['entity'] = target_coords.index
    target_attitudinal = attiembedding_model.transform(target_coords)
    source_coords['entity'] = source_coords.index
    source_attitudinal = attiembedding_model.transform(source_coords)
    target_attitudinal['k'] = target_attitudinal['entity'].map(pd.Series(index=df_ref_group['i'].values,data=df_ref_group['k'].values))

    if vis == "users":
        df = source_attitudinal.drop_duplicates()
    else:
        df = target_attitudinal.drop_duplicates()
    series = df[selected_coords_names_att[coord_idx]]
    axe_att.hist(series.dropna(), color=color_dic[str(coord_idx+2+coordtargetplot)], bins=20, edgecolor='black', alpha=0.7, density=True)
    if coord_idx == 0:
        axe_att.set_title("{} - {}".format(country, year))
    if series.name == "lrecon":
        axe_att.set_xlabel("Left-Right Economic")
    elif series.name == "galtan":
        axe_att.set_xlabel("Liberal-Conservative")
    elif series.name == "antielite_salience":
        axe_att.set_xlabel("Anti-Elite Salience")    
    elif series.name == "Party":
        axe_att.set_xlabel("Democrats - Republicans")
    kde = gaussian_kde(series.dropna().values)
    x_vals = np.linspace(series.dropna().values.min(), series.dropna().values.max(), 200)
    axe_att.plot(x_vals, kde(x_vals), color="red", linewidth=2, label="")
        
    plt.tight_layout() 
    
    return




if __name__ == "__main__":
    
    # standardise resulting CA dimensions
    
    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1
    
    countries = ["poland", "netherlands", "finland", "uk", "france", "germany", "us"]
    dataspace = "/mnt/hdd2/ioannischalkiadakis/epodata_rsspaper/"

    for year in [2020, 2023]:
        # CHES2019:  0: 'lrecon', 2: 'antielite_salience', 28: 'civlib_laworder', 30: 'country', 36: 'lrgen', 47: 'people_vs_elite', 15 : "galtan" (liberal-conservative)
        # CHES2023:  5: 'lrecon', 0: 'antielite_salience', 12: "galtan" (liberal-conservative)
        if year == 2020:
            #selected_coords = [36, 47] # choose two CHES dimensions related to COVID-19 polarised debates, set dataframe names to the names of the coords
            selected_coords_top = [0, 15, 2]
        elif year == 2023:
            selected_coords_top = [5, 12, 0]        
        selected_coords_names = ['latent_dimension_1', 'latent_dimension_2']
        selected_coords_names_att_top = ["lrecon", "galtan", "antielite_salience"]
        
        for country in countries: 
            
            if country == "us":
                fig_users, axes_users = plt.subplots(1, 1, figsize=(8, 10))
                fig_leadusers, axes_leadusers = plt.subplots(1, 1, figsize=(8, 10))
                fig_users_att, axes_users_att = plt.subplots(1, 1, figsize=(8, 10))
                fig_leadusers_att, axes_leadusers_att = plt.subplots(1, 1, figsize=(8, 10))
            else:
                fig_users, axes_users = plt.subplots(2, 1, figsize=(8, 10))
                fig_leadusers, axes_leadusers = plt.subplots(2, 1, figsize=(8, 10))
                fig_users_att, axes_users_att = plt.subplots(3, 1, figsize=(8, 10))
                fig_leadusers_att, axes_leadusers_att = plt.subplots(3, 1, figsize=(8, 10))
            coordsplotted = []
            
            for selected_coords, selected_coords_names_att in zip([selected_coords_top[:2], selected_coords_top[1:], selected_coords_top[::2]],
                    [selected_coords_names_att_top[:2], selected_coords_names_att_top[1:], selected_coords_names_att_top[::2]]):
                
                print('Selected coords:', selected_coords, selected_coords_names, selected_coords_names_att)    
                if country == "us" and year == 2020:
                    continue
                datasets_names = [file.name for file in pathlib.Path(dataspace).iterdir() if file.is_file() and (country in file.name and str(year) in file.name and "mappings" in file.name)]
                if len(datasets_names) == 0:
                    continue                

                K = int(datasets_names[0].split("_")[3].replace("K", ""))
                J = int(datasets_names[0].split("_")[4].replace("J", ""))            
                print(parallel, K, J, elementwise, evaluate_posterior)            
                parameter_names = ["X", "Z"]
                d = 2
                mappings, node_to_index_start, index_to_node_start, \
                        node_to_index_end, index_to_node_end, Y = load_matrix("{}/Y_{}_{}".format(dataspace, country, year), K, J)      
                
                try:
                    bipartite = pd.read_csv("{}/{}/bipartite_{}_{}.csv".format(dataspace, country, country, year))
                    X_hat_df = pd.read_csv("{}/{}/X_{}_{}.csv".format(dataspace, country, country, year), index_col=0)
                    Z_hat_df = pd.read_csv("{}/{}/Z_{}_{}.csv".format(dataspace, country, country, year), index_col=0)
                    Z_hat = Z_hat_df.to_numpy()
                    X_hat = X_hat_df.to_numpy()
                except:                
                    ideoembedding_model = IdeologicalEmbedding(n_latent_dimensions = 2, 
                                                            in_degree_threshold = 10, 
                                                            out_degree_threshold = 10)
                    if country == "us":                   
                        # total followers per politician
                        rsum = np.sum(Y, axis=0)
                        rsum = np.asarray(rsum).flatten()
                        # total politicians followed per user                    
                        csum = np.sum(Y, axis=1)
                        csum = np.asarray(csum).flatten()
                        goal_n = 1.5e7                   
                        c = 30000                                    
                        threshold_census = 1000
                        y_idx_subsample = []                    
                        m_j_all = allocate_followers(N=rsum, n_h=goal_n, cap=c, threshold=threshold_census)                         
                        for i in range(len(m_j_all)):
                            y_idx_subsample.extend(np.random.choice(np.argwhere(Y[:, i]==1)[:, 0], size=m_j_all[i], replace=False).tolist())                        
                        diagnostics(N=rsum, m=m_j_all, cap=c, n_h=goal_n, dirout=dataspace)
                        Y = Y[y_idx_subsample, :]
                        Y = Y.todense().astype(np.int8)
                        K = Y.shape[0]
                        # note that if Users IDs are needed, they must be stored here and retrieved when needed - subsampling changes order
                        parameter_space_dim = (K+J)*d + J + K + 2
                        print(Y.shape)
                    else:
                        Y = Y.todense().astype(np.int8)
                    
                    bipartite = adjacency_to_edge_list(Y)            
                    print('columns :'+str(bipartite.columns))
                    print('edges: '+str(bipartite.shape[0]))
                    print('num. of reference nodes i: '+ str(bipartite['i'].nunique()))
                    print('num. of follower nodes j: '+ str(bipartite['j'].nunique()))
                    
                    bipartite.rename(columns={'i':'target','j':'source'},inplace=True)
                    bipartite.to_csv("{}/{}/bipartite_{}_{}.csv".format(dataspace, country, country, year), index=False)

                    ideoembedding_model.fit(bipartite)
                    target_coords = ideoembedding_model.ideological_embedding_target_latent_dimensions_
                    print(len(target_coords))
                    target_coords.columns = selected_coords_names
                    source_coords = ideoembedding_model.ideological_embedding_source_latent_dimensions_
                    print(len(source_coords))
                    source_coords.columns = selected_coords_names
                    Z_hat_df = target_coords
                    Z_hat_df.to_csv("{}/{}/Z_{}_{}.csv".format(dataspace, country, country, year))
                    Z_hat = target_coords.to_numpy()
                    X_hat_df = source_coords
                    X_hat_df.to_csv("{}/{}/X_{}_{}.csv".format(dataspace, country, country, year))

                mp_mapping = pd.read_csv("{}/parties_lead_users_{}_{}.csv".format(dataspace, country, year))
                if year == 2020:
                    all_parties = np.unique(mp_mapping.CHES2019_party_acronym.dropna().values).tolist()
                elif year == 2023:
                    if country == "us":
                        all_parties = np.unique(mp_mapping.GPS2019_party_acronym.dropna().values).tolist()
                    else:
                        all_parties = np.unique(mp_mapping.CHES2023_party_acronym.dropna().values).tolist()

                # linate map in same order as all_parties, keep only dimensions of interest
                parties_politicians = dict()
                linate_map_y = []
                for party in all_parties:
                    if year == 2020:
                        parties_politicians[party] = mp_mapping.loc[mp_mapping.CHES2019_party_acronym==party, "mp_pseudo_id"].values.tolist()
                        map_y_tmp = pd.read_csv("{}/y_party_ches2019_{}_{}.csv".format(dataspace, country, year))
                        map_y_tmp = map_y_tmp.loc[map_y_tmp.CHES2019_party_acronym==party, :].drop(columns=['CHES2019_party_acronym', 
                                                                                                            "EPO_party_acronym",
                                                                                                            "eu_econ_require",
                                                                                                            "eu_political_require",
                                                                                                            "eu_googov_require",
                                                                                                            "lrecon_dissent"])
                        feature_names = map_y_tmp.columns.values.tolist()
                        map_y_tmp = map_y_tmp.values.flatten()
                        selected_coords = [feature_names.index(selected_coords_names_att[0]), feature_names.index(selected_coords_names_att[1])]
                        print(np.asarray(feature_names)[selected_coords])
                        linate_map_y.append(map_y_tmp[selected_coords])
                    elif year == 2023:
                        if country == "us":
                            parties_politicians[party] = mp_mapping.loc[mp_mapping.GPS2019_party_acronym==party, "mp_pseudo_id"].values.tolist()   
                            if party == "Dem":                            
                                map_y_tmp = np.array([-1])
                            else:                            
                                map_y_tmp = np.array([1])
                            linate_map_y.append(map_y_tmp)
                        else:
                            parties_politicians[party] = mp_mapping.loc[mp_mapping.CHES2023_party_acronym==party, "mp_pseudo_id"].values.tolist()   
                            map_y_tmp = pd.read_csv("{}/y_party_ches2023_{}_{}.csv".format(dataspace, country, year))                    
                            map_y_tmp = map_y_tmp.loc[map_y_tmp.CHES2023_party_acronym==party, :].drop(columns=['CHES2023_party_acronym', "country", "electionyear",
                                                                                                                "EPO_party_acronym", "family", "in_gov"])
                            feature_names = map_y_tmp.columns.values.tolist()
                            selected_coords = [feature_names.index(selected_coords_names_att[0]), feature_names.index(selected_coords_names_att[1])]
                            print(np.asarray(feature_names)[selected_coords])
                            map_y_tmp = map_y_tmp.values.flatten()
                            linate_map_y.append(map_y_tmp[selected_coords])
                linate_map_y = np.stack(linate_map_y)
            
                # party ideal points in estimated space, average over MPs of each party
                # parties in order of appearance in all_parties
                party_ideal_points_est = np.zeros((len(all_parties), d))
                party_idx = []
                leaduser_i = []
                for party in all_parties:
                    z_loc = []
                    for mp in parties_politicians[party]:
                        # second condition accounts for dropped nodes due to degree thresholding
                        if mp in node_to_index_end.keys():
                            try:
                                user_idx = node_to_index_end[mp]                            
                                z_loc.append(Z_hat[user_idx, :])
                                leaduser_i.append(user_idx)                
                                party_idx.append(all_parties.index(party))
                            except:
                                print("{} dropped due to degree thresholding".format(user_idx))
                        else:
                            print("Lead user {} not found in mapping.".format(mp))
                            continue                    
                    party_ideal_points_est[all_parties.index(party), :] = np.mean(np.stack(z_loc), axis=0)     
                
                df_ref_group = pd.DataFrame({"i": leaduser_i, "k": party_idx}) # i: idx of lead user, k: idx of party
                df_ref_group = df_ref_group.astype({"i": str, "k": str})
                
                if country == "us" and year == 2023:
                    all_parties = ["Dem", "Rep"]
                    linate_map_y = np.array([[-1], [1]])
                    selected_coords_names_att = ['Party']
                    group_attitudes = pd.DataFrame(np.column_stack([np.arange(0, len(all_parties), 1), linate_map_y]), columns=["k", selected_coords_names_att[0]])
                else:
                    group_attitudes = pd.DataFrame(np.column_stack([np.arange(0, len(all_parties), 1), linate_map_y]), columns=["k", selected_coords_names_att[0], selected_coords_names_att[1]])
                group_attitudes = group_attitudes.astype({"k": int}).astype({"k": str})

                Z_hat_df.index = Z_hat_df.index.map(str)
                Z_hat_df = Z_hat_df.loc[Z_hat_df.index.isin(df_ref_group['i'])].copy()              

                plot_hexhist(Z_hat_df, X_hat_df, df_ref_group, group_attitudes, selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, vis="users", seed_value=seed_value)
                plot_hexhist(Z_hat_df, X_hat_df, df_ref_group, group_attitudes, selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, vis="leadusers", seed_value=seed_value)

                if len(coordsplotted) == 2:
                    coordtargetplot = 1
                else:
                    coordtargetplot = 0
                if len(coordsplotted) == 1 and country == "us":
                    continue
                if selected_coords_names_att[0] not in coordsplotted:
                    if country == "us":
                        plot_single_coord(axes_users, axes_users_att, 0, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                      selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="users", seed_value=seed_value)                                    
                        plot_single_coord(axes_leadusers, axes_leadusers_att, 0, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                      selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="leadusers", seed_value=seed_value)
                    else:
                        plot_single_coord(axes_users[0+coordtargetplot], axes_users_att[0+coordtargetplot], 0, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                      selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="users", seed_value=seed_value)                                    
                        plot_single_coord(axes_leadusers[0+coordtargetplot], axes_leadusers_att[0+coordtargetplot], 0, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                      selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="leadusers", seed_value=seed_value)
                    coordsplotted.append(selected_coords_names_att[0])
                if country != "us":
                    if selected_coords_names_att[1] not in coordsplotted and len(coordsplotted) < 2:                           
                        plot_single_coord(axes_users[1+coordtargetplot], axes_users_att[1+coordtargetplot], 1, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                    selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="users", seed_value=seed_value)                    
                        plot_single_coord(axes_leadusers[1+coordtargetplot], axes_leadusers_att[1+coordtargetplot], 1, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                    selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="leadusers", seed_value=seed_value)
                        coordsplotted.append(selected_coords_names_att[1])
                    elif selected_coords_names_att[1] not in coordsplotted and len(coordsplotted) == 2:
                        plot_single_coord(None, axes_users_att[1+coordtargetplot], 1, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                    selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="users", seed_value=seed_value)                    
                        plot_single_coord(None, axes_leadusers_att[1+coordtargetplot], 1, Z_hat_df, X_hat_df, df_ref_group, group_attitudes, 
                                    selected_coords_names, selected_coords_names_att, all_parties, country, dataspace, year, coordtargetplot, vis="leadusers", seed_value=seed_value)
                        coordsplotted.append(selected_coords_names_att[1])
                
            if country == "us" and year == 2020:
                continue
            outpath = "{}/plots_upd_fixeddims_ridgecv/{}".format(dataspace, country)
            pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
            savename = "{}/{}_{}_{}_{}_hist.png".format(outpath, country, year, selected_coords_names[0], selected_coords_names[1])
            print('About to save the figure...')
            fig_users.savefig(savename,
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False) 
            fig_users.savefig(savename.replace("png","pdf"),
                dpi=300,
                bbox_inches='tight',
                transparent=False) 
            print('✅ Figure saved to {}'.format(savename))         
            savename = "{}/{}_{}_hist_att.png".format(outpath, country, year)
            print('About to save the figure...')
            fig_users_att.savefig(savename,
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False) 
            fig_users_att.savefig(savename.replace("png","pdf"),
                dpi=300,
                bbox_inches='tight',
                transparent=False) 
            print('✅ Figure saved to {}'.format(savename))    

            outpath = "{}/plots_upd_fixeddims_ridgecv/{}".format(dataspace, country)
            pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
            savename = "{}/{}_{}_{}_{}_hist_lead.png".format(outpath, country, year, selected_coords_names[0], selected_coords_names[1])
            print('About to save the figure...')
            fig_leadusers.savefig(savename,
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False) 
            fig_leadusers.savefig(savename.replace("png","pdf"),
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False) 
            print('✅ Figure saved to {}'.format(savename))         
            savename = "{}/{}_{}_hist_att_lead.png".format(outpath, country, year)
            print('About to save the figure...')
            fig_leadusers_att.savefig(savename,
                    dpi=300,
                    bbox_inches='tight',
                    transparent=False) 
            fig_leadusers_att.savefig(savename.replace("png","pdf"),
                dpi=300,
                bbox_inches='tight',
                transparent=False) 
            print('✅ Figure saved to {}'.format(savename))                 

    
    