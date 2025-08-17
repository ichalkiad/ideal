import os 

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["NUMBA_NUM_THREADS"] = "20"

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
                                            load_matrix

def create_density_overlay_plot(data=None, x_col='dimension_1', y_col='dimension_2', 
                               user_type_col='user_type', title='2D Ideal Points with Density'):
       
    fig = make_subplots(rows=1, cols=1)
    
    # Add scatter points for each user type
    colors = {'User': '#2E86AB', 'Main User': '#FFFF00', "Party": "#A23B72"}
    symbols = {'User': 'circle', 'Lead User': 'cross', "Party": "triangle-up"}
    
    for user_type in data[user_type_col].unique():
        subset = data[data[user_type_col] == user_type]
        
        # if user_type == "User":
        #     fig.add_trace(go.Scatter(
        #         x=subset[x_col],
        #         y=subset[y_col],
        #         mode='markers',
        #         name=user_type,
        #         marker=dict(
        #             symbol=symbols.get(user_type, 'circle'),
        #             size=8,
        #             color=colors.get(user_type, '#2E86AB'),
        #             opacity=0.6,
        #             line=dict(width=1, color='white')
        #         )
        #     ))
        
        if user_type == "Party":
            fig.add_trace(go.Scatter(
                x=subset[x_col],
                y=subset[y_col],
                mode='markers',
                name=user_type,
                marker=dict(
                    symbol=symbols.get(user_type, 'circle'),
                    size=10,
                    color=colors.get(user_type, '#2E86AB'),
                    opacity=0.8,
                    line=dict(width=1, color='white')
                )
            ))

        if user_type == "User":
            fig.add_trace(go.Histogram2dContour(
                x=subset[x_col],
                y=subset[y_col],
                name=f'{user_type} Density',
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors.get(user_type, '#2E86AB')]],
                showscale=False,
                opacity=0.5,
                line=dict(width=1, color=colors.get(user_type, '#2E86AB'))
            ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Latent dimension 1",
        yaxis_title="Latent dimension 2",
        template='plotly_white',
        width=900,
        height=700,
        font=dict(family="Arial", size=12)
    )

    ipdb.set_trace()
    
    return fig


if __name__ == "__main__":
    
    
    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1
    
    countries = ["finland", "france", "germany", "netherlands", "poland", "uk", "us"]
    dataspace = "/mnt/hdd2/ioannischalkiadakis/epodata_rsspaper/"

    for year in [2020, 2023]:
        for country in countries:

            datasets_names = [file.name for file in pathlib.Path(dataspace).iterdir() if file.is_file() and (country in file.name and str(year) in file.name and "mappings" in file.name)]
            if len(datasets_names) == 0:
                continue

            K = int(datasets_names[0].split("_")[3].replace("K", ""))
            J = int(datasets_names[0].split("_")[4].replace("J", ""))
            
            print(parallel, K, J, elementwise, evaluate_posterior)
            
            parameter_names = ["X", "Z"]
            d = 2  

            mappings, node_to_index_start, index_to_node_start, \
                    node_to_index_end, index_to_node_end, _ = load_matrix("{}/Y_{}_{}".format(dataspace, country, year), K, J)      
            
            mp_mapping = pd.read_csv("{}/parties_lead_users_{}_{}.csv".format(dataspace, country, year))
            all_parties = np.unique(mp_mapping.EPO_party_acronym.dropna().values).tolist()
            parties_politicians = dict()
            for party in all_parties:
                parties_politicians[party] = mp_mapping.loc[mp_mapping.EPO_party_acronym==party, "mp_pseudo_id"].values.tolist()
                       
            with jsonlines.open("{}/{}/estimation_CA_{}/params_out_global_theta_hat.jsonl".format(dataspace, country, year), mode="r") as f: 
                for result in f.iter(type=dict, skip_invalid=True):                    
                    param_hat = result["X"]
                    X_hat = np.asarray(param_hat).reshape((d, K), order="F").T                         
                    param_hat = result["Z"]
                    Z_hat = np.asarray(param_hat).reshape((d, J), order="F").T                         
                    break
            
            # parties in order of appearance in all_parties
            party_ideal_points_est = np.zeros((len(all_parties), d))
            for party in all_parties:
                z_loc = []
                for mp in parties_politicians[party]:
                    try:
                        z_loc.append(Z_hat[node_to_index_end[mp], :])                
                    except KeyError:
                        print("Lead user {} not found in mapping.".format(mp))
                        continue
                party_ideal_points_est[all_parties.index(party), :] = np.mean(np.stack(z_loc), axis=0)          

            data = pd.DataFrame({
                "dim1": np.concatenate([X_hat[:, 0], Z_hat[:, 0], party_ideal_points_est[:, 0]]),
                "dim2": np.concatenate([X_hat[:, 1], Z_hat[:, 1], party_ideal_points_est[:, 1]]),
                "user_type": ['User'] * K + ['Lead User'] * J + ["Party"] * party_ideal_points_est.shape[0],
                'user_id': range(K+J+party_ideal_points_est.shape[0])
            })   
            # ipdb.set_trace()
            # data = data.drop_duplicates(subset=['dim1', 'dim2'], keep='first').reset_index(drop=True)         
            # ipdb.set_trace()
            fig = create_density_overlay_plot(data=data, x_col='dim1', y_col='dim2', user_type_col='user_type', title='')
    


    