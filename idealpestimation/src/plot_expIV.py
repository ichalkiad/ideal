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
                                            load_matrix, fix_plot_layout_and_save
import plotly.express as px

def create_density_overlay_plot(data=None, x_col='dimension_1', y_col='dimension_2', country=None, year=None,
                               user_type_col='user_type', title='2D Ideal Points with Density', 
                               party_names=None, dirout="/tmp/"):

       
    # fig = go.Figure() #make_subplots(rows=1, cols=1)
   
    # Add scatter points for each user type
    colors = {'User': "#00F7FF", 'Lead User': '#FFFF00', "Party": "#A23B72"}
    symbols = {'User': 'circle', 'Lead User': 'cross', "Party": "triangle-up"}

    user_type = "User"
    fig = px.density_heatmap(data, x=x_col, y=y_col, nbinsx=50, nbinsy=50, 
                            color_continuous_scale=[[0, 'rgba(0,0,0,0)'], [1, colors.get(user_type, "#00F7FF")]],                             
                            marginal_x="histogram", marginal_y="histogram")
    for user_type in ["User", "Lead User", "Party"]:
        subset = data[data[user_type_col] == user_type]
        
        if user_type == "Lead User":
            fig.add_trace(go.Scatter(
                x=subset[x_col],
                y=subset[y_col],
                mode='markers',
                name=user_type,
                marker=dict(
                    symbol=symbols.get(user_type, 'circle'),
                    size=8,
                    color=colors.get(user_type, '#2E86AB'),
                    opacity=0.6,
                    line=dict(width=1, color='white')
                )
            ))
        
        if user_type == "Party":
            for pidx in range(subset.shape[0]):
                party_names = party_names if party_names is not None else ["Party_{}".format(i) for i in range(subset.shape[0])]
                fig.add_trace(go.Scatter(
                    x=[subset.iloc[pidx][x_col]],
                    y=[subset.iloc[pidx][y_col]],
                    mode='markers+text',
                    name=party_names[pidx],
                    text=party_names[pidx],
                    textposition="top center",
                    marker=dict(
                        symbol=symbols.get(user_type, 'triangle-up'),
                        size=15,
                        color=colors.get(user_type, '#A23B72'),
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    )
                ))
            # fig.add_trace(go.Scatter(
            #     x=subset[x_col],
            #     y=subset[y_col],
            #     mode='markers',
            #     name=party_names,
            #     marker=dict(
            #         symbol=symbols.get(user_type, 'circle'),
            #         size=10,
            #         color=colors.get(user_type, '#2E86AB'),
            #         opacity=0.8,
            #         line=dict(width=1, color='white')
            #     )
            # ))

        # if user_type == "User":
        #     fig.add_trace(go.Histogram2d(
        #         x=subset[x_col],
        #         y=subset[y_col],
        #         name=f'{user_type} Density',
        #         colorscale=[[0, 'rgba(0,0,0,0)'], [1, colors.get(user_type, "#00F7FF")]],
        #         showscale=True,
        #         # nbinsx=100,
        #         # opacity=0.9
        #     ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.4)
        
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        width=900,
        height=700,
        font=dict(family="Arial", size=12),
        legend=dict(
                orientation="h")
    )

    outpath = "{}/plots/{}".format(dirout, country)
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    savename = "{}/{}_{}_{}_{}.html".format(outpath, country, year, x_col, y_col)
    fix_plot_layout_and_save(fig, savename, xaxis_title=x_col, 
                            yaxis_title=y_col, title="{}-{}".format(country, year), 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)     
   
    return fig


if __name__ == "__main__":
    
    # standardise resulting CA dimensions
    
    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1
    
    countries = ["france", "finland", "germany", "netherlands", "poland", "uk"] #, "us"]
    dataspace = "/mnt/hdd2/ioannischalkiadakis/epodata_rsspaper/"

    for year in [2023, 2020]:
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
            # all_parties = np.unique(mp_mapping.EPO_party_acronym.dropna().values).tolist()
            if year == 2020:
                all_parties = np.unique(mp_mapping.CHES2019_party_acronym.dropna().values).tolist()
            elif year == 2023:
                all_parties = np.unique(mp_mapping.CHES2023_party_acronym.dropna().values).tolist()

            # linate map in same order as all_parties
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
                    linate_map_y.append(map_y_tmp)
                elif year == 2023:
                    parties_politicians[party] = mp_mapping.loc[mp_mapping.CHES2023_party_acronym==party, "mp_pseudo_id"].values.tolist()   
                    map_y_tmp = pd.read_csv("{}/y_party_ches2023_{}_{}.csv".format(dataspace, country, year))                    
                    map_y_tmp = map_y_tmp.loc[map_y_tmp.CHES2023_party_acronym==party, :].drop(columns=['CHES2023_party_acronym', "country", "electionyear",
                                                                                                        "EPO_party_acronym", "family", "in_gov"])
                    feature_names = map_y_tmp.columns.values.tolist()
                    map_y_tmp = map_y_tmp.values.flatten()
                    linate_map_y.append(map_y_tmp)
            linate_map_y = np.stack(linate_map_y)

            with jsonlines.open("{}/{}/estimation_CA_{}/params_out_global_theta_hat.jsonl".format(dataspace, country, year), mode="r") as f: 
                for result in f.iter(type=dict, skip_invalid=True):                    
                    param_hat = result["X"]
                    X_hat = np.asarray(param_hat).reshape((d, K), order="F").T                         
                    param_hat = result["Z"]
                    Z_hat = np.asarray(param_hat).reshape((d, J), order="F").T                         
                    break
            
            # standardise dimensions X_hat, Z_hat, over columns
            standardise_mean = np.vstack([X_hat, Z_hat]).mean(axis=0)
            standardise_std = np.vstack([X_hat, Z_hat]).std(axis=0)            
            X_hat = (X_hat - standardise_mean) / standardise_std
            Z_hat = (Z_hat - standardise_mean) / standardise_std
            
            linate_map_y = (linate_map_y - np.mean(linate_map_y, axis=0)) / np.std(linate_map_y, axis=0)
            y_map_tilde = linate_map_y.T
            y_map_tilde = np.vstack([y_map_tilde, np.ones(y_map_tilde.shape[1])])
            # party ideal points in estimated space, average over MPs of each party
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
            # same party order as y_map_tilde
            X_tilde_aff = party_ideal_points_est.T
            X_tilde_aff = np.vstack([X_tilde_aff, np.ones(X_tilde_aff.shape[1])])
            
            # compute affine map, P = party_ideal_points_est.shape[0], M+1: y_map_tilde.shape[0], N = d = 2
            # affine_map = np.linalg.lstsq(X_tilde.T, y_map_tilde.T, rcond=None)[0].T
            # ipdb.set_trace()
            affine_map = y_map_tilde @ X_tilde_aff.T @ np.linalg.inv(X_tilde_aff @ X_tilde_aff.T)

            party_tilde = party_ideal_points_est.T
            party_tilde = np.vstack([party_tilde, np.ones(party_tilde.shape[1])])
            # M feature coord + 1 intercept coord  X  party_num
            party_attitudinal_space = affine_map @ party_tilde

            z_tilde = Z_hat.T
            z_tilde = np.vstack([z_tilde, np.ones(z_tilde.shape[1])])
            z_attitudinal_space = affine_map @ z_tilde

            x_tilde = X_hat.T
            x_tilde = np.vstack([x_tilde, np.ones(x_tilde.shape[1])])
            x_attitudinal_space = affine_map @ x_tilde

            # ipdb.set_trace()

            # CHES2019:  0: 'lrecon', 2: 'antielite_salience', 28: 'civlib_laworder', 30: 'country', 36: 'lrgen', 47: 'people_vs_elite'
            # CHES2023:  5: 'lrecon', 0: 'antielite_salience'
            if year == 2020:
                selected_coords = [36, 47] # choose two CHES dimensions related to COVID-19 polarised debates, set dataframe names to the names of the coords
            elif year == 2023:
                selected_coords = [1, 3]
            selected_coords_names = ['lrecon', 'antielite_salience']
            
            # ipdb.set_trace()
            
            data_raw = pd.DataFrame({
                selected_coords_names[0]: np.concatenate([X_hat[:, 0], Z_hat[:, 0], party_ideal_points_est[:, 0]]),
                selected_coords_names[1]: np.concatenate([X_hat[:, 1], Z_hat[:, 1], party_ideal_points_est[:, 1]]),
                "user_type": ['User'] * K + ['Lead User'] * J + ["Party"] * party_ideal_points_est.shape[0],
                'user_id': range(K+J+party_ideal_points_est.shape[0])
            })   

            data_attitudinal = pd.DataFrame({
                selected_coords_names[0]: np.concatenate([x_attitudinal_space[selected_coords[0], :], z_attitudinal_space[selected_coords[0], :], 
                                        party_attitudinal_space[selected_coords[0], :]]).flatten(),
                selected_coords_names[1]: np.concatenate([x_attitudinal_space[selected_coords[1], :], z_attitudinal_space[selected_coords[1], :], 
                                        party_attitudinal_space[selected_coords[1], :]]).flatten(),
                "user_type": ['User'] * K + ['Lead User'] * J + ["Party"] * party_ideal_points_est.shape[0],
                'user_id': range(K+J+party_ideal_points_est.shape[0])
            })  
            
            # ipdb.set_trace()
            # data = data.drop_duplicates(subset=['dim1', 'dim2'], keep='first').reset_index(drop=True)         
            # ipdb.set_trace()
            fig = create_density_overlay_plot(data=data_raw, x_col=selected_coords_names[0], 
                                              country=country, year=year,
                                              y_col=selected_coords_names[1], 
                                              user_type_col='user_type', 
                                              title='', party_names=all_parties, 
                                              dirout=dataspace)
    


    