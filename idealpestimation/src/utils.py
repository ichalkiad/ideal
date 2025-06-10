from scipy.stats import qmc
import ipdb
import numpy as np
import math
import time
from datetime import timedelta, datetime
import pathlib
from scipy.stats import norm, multivariate_normal, invgamma
import plotly.graph_objs as go
import plotly.io as pio
import jax
import jax.numpy as jnp
from jax import jvp, grad
import jsonlines
from scipy.optimize import approx_fprime
import argparse
from scipy.special import gammaincc, gammaln
import itertools
from sklearn.decomposition import PCA
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots
from itertools import product
import numba
from numba import prange
from joblib import Parallel, delayed
import pandas as pd
from threadpoolctl import threadpool_info
import pickle
from prince import svd as ca_svd


def get_slurm_experiment_csvs(Ks, Js, sigma_es, M, batchsize, dir_in, dir_out):

    ca = {"trial":[], "K":[], "J":[], "sigma_e":[]}
    mle = {"trial":[], "K":[], "J":[], "sigma_e":[], "batchsize":[], "data_start":[], "data_end":[]}
    icm_poster = {"trial":[], "K":[], "J":[], "sigma_e":[]}
    icm_data = {"trial":[], "K":[], "J":[], "sigma_e":[], "batchsize":[], "data_start":[], "data_end":[]}
    for m in range(M):
        for K in Ks:
            for J in Js:
                for sigma_e in sigma_es:
                    ca["trial"].append(m)                    
                    icm_poster["trial"].append(m)                    
                    ca["K"].append(K)                    
                    icm_poster["K"].append(K)                    
                    ca["J"].append(J)                    
                    icm_poster["J"].append(J)                    
                    ca["sigma_e"].append(sigma_e)                    
                    icm_poster["sigma_e"].append(sigma_e)                 

                    # mle, icm_data batches
                    path = pathlib.Path("{}/data_K{}_J{}_sigmae{}/{}/{}/".format(dir_in, K, J, str(sigma_e).replace(".", ""), m, batchsize))
                    subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]               
                    for dataset_index in range(len(subdatasets_names)):               
                        mle["trial"].append(m)
                        icm_data["trial"].append(m)
                        mle["K"].append(K)
                        icm_data["K"].append(K)
                        mle["J"].append(J)
                        icm_data["J"].append(J)
                        mle["sigma_e"].append(sigma_e)
                        icm_data["sigma_e"].append(sigma_e)
                        mle["batchsize"].append(batchsize)
                        icm_data["batchsize"].append(batchsize)
                        subdataset_name = subdatasets_names[dataset_index]                            
                        start = int(subdataset_name.split("_")[1])
                        end = int(subdataset_name.split("_")[2])
                        mle["data_start"].append(start)
                        icm_data["data_start"].append(start)
                        mle["data_end"].append(end)
                        icm_data["data_end"].append(end)
    caout = pd.DataFrame.from_dict(ca)              
    caout.to_csv("{}/slurm_experimentI_ca.csv".format(dir_out), header=None, index=False)
    mleout = pd.DataFrame.from_dict(mle)              
    mleout.to_csv("{}/slurm_experimentI_mle.csv".format(dir_out), header=None, index=False)
    icmpout = pd.DataFrame.from_dict(icm_poster)              
    icmpout.to_csv("{}/slurm_experimentI_icm_poster.csv".format(dir_out), header=None, index=False)
    icmdout = pd.DataFrame.from_dict(icm_data)              
    icmdout.to_csv("{}/slurm_experimentI_icm_data.csv".format(dir_out), header=None, index=False)



def print_threadpool_info():
    print("\n📊 Threadpool Info:")
    for lib in threadpool_info():
        print(f"- {lib['prefix']} ({lib['internal_api']}) → Threads: {lib['num_threads']}")

def fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=False,
                            print_png=True, print_html=True, print_pdf=True):
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(title=title, plot_bgcolor='rgb(255,255,255)',
                        yaxis=dict(
                            title=yaxis_title,
                            titlefont_size=20,
                            tickfont_size=20,
                            showgrid=showgrid,
                        ),
                        xaxis=dict(
                            title=xaxis_title,
                            titlefont_size=20,
                            tickfont_size=20,
                            showgrid=showgrid
                        ),
                        font=dict(
                            size=20
                        ),
                        showlegend=showlegend)
        if showlegend:
            fig.update_layout(legend=dict(
                yanchor="top",
                y=1.1,  # 0.01
                xanchor="right",  # "left", #  "right"
                x=1,    #0.01,  # 0.99
                bordercolor="Black",
                borderwidth=0.3,
                font=dict(
                    size=18,
        )))

        if print_html:
            pio.write_html(fig, savename, auto_open=False)
        if print_pdf:
            pio.write_image(fig, savename.replace(".html", ".pdf"), engine="kaleido")
        if print_png:
            pio.write_image(fig, savename.replace("html", "png"), width=1540, height=871, scale=1)


def norm_logcdf_thresholdapprox(x, loc, scale, threshold_std_no=4):
    
    x1 = x.reshape((x.shape[0]*x.shape[1],), order="F")
    print(x1.mean(), x1.std())
    xx = (x1-x1.mean())/x1.std()

    min_logcdf_val = norm.logcdf(-threshold_std_no, loc=0, scale=1)
    max_logcdf_val = norm.logcdf(threshold_std_no, loc=0, scale=1)

    if isinstance(x, np.ndarray):
        res = np.zeros(xx.shape)
    
    ltval = np.nonzero(xx <= -threshold_std_no)
    gtval = np.nonzero(xx >= threshold_std_no)
    inval = np.nonzero((xx > -threshold_std_no) & (xx < threshold_std_no))

    if len(ltval) > 0:
        # print([i for i in x[ltval]])
        res[ltval] = min_logcdf_val
        # ipdb.set_trace()
    if len(gtval) > 0:
        res[gtval] = max_logcdf_val
    if len(inval) > 0:
        res[inval] = norm.logcdf(xx[inval], loc=0, scale=1)
    
    res = res.reshape((x.shape[0], x.shape[1]), order="F")
    # ipdb.set_trace()    
    # t0 = time.time()
    # norm.logcdf(x, loc, scale)
    # print(str(timedelta(seconds=time.time()-t0)))    
    # t0 = time.time()
    # norm.logcdf(xx, loc=0, scale=1)
    # print(str(timedelta(seconds=time.time()-t0)))
    # ipdb.set_trace()
    
    return res, ltval


def test_fastapprox_cdf(parameter_names, data_location, m, K, J, d, args=None):

    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))

    param_positions_dict = dict()            
    k = 0
    for param in parameter_names:
        if param == "X":
            param_positions_dict[param] = (k, k + K*d)                       
            k += K*d    
        elif param in ["Z"]:
            param_positions_dict[param] = (k, k + J*d)                                
            k += J*d
        elif param in ["Phi"]:            
            param_positions_dict[param] = (k, k + J*d)                                
            k += J*d
        elif param == "beta":
            param_positions_dict[param] = (k, k + K)                                   
            k += K    
        elif param == "alpha":
            param_positions_dict[param] = (k, k + J)                                       
            k += J    
        elif param == "gamma":
            param_positions_dict[param] = (k, k + 1)                                
            k += 1
        elif param == "delta":
            param_positions_dict[param] = (k, k + 1)                                
            k += 1
        elif param == "sigma_e":
            param_positions_dict[param] = (k, k + 1)                                
            k += 1

    with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
        for result in f.iter(type=dict, skip_invalid=True):
            for param in parameter_names:
                theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
    
    rng = np.random.default_rng()
    theta_samples_list = None 
    idx_all = None
    nonscaled = []
    scaled = []
    while True:
        try:
            # if idx_all is not None:
            #     ipdb.set_trace()
            theta_true, theta_samples_list, idx_all = sample_theta_curr_init(parameter_space_dim, 2, param_positions_dict, 
                                                                args, samples_list=theta_samples_list, idx_all=idx_all, rng=rng)
            # print(len(idx_all))
        except:
            break
    
        params_hat = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        sigma_e = params_hat["sigma_e"]

        pijs = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K)     
    
        # print("Non-scaled") 
        t0 = time.time()
        logcdfs = norm.logcdf(pijs, loc=0, scale=sigma_e)
        t1 = time.time()
        # print(str(timedelta(seconds=t1-t0)))
        nonscaled.append(t1-t0)
        pijs1 = pijs.reshape((pijs.shape[0]*pijs.shape[1],), order="F")    
        pijs1 = (pijs1-pijs1.mean())/pijs1.std()
        pijs1 = pijs1.reshape((pijs.shape[0], pijs.shape[1]), order="F")    

        # print("Scaled") 
        t0 = time.time()
        # logcdfs = norm.logcdf(pijs, loc=0, scale=sigma_e)  
        logcdfs = norm.logcdf(pijs1, loc=0, scale=1)
        # log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=0, variance=sigma_e)
        t1 = time.time()
        # print(str(timedelta(seconds=t1-t0)))
        scaled.append(t1-t0)
        # print("\n\n")

    
    print(str(timedelta(seconds=np.percentile(nonscaled, q=50, method="lower"))))
    print(str(timedelta(seconds=np.percentile(scaled, q=50, method="lower"))))
    import sys
    sys.exit(0)

    t0 = time.time()
    logcdfs_fast, idxval = norm_logcdf_thresholdapprox(pijs1, 0, sigma_e, threshold_std_no=3)    
    # log1mcdfs_fast = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=0, variance=sigma_e, approxfast=False, threshold_std_no=3)
    print(str(timedelta(seconds=time.time()-t0)))
    
    print(np.allclose(logcdfs.reshape((K*J,), order="F")[idxval], logcdfs_fast.reshape((K*J,), order="F")[idxval]))
    # print(np.allclose(log1mcdfs[idxval], log1mcdfs_fast[idxval]))
    print(logcdfs.reshape((K*J,), order="F")[idxval])
    print(logcdfs_fast.reshape((K*J,), order="F")[idxval])
    # print(logcdfs_fast[inval])
    

def plot_posterior_vec_runtimes(files, names, outdir="/tmp/"):

    fig = go.Figure()
    for i in range(len(files)):
        data = {"data": []}  
        filein = files[i]
        name = names[i]      
        with jsonlines.open(filein, mode="r") as f:              
            for result in f.iter(type=dict, skip_invalid=True):
                if np.isnan(result["milliseconds"]) or not isinstance(result["milliseconds"], float):
                    continue
                data["data"].append(result["milliseconds"])
            df = pd.DataFrame.from_dict(data)
            lower_bound = df.data.quantile(0.05)
            upper_bound = df.data.quantile(0.95)
            df_filtered = df.data[(df.data >= lower_bound) & (df.data <= upper_bound)].dropna()
            fig.add_trace(go.Box(
                            y=df_filtered.values.flatten().tolist(), 
                            showlegend=True, opacity=1.0,                            
                            boxpoints="outliers", name=name
                        ))
    savename = "{}/posterior_timings_parallelblock_vs_numbaonlycomputation.html".format(outdir)
    pathlib.Path("{}/".format(outdir)).mkdir(parents=True, exist_ok=True)     
    fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="milliseconds", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, print_pdf=False)

def plot_loglik_runtimes(filerec, outdir="/tmp/"):

    rows_100k_nonparallel = {"block_rows_500": [], "block_rows_1000": []}
    rows_1M_nonparallel = {"block_rows_3000": [], "block_rows_5000": []}
    rows_100k_parallel = {"block_rows_500": [], "block_rows_1000": []}
    rows_1M_parallel = {"block_rows_3000": [], "block_rows_5000": []}
    with jsonlines.open(filerec, mode="r") as f:              
        for result in f.iter(type=dict, skip_invalid=True):
            if np.isnan(result["milliseconds"]) or not isinstance(result["milliseconds"], float):
                continue
            if result["parallel"] == 1:                
                if result["K"] == 3004:
                    if result["block_size_rows"] == 500:
                        rows_100k_parallel["block_rows_500"].append(result["milliseconds"])
                    else:
                        rows_100k_parallel["block_rows_1000"].append(result["milliseconds"])
                else:
                    if result["block_size_rows"] == 3000:
                        rows_1M_parallel["block_rows_3000"].append(result["milliseconds"])
                    else:
                        rows_1M_parallel["block_rows_5000"].append(result["milliseconds"])
            else:
                if result["K"] == 3004:
                    if result["block_size_rows"] == 500:
                        rows_100k_nonparallel["block_rows_500"].append(result["milliseconds"])
                    else:
                        rows_100k_nonparallel["block_rows_1000"].append(result["milliseconds"])
                else:
                    if result["block_size_rows"] == 3000:
                        rows_1M_nonparallel["block_rows_3000"].append(result["milliseconds"])
                    else:
                        rows_1M_nonparallel["block_rows_5000"].append(result["milliseconds"])
   
    fig = go.Figure()
    fig100k = go.Figure()
    fig1M = go.Figure()
    for par in [0, 1]:
        for Ksize in ["100k", "1M"]:
            if par == 0:
                market_color = "red"
                if Ksize == "100k":
                    opacity = 0.5
                    loaddict = rows_100k_nonparallel
                else:
                    opacity = 1
                    loaddict = rows_1M_nonparallel
            else:
                market_color = "green"
                if Ksize == "100k":
                    opacity = 0.5
                    loaddict = rows_100k_parallel
                else:
                    opacity = 1
                    loaddict = rows_1M_parallel
            for keys in loaddict.keys():                
                df = pd.DataFrame.from_dict({"vals":loaddict[keys]})
                lower_bound = df.vals.quantile(0.05)
                upper_bound = df.vals.quantile(0.95)
                df_filtered = df.vals[(df.vals >= lower_bound) & (df.vals <= upper_bound)].dropna()
                if "500" in keys and "5000" not in keys:
                    name = "500x100"
                elif "1000" in keys:
                    name = "1000x100"
                elif "3000" in keys:
                    name = "3000x100"
                elif "5000" in keys:
                    name = "5000x100"                
                fig.add_trace(go.Box(
                                y=df_filtered.values.flatten().tolist(), 
                                showlegend=True, opacity=opacity, marker_color=market_color,
                                # x=x, 
                                boxpoints="outliers", name=name
                            ))
                if Ksize == "100k":
                    fig100k.add_trace(go.Box(
                                y=df_filtered.values.flatten().tolist(), 
                                showlegend=True, opacity=1, marker_color=market_color,
                                # x=x, 
                                boxpoints="outliers", name=name
                            ))
                else:
                    fig1M.add_trace(go.Box(
                                y=df_filtered.values.flatten().tolist(), 
                                showlegend=True, opacity=1, marker_color=market_color,
                                # x=x, 
                                boxpoints="outliers", name=name
                            ))

    fig.update_layout(
        boxmode='group'
    )
    savename = "{}/likelihood_timings_parallel.html".format(outdir)
    pathlib.Path("{}/".format(outdir)).mkdir(parents=True, exist_ok=True)     
    fix_plot_layout_and_save(fig, savename, xaxis_title="Block sizes", yaxis_title="milliseconds", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, print_pdf=False)

    fig100k.update_layout(
        boxmode='group'
    )
    savename = "{}/likelihood_timings_parallel_100k.html".format(outdir)
    pathlib.Path("{}/".format(outdir)).mkdir(parents=True, exist_ok=True)     
    fix_plot_layout_and_save(fig100k, savename, xaxis_title="Block sizes", yaxis_title="milliseconds", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, print_pdf=False)
    fig1M.update_layout(
        boxmode='group'
    )
    savename = "{}/likelihood_timings_parallel_1M.html".format(outdir)
    pathlib.Path("{}/".format(outdir)).mkdir(parents=True, exist_ok=True)     
    fix_plot_layout_and_save(fig1M, savename, xaxis_title="Block sizes", yaxis_title="milliseconds", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, print_pdf=False)

####################### MLE #############################

def params2optimisation_dict(J, K, d, parameter_names, X, Z, Phi, alpha, beta, gamma, delta, sigma_e):
    
    param_positions_dict = dict()
    optim_vector = []
    k = 0
    for param in parameter_names:
        if param == "X":
            param_positions_dict[param] = (k, k + K*d)   
            Xvec = X.reshape((d*K,), order="F").tolist()        
            optim_vector.extend(Xvec)            
            k += K*d    
        elif param == "Z":
            param_positions_dict[param] = (k, k + J*d)            
            Zvec = Z.reshape((d*J,), order="F").tolist()         
            optim_vector.extend(Zvec)            
            k += J*d
        elif param == "Phi":            
            param_positions_dict[param] = (k, k + J*d)            
            Phivec = Phi.reshape((d*J,), order="F").tolist()         
            optim_vector.extend(Phivec)            
            k += J*d
        elif param == "beta":
            param_positions_dict[param] = (k, k + K)               
            optim_vector.extend(beta.tolist())            
            k += K    
        elif param == "alpha":
            param_positions_dict[param] = (k, k + J)               
            optim_vector.extend(alpha.tolist())                        
            k += J    
        elif param == "gamma":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(gamma)
            k += 1
        elif param == "delta":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(delta)
            k += 1
        elif param == "sigma_e":
            param_positions_dict[param] = (k, k + 1)            
            optim_vector.append(sigma_e)
            k += 1

    return optim_vector, param_positions_dict


def optimisation_dict2params(optim_vector, param_positions_dict, J, K, d, parameter_names):
    
    params_out = dict()
    for param in parameter_names:
        param_out = optim_vector[param_positions_dict[param][0]:param_positions_dict[param][1]]
        if param == "X":            
            param_out = param_out.reshape((d, K), order="F")                     
        elif param in ["Z"]:            
            param_out = param_out.reshape((d, J), order="F")                      
        elif param in ["Phi"]:            
            param_out = param_out.reshape((d, J), order="F")                                
        params_out[param] = param_out
        
    return params_out

def optimisation_dict2paramvectors(optim_vector, param_positions_dict, J, K, d, parameter_names):
    
    params_out = dict()
    for param in parameter_names:        
        param_out = optim_vector[param_positions_dict[param][0]:param_positions_dict[param][1]]                
        params_out[param] = param_out.tolist()
        
    return params_out


def create_constraint_functions(n, param_positions_dict=None, sum_z_constant=0, min_sigma_e=1e-6, args=None, target_param=None, target_idx=None):
    
    if args is not None:
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method,\
            parameter_names, J, K, d, N, dst_func, niter, parameter_space_dim, m, penalty_weight_Z,\
                constant_Z, retries, parallel, min_sigma_e, prior_loc_x, prior_scale_x, prior_loc_z,\
                    prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha,\
                        prior_scale_alpha, prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta,\
                            prior_loc_sigmae, prior_scale_sigmae, param_positions_dict, rng, batchsize, _ = args
        
        grid_width_std = 5
        bounds = []
        if target_param is not None:
            if target_param == "X":
                if target_idx is None:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0])]*(K*d)) 
                else:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0])]) 
            elif target_param == "Z":
                if target_idx is None:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0])]*(J*d)) 
                else:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0])])
            elif target_param == "Phi":
                if target_idx is None:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0])]*(J*d)) 
                else:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0])]) 
            elif target_param == "alpha":
                if target_idx is None:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha)]*J) 
                else:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha)]) 
            elif target_param == "beta":
                if target_idx is None:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta)]*K) 
                else:
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta)]) 
            elif target_param == "gamma":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma)) 
            elif target_param == "delta":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta)) 
            elif target_param == "sigma_e":
                bounds.append((min_sigma_e, None))
        else:
            for param in parameter_names:
                if param == "X":
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0])]*(K*d)) 
                elif param == "Z":
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0])]*(J*d)) 
                elif param == "Phi":
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0])]*(J*d)) 
                elif param == "alpha":
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha)]*J) 
                elif param == "beta":
                    bounds.extend([(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta)]*K) 
                elif param == "gamma":
                    bounds.append((-grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma)) 
                elif param == "delta":
                    bounds.append((-grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta)) 
                elif param == "sigma_e":
                    bounds.append((min_sigma_e, None))
    else:
        # def sum_zero_constraint(x):
        #     """Constraint: Sum of Z's should be a constant - default 0, i.e. balanced politician set"""
        #     return np.sum(x[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]])
        bounds = [(None, None)]*(n-1)
        bounds.append((min_sigma_e, None))
    
    return bounds, None


def initialise_optimisation_vector_sobol(m=16, J=2, K=2, d=1, min_sigma_e=0.1):

    sobol_generators = dict()
    # sampler = qmc.Sobol(d=1, scramble=False)   
    # sample = sampler.random_base2(m=m)
    # gamma = sample[:int(len(sample)/2)]
    # delta = sample[int(len(sample)/2):]
    # sobol_generators["gammadelta"] = [sampler]
    
    # gamma-delta: unidimensional parameters, generate uniform grid
    gamma = np.linspace(-2, 2, 100).tolist()
    if 0 in gamma:
        gamma.remove(0)
    delta = np.linspace(-2, 2, 100).tolist()
    if 0 in delta:
        delta.remove(0)        
    
    # sampler = qmc.Sobol(d=d, scramble=False)   
    # sample = sampler.random_base2(m=2)                               
    # c = sample[0]    
    # sobol_generators["c"] = [sampler]

    # K will be quite high dimensional so better use Sobol sequence
    sampler = qmc.Sobol(d=K, scramble=False)   
    sample = sampler.random_base2(m=m)                               
    beta = sample.tolist()
    sobol_generators["beta"] = [sampler]

    # J will be quite high dimensional so better use Sobol sequence
    sampler = qmc.Sobol(d=J, scramble=False)   
    sample = sampler.random_base2(m=m)                               
    alpha = sample.tolist()
    sobol_generators["alpha"] = [sampler]

    # Could also use uniform grid, given that d=2
    sampler = qmc.Sobol(d=d, scramble=False)   
    sample = sampler.random_base2(m=m)
    Phi = sample[:int(len(sample)/3), :].tolist()    
    Z = sample[int(len(sample)/3):2*int(len(sample)/3), :].tolist()    
    X = sample[2*int(len(sample)/3):, :].tolist()
    sobol_generators["XZPhi"] = [sampler]
            
    sigma_e = np.linspace(min_sigma_e, 5, 100).tolist()

    return X, Z, Phi, alpha, beta, gamma, delta, sigma_e

def visualise_hessian(hessian, title='Hessian matrix'):
    
    hessian = np.asarray(hessian)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=hessian,
        colorscale='RdBu_r',  
        zmin=np.min(np.diag(hessian)), 
        zmax=np.max(np.diag(hessian)),
        colorbar=dict(title='Hessian Value')
    ))
    
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title='Parameter Index',
        yaxis_title='Parameter Index'        
    )
    
    return fig

def uni_gaussian_hessian(mu, sigma, x):
    
    x = np.asarray(x)    
    z = (x - mu) / sigma
    
    # First parameter (mu): [d²/d2mu]
    d2l_dmu2 = len(x) / (sigma**2)
    # print(d2l_dmu2)
    # Second parameter (sigma): [d²/d2σ]
    d2l_d2sigma = -len(x) / (sigma**2) + 3*np.sum(z**2) / (sigma**2)
    # print(d2l_d2sigma)
    # Mixed partial derivative [d²/dμdσ]
    d2l_dmu_dsigma2 = 2*np.sum(z) / (sigma**2)
    # print(d2l_dmu_dsigma2)
        
    hessian = np.array([
        [d2l_dmu2, d2l_dmu_dsigma2],
        [d2l_dmu_dsigma2, d2l_d2sigma]
    ])
    
    return hessian

def get_jacobian(params, likelihood_function):
    """Numerical approximation of Jacobian"""
    fprime = approx_fprime(params, likelihood_function)    
    return fprime

def hvp(f, x, v):
  return jvp(grad(f), (x,), (v,))[1]

def get_hessian_diag_jax(f, x):
    # f: function w.r.t to parameter vector x
    return hvp(f, x, jnp.ones_like(x))

def rank_and_return_best_theta(estimated_thetas, Y, J, K, d, parameter_names, dst_func, param_positions_dict, DIR_out, args):
    
    best_theta = None
    computed_loglik = []
    for theta_set in estimated_thetas:
        theta = theta_set[0]
        loglik, _ = log_full_posterior(Y, theta.copy(), param_positions_dict, args)
        computed_loglik.append(loglik)
    # sort in increasing order, i.e. from worst to best solution
    sorted_idx = np.argsort(np.asarray(computed_loglik))
    sorted_idx_lst = sorted_idx.tolist() 
    best_theta = estimated_thetas[sorted_idx_lst[-1]][0]
    best_theta_var = estimated_thetas[sorted_idx_lst[-1]][1]
    current_pid = estimated_thetas[sorted_idx_lst[-1]][2]
    timestamp = estimated_thetas[sorted_idx_lst[-1]][3]
    eltime = estimated_thetas[sorted_idx_lst[-1]][4]
    hours = estimated_thetas[sorted_idx_lst[-1]][5]
    retry = estimated_thetas[sorted_idx_lst[-1]][6]
    success = estimated_thetas[sorted_idx_lst[-1]][7]
    varstatus = estimated_thetas[sorted_idx_lst[-1]][8]
    
    return best_theta, best_theta_var, current_pid, timestamp, eltime, hours, retry, success, varstatus

def combine_estimate_variance_rule(DIR_out, J, K, d, parameter_names, sq_error_dict, sq_error_nonrotated_dict, error_dict, error_nonrotated_dict,
                                   theta_true, param_positions_dict, seedint=1234):

    params_out = dict()
    params_out["X"] = np.zeros((d*K,))
    params_out["beta"] = np.zeros((K,))    
    for param in parameter_names:             
        weighted_estimate = None
        weight = None
        theta = None
        all_weights = []
        all_estimates = []
        path = pathlib.Path(DIR_out)  
        subdatasets_names = [file.name for file in path.iterdir() if not file.is_file() and "dataset_" in file.name]                    
        for dataset_index in range(len(subdatasets_names)):                    
            subdataset_name = subdatasets_names[dataset_index]                        
            DIR_read = "{}/{}/estimation/".format(DIR_out, subdataset_name)
            path = pathlib.Path(DIR_read)  
            estimates_names = [file.name for file in path.iterdir() if file.is_file() and "estimationresult_dataset" in file.name]
            if len(estimates_names) > 1:
                raise AttributeError("Should have 1 output estimation file.")
            for estim in estimates_names:
                with jsonlines.open("{}/{}".format(DIR_read, estim), mode="r") as f: 
                    for result in f.iter(type=dict, skip_invalid=True):
                        if result["mle_estimation_status"] is False or \
                            result["variance_estimation_status"] is None or\
                                result["variance_estimation_status"] is False or result["local theta"] is None:
                            continue
                        if param in ["X", "beta"]:
                            # single estimate per data split
                            theta = result[param]
                            namesplit = estim.split("_")
                            start = int(namesplit[2])
                            end   = int(namesplit[3].replace(".jsonl", ""))
                            if param == "X":
                                params_out[param][start*d:end*d] = theta
                                X_true = np.asarray(theta_true[param_positions_dict[param][0]+start*d:param_positions_dict[param][0]+end*d]).reshape((d, end-start), order="F")
                                X_hat = np.asarray(theta).reshape((d, end-start), order="F")
                                Rx, tx, mse_trial_m_batch_index, mse_nonrotated_trial_m_batch_index, err_trial_m_batch_index, err_nonrotated_trial_m_batch_index =\
                                    get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)
                            else:
                                params_out[param][start:end] = theta      
                                rel_err = (theta_true[param_positions_dict[param][0]+start:param_positions_dict[param][0]+end] - theta)/theta_true[param_positions_dict[param][0]+start:param_positions_dict[param][0]+end]                      
                                err_trial_m_batch_index = np.mean(rel_err)
                                mse_trial_m_batch_index = np.mean(rel_err**2)
                        else:                            
                            weight = result["variance_{}".format(param)]                            
                            theta = result[param]
                            all_weights.append(weight)
                            all_estimates.append(theta)
                            if param == "Z":
                                Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")  
                                Z_hat = np.asarray(theta).reshape((d, J), order="F")
                                Rz, tz, mse_trial_m_batch_index, mse_nonrotated_trial_m_batch_index, err_trial_m_batch_index, err_nonrotated_trial_m_batch_index =\
                                    get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)
                            else:
                                rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - theta)/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                                err_trial_m_batch_index = rel_err
                                mse_trial_m_batch_index = np.mean(rel_err**2)
            
            sq_error_dict[param].append(mse_trial_m_batch_index)
            if param in ["X", "Z", "Phi"]:
                sq_error_nonrotated_dict[param].append(mse_nonrotated_trial_m_batch_index)
            else: 
                # for parameters that are not rotated in any case, just add the same estimate
                sq_error_nonrotated_dict[param].append(mse_trial_m_batch_index)

            error_dict[param].append(err_trial_m_batch_index)
            if param in ["X", "Z", "Phi"]:
                error_nonrotated_dict[param].append(err_nonrotated_trial_m_batch_index)
            else: 
                # for parameters that are not rotated in any case, just add the same estimate
                error_nonrotated_dict[param].append(err_trial_m_batch_index)
            
            
        if param in ["X", "beta"]:
            params_out[param] = params_out[param]        
        else:             
            all_weights = np.stack(all_weights)
            if param not in ["Z", "Phi", "alpha"]:
                all_weights = all_weights.flatten()
            all_estimates = np.stack(all_estimates)
            # sum acrocs each coordinate's weight
            all_weights_sum = np.sum(all_weights, axis=0)
            all_weights_norm = all_weights/all_weights_sum
            assert np.allclose(np.sum(all_weights_norm, axis=0), np.ones(all_weights_sum.shape))
            # element-wise multiplication
            weighted_estimate = np.sum(all_weights_norm*all_estimates, axis=0)
            params_out[param] = weighted_estimate
    
    return params_out, sq_error_dict, sq_error_nonrotated_dict, error_dict, error_nonrotated_dict

def parse_input_arguments():

    parser = argparse.ArgumentParser(
        description='Input values: trials, either integer or range of values, e.g. 0-5, K, J, sigmae as a string without a decimal point, e.g. 0.5 -> 05.\
            Pass all arguments otherwise those hardcoded will be used.'
    )
    
    # Add optional arguments of different types with default values
    parser.add_argument(
        '--trials',
        type=str,
        default=None,
        help='Trial folders to run.'
    )
    
    parser.add_argument(
        '--K',
        type=int,
        default=None,
        help='K users'
    )
    
    parser.add_argument(
        '--J',
        type=int,
        default=None,
        help='J users'
    )

    parser.add_argument(
        '--sigmae',
        type=str,
        default=None,
        help='sigma_e'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='parallel execution flag - omit for single thread execution'
    )

    parser.add_argument(
        '--elementwise',
        action='store_true',
        help='for ICM: elementwise posterior optimisation'
    )

    parser.add_argument(
        '--evaluate_posterior',
        action='store_true',
        help='for ICM: evaluate posterior on a grid (True), differentiate (False)'
    )

    parser.add_argument(
        '--total_running_processes',
        type=int,
        default=10,
        help='total number of running processes in parallel'
    )
    
    return parser.parse_args()

def negative_loglik_coordwise(theta_coordval, theta_idx, full_theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, 
                              penalty_weight_Z, constant_Z, debug=False, numbafast=True):

    full_theta[theta_idx] = theta_coordval
    params_hat = optimisation_dict2params(full_theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    errscale = sigma_e
    errloc = mu_e          
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")    
    _nll = 0
    if debug:
        for i in range(K):
            for j in range(J):
                pij_arg = p_ij_arg(i, j, full_theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
                philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, variance=errscale)
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
    
    if numbafast:    
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")         
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        pij_arg = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K)     
    else:
        pij_arg = p_ij_arg(None, None, full_theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
    
    philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
    log_one_minus_cdf = log_complement_from_log_cdf_vec(philogcdf, pij_arg, mean=errloc, variance=errscale)
    nll = np.sum(Y*philogcdf + (1-Y)*log_one_minus_cdf)

    if debug:
        assert(np.allclose(nll, _nll))

    # sum_Z_J_vectors = np.sum(Z, axis=1)    
    return -nll #+ penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)

def process_blocks_parallel(Y, X, Z, alpha, beta, gamma, K, J, errloc, errscale, row_blocks, col_blocks,
                            block_size_rows, block_size_cols, njobs=20, prior=None):
    
    nll = 0.0
    blocks = []
    if col_blocks == -1:
        # index of vector j
        col_start = block_size_cols
        col_end = col_start + 1
        for i in range(row_blocks):
            row_start = i * block_size_rows
            row_end = min(row_start + block_size_rows, K)
            blocks.append((row_start, row_end, col_start, col_end))
        results = Parallel(n_jobs=njobs)(
            delayed(process_single_block)(Y[row_start:row_end, col_start:col_end], 
                                            X[:, row_start:row_end] , Z[:, col_start], 
                                            alpha[col_start], beta[row_start:row_end], 
                                            gamma, errloc, errscale, row_end-row_start, prior) 
                                        for row_start, row_end, col_start, col_end in blocks
        )
    else:
        for i in range(row_blocks):       
            for j in range(col_blocks):
                row_start = i * block_size_rows
                row_end = min(row_start + block_size_rows, K)
                col_start = j * block_size_cols
                col_end = min(col_start + block_size_cols, J)                       
                blocks.append((row_start, row_end, col_start, col_end))
    
        results = Parallel(n_jobs=njobs)(
            delayed(process_single_block)(Y[row_start:row_end, col_start:col_end], 
                                            X[:, row_start:row_end] , Z[:, col_start:col_end], 
                                            alpha[col_start:col_end], beta[row_start:row_end], 
                                            gamma, errloc, errscale, row_end-row_start, prior) 
                                        for row_start, row_end, col_start, col_end in blocks
        )
    nll = np.sum(results)

    return nll

def process_single_block(Y, X, Z, alpha, beta, gamma, errloc, errscale, block_size_rows, prior=None):
    
    if len(Z.shape) == 1:
        pij_arg = p_j_arg_numbafast(X, Z, alpha, beta, gamma)          
    else:
        pij_arg = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, block_size_rows) 
    philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
    log_one_minus_cdf = log_complement_from_log_cdf_vec(philogcdf, pij_arg, mean=errloc, variance=errscale)    

    if prior is not None:
        nll = np.sum(Y*philogcdf + (1-Y)*log_one_minus_cdf + prior)    
    else:
        nll = np.sum(Y*philogcdf + (1-Y)*log_one_minus_cdf)    
            
    return nll

def negative_loglik_coordwise_parallel(theta_coordval, theta_idx, full_theta, Y, J, K, d, 
                                    parameter_names, dst_func, param_positions_dict, 
                                    penalty_weight_Z, constant_Z, debug=False, numbafast=True,
                                    block_size_rows=500, block_size_cols=100):

    full_theta[theta_idx] = theta_coordval
    params_hat = optimisation_dict2params(full_theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    errscale = sigma_e
    errloc = mu_e          
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")    
    _nll = 0
    if debug:
        for i in range(K):
            for j in range(J):
                pij_arg = p_ij_arg(i, j, full_theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
                philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, variance=errscale)
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf

    if numbafast and debug:    
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")         
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        t0 = time.time()
        pij_arg = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K)        
        philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
        log_one_minus_cdf = log_complement_from_log_cdf_vec(philogcdf, pij_arg, mean=errloc, variance=errscale)
        nll_nonparallel = np.sum(Y*philogcdf + (1-Y)*log_one_minus_cdf)
        elapsedtime = timedelta(seconds=time.time()-t0)            
        total_seconds = int(elapsedtime.total_seconds())
        milliseconds = (time.time()-t0)*1000 
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        # out_file = "/mnt/hdd2/ioannischalkiadakis/idealdata/timings_parallellikelihood.jsonl"
        # with open(out_file, 'a') as f:         
        #     writer = jsonlines.Writer(f)
        #     writer.write({"K":K, "J":J, "block_size_rows": block_size_rows, 
        #                   "block_size_cols":block_size_cols, "parallel": 0, 
        #                   "hours": hours, "minutes": minutes, "seconds":total_seconds, "milliseconds":milliseconds})
    
    row_blocks = (K + block_size_rows - 1) // block_size_rows
    col_blocks = (J + block_size_cols - 1) // block_size_cols
    t0 = time.time()
    nll = process_blocks_parallel(Y, X, Z, alpha, beta, gamma, K, J, errloc, errscale, 
                                row_blocks, col_blocks, block_size_rows, block_size_cols)
    if debug:
        elapsedtime = timedelta(seconds=time.time()-t0)    
        milliseconds = (time.time()-t0)*1000         
        total_seconds = int(elapsedtime.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        # out_file = "/mnt/hdd2/ioannischalkiadakis/idealdata/timings_parallellikelihood.jsonl"
        # with open(out_file, 'a') as f:         
        #     writer = jsonlines.Writer(f)
        #     writer.write({"K":K, "J":J, "block_size_rows":block_size_rows, 
        #                 "block_size_cols":block_size_cols, "parallel":1, 
        #                 "hours": hours, "minutes": minutes, "seconds":total_seconds, "milliseconds":milliseconds})

    if debug:
        assert(np.allclose(nll, _nll))
        assert(np.allclose(nll_nonparallel, nll))
  
    return -nll


def negative_loglik_parallel(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, 
                            constant_Z, debug=False, numbafast=True, block_size_rows=2000, block_size_cols=100):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    errscale = sigma_e
    errloc = mu_e          
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")    
    _nll = 0
    if debug:
        for i in range(K):
            for j in range(J):
                pij_arg = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
                philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, variance=errscale)
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
   
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")         
    gamma = params_hat["gamma"][0]
    alpha = params_hat["alpha"]
    beta = params_hat["beta"]    
    row_blocks = (K + block_size_rows - 1) // block_size_rows
    col_blocks = (J + block_size_cols - 1) // block_size_cols    
    nll = process_blocks_parallel(Y, X, Z, alpha, beta, gamma, K, J, errloc, errscale, 
                                row_blocks, col_blocks, block_size_rows, block_size_cols)
    if debug:
        assert(np.allclose(nll, _nll))            

    return -nll

def negative_loglik(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False, numbafast=True):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)    
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    errscale = sigma_e
    errloc = mu_e          
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")    
    _nll = 0
    if debug:
        for i in range(K):
            for j in range(J):
                pij_arg = p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
                philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, variance=errscale)
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
    
    if numbafast:    
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")         
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        pij_arg = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K)     
    else:
        pij_arg = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)  
    
    philogcdf = norm.logcdf(pij_arg, loc=errloc, scale=errscale)
    log_one_minus_cdf = log_complement_from_log_cdf_vec(philogcdf, pij_arg, mean=errloc, variance=errscale)
    nll = np.sum(Y*philogcdf + (1-Y)*log_one_minus_cdf)

    if debug:
        assert(np.allclose(nll, _nll))

    # sum_Z_J_vectors = np.sum(Z, axis=1)    
    return -nll #+ penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)

def negative_loglik_coordwise_jax(theta_coordval, theta_idx, full_theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False):

    full_theta[theta_idx] = theta_coordval
    params_hat = optimisation_dict2params(full_theta, param_positions_dict, J, K, d, parameter_names)
    X = jnp.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    Z = jnp.asarray(params_hat["Z"]).reshape((d, J), order="F")   
    Y = jnp.asarray(Y)    
    if "Phi" in params_hat.keys():
        Phi = jnp.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
        delta = jnp.asarray(params_hat["delta"])
    else:
        Phi = jnp.zeros(Z.shape)
        delta = 0
    alpha = jnp.asarray(params_hat["alpha"])
    beta = jnp.asarray(params_hat["beta"])
    # c = params_hat["c"]
    gamma = jnp.asarray(params_hat["gamma"])    
    mu_e = 0
    sigma_e = jnp.asarray(params_hat["sigma_e"])
    errscale = sigma_e
    errloc = mu_e 
    _nll = 0
    dst_func = lambda x, y: jnp.sum((x-y)**2)

    pij_argJ = p_ij_arg(None, None, full_theta, J, K, d, parameter_names, dst_func, param_positions_dict, use_jax=True)  
    philogcdfJ = jax.scipy.stats.norm.logcdf(pij_argJ, loc=errloc, scale=errscale)
    # log_one_minus_cdfJ = log_complement_from_log_cdf_vec(philogcdfJ, pij_argJ, mean=errloc, variance=errscale, use_jax=True) - probably numerical errors vs iterative
    log_one_minus_cdfJ = jnp.zeros(philogcdfJ.shape)
    nlltest = 0
    for i in range(K):
        for j in range(J):
            if debug:
                pij_arg = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]    
                philogcdf = jax.scipy.stats.norm.logcdf(pij_arg, loc=errloc, scale=errscale)
            else:
                pij_arg = pij_argJ[i, j]
                philogcdf = philogcdfJ[i, j]
            
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, 
                                                            variance=errscale, use_jax=True)
            if debug:
                log_one_minus_cdfJ.at[i,j].set(log_one_minus_cdf[0])
                nlltest += (1-Y[i, j])*log_one_minus_cdfJ[i,j]
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
            else:
                nlltest += (1-Y[i, j])*log_one_minus_cdf
                
    nll = jnp.sum(Y*philogcdfJ) + nlltest
    if debug:
        assert(jnp.allclose(nll, _nll))
        
    # sum_Z_J_vectors = jnp.sum(Z, axis=1)
    return -nll[0] #+ jnp.asarray(penalty_weight_Z) * jnp.sum((sum_Z_J_vectors-jnp.asarray([constant_Z]*d))**2)    

def negative_loglik_jax(theta, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = jnp.asarray(params_hat["X"]).reshape((d, K), order="F")                     
    Z = jnp.asarray(params_hat["Z"]).reshape((d, J), order="F")   
    Y = jnp.asarray(Y)    
    if "Phi" in params_hat.keys():
        Phi = jnp.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
        delta = jnp.asarray(params_hat["delta"])
    else:
        Phi = jnp.zeros(Z.shape)
        delta = 0
    alpha = jnp.asarray(params_hat["alpha"])
    beta = jnp.asarray(params_hat["beta"])
    # c = params_hat["c"]
    gamma = jnp.asarray(params_hat["gamma"])    
    mu_e = 0
    sigma_e = jnp.asarray(params_hat["sigma_e"])
    errscale = sigma_e
    errloc = mu_e 
    _nll = 0
    dst_func = lambda x, y: jnp.sum((x-y)**2)

    pij_argJ = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict, use_jax=True)  
    philogcdfJ = jax.scipy.stats.norm.logcdf(pij_argJ, loc=errloc, scale=errscale)
    # log_one_minus_cdfJ = log_complement_from_log_cdf_vec(philogcdfJ, pij_argJ, mean=errloc, variance=errscale, use_jax=True) - probably numerical errors vs iterative
    log_one_minus_cdfJ = jnp.zeros(philogcdfJ.shape)
    nlltest = 0
    for i in range(K):
        for j in range(J):
            if debug:
                pij_arg = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]    
                philogcdf = jax.scipy.stats.norm.logcdf(pij_arg, loc=errloc, scale=errscale)
            else:
                pij_arg = pij_argJ[i, j]
                philogcdf = philogcdfJ[i, j]
            
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij_arg, mean=errloc, 
                                                            variance=errscale, use_jax=True)
            if debug:
                log_one_minus_cdfJ.at[i,j].set(log_one_minus_cdf[0])
                nlltest += (1-Y[i, j])*log_one_minus_cdfJ[i,j]
                _nll += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf
            else:
                nlltest += (1-Y[i, j])*log_one_minus_cdf
                
    nll = jnp.sum(Y*philogcdfJ) + nlltest
    if debug:
        assert(jnp.allclose(nll, _nll))
        
    sum_Z_J_vectors = jnp.sum(Z, axis=1)
    return -nll[0] + jnp.asarray(penalty_weight_Z) * jnp.sum((sum_Z_J_vectors-jnp.asarray([constant_Z]*d))**2)    

def collect_mle_results(efficiency_measures, data_topdir, M, K, J, sigma_e_true, d, parameter_names, param_positions_dict, batchsize, args, seedint=1234):
    
    wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = efficiency_measures
                
    parameter_space_dim = (K+J)*d + J + K + 2
    estimation_sq_error_per_trial_per_batch = dict()
    estimation_sq_error_per_trial_per_batch_nonRT = dict()
    estimation_error_per_trial_per_batch = dict()
    estimation_error_per_trial_per_batch_nonRT = dict()
    estimation_sq_error_per_trial = dict()
    estimation_sq_error_per_trial_nonRT = dict()
    estimation_error_per_trial = dict()
    estimation_error_per_trial_nonRT = dict()
    for param in parameter_names:
        estimation_sq_error_per_trial[param] = []
        estimation_sq_error_per_trial_nonRT[param] = []
        estimation_error_per_trial[param] = []
        estimation_error_per_trial_nonRT[param] = []
    
    for m in range(M):
        params_out_jsonl = dict()
        params_out_jsonl["param_positions_dict"] = param_positions_dict
        theta_true = np.zeros((parameter_space_dim,))
        with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_topdir, m), "r") as f:
            for result in f.iter(type=dict, skip_invalid=True):
                for param in parameter_names:
                    theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param]


        ############### !!!UPDATED THETA TRUE!!! #################
        # with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
        #     Y = pickle.load(f)
        # Y, K, J, theta_true, param_positions_dict, parameter_space_dim = clean_up_data_matrix(Y, K, J, d, 
        #                                                                                     theta_true, parameter_names, 
        #                                                                                     param_positions_dict)


        fig_sq_m_over_databatches = go.Figure()
        fig_sq_m_over_databatches_nonRT = go.Figure()
        fig_m_over_databatches = go.Figure()
        fig_m_over_databatches_nonRT = go.Figure()
        estimation_sq_error_per_trial_per_batch[m] = dict()
        estimation_sq_error_per_trial_per_batch_nonRT[m] = dict()
        estimation_error_per_trial_per_batch[m] = dict()
        estimation_error_per_trial_per_batch_nonRT[m] = dict()
        for param in parameter_names:            
            estimation_sq_error_per_trial_per_batch[m][param] = []
            estimation_sq_error_per_trial_per_batch_nonRT[m][param] = []
            estimation_error_per_trial_per_batch[m][param] = []
            estimation_error_per_trial_per_batch_nonRT[m][param] = []
                
        data_location = "{}/{}/{}/".format(data_topdir, m, batchsize)
        params_out, estimation_sq_error_per_trial_per_batch[m], estimation_sq_error_per_trial_per_batch_nonRT[m],\
            estimation_error_per_trial_per_batch[m], estimation_error_per_trial_per_batch_nonRT[m] = \
                    combine_estimate_variance_rule(data_location, J, K, d, parameter_names, 
                                                estimation_sq_error_per_trial_per_batch[m], 
                                                estimation_sq_error_per_trial_per_batch_nonRT[m],
                                                estimation_error_per_trial_per_batch[m], 
                                                estimation_error_per_trial_per_batch_nonRT[m],
                                                theta_true, param_positions_dict, seedint=seedint)    

        for param in parameter_names:
            if param == "X":                
                params_out_jsonl[param] = params_out[param].reshape((d*K,), order="F").tolist()     
                X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                X_hat = np.asarray(params_out_jsonl[param]).reshape((d, K), order="F")
                Rx, tx, mse_x, mse_x_nonRT, err_x, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)
                estimation_sq_error_per_trial[param].append(mse_x)
                estimation_sq_error_per_trial_nonRT[param].append(mse_x_nonRT)
                estimation_error_per_trial[param].append(err_x)
                estimation_error_per_trial_nonRT[param].append(err_x_nonRT)
                params_out_jsonl["mse_x_RT"] = mse_x
                params_out_jsonl["mse_x_nonRT"] = mse_x_nonRT
                params_out_jsonl["err_x_RT"] = err_x
                params_out_jsonl["err_x_nonRT"] = err_x_nonRT
            elif param == "Z":
                params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                Z_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                Rz, tz, mse_z, mse_z_nonRT, err_z, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)
                estimation_sq_error_per_trial[param].append(mse_z)
                estimation_sq_error_per_trial_nonRT[param].append(mse_z_nonRT)
                estimation_error_per_trial[param].append(err_z)
                estimation_error_per_trial_nonRT[param].append(err_z_nonRT)  
                params_out_jsonl["mse_z_RT"] = mse_z
                params_out_jsonl["mse_z_nonRT"] = mse_z_nonRT
                params_out_jsonl["err_z_RT"] = err_z
                params_out_jsonl["err_z_nonRT"] = err_z_nonRT               
            elif param == "Phi":            
                params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                Phi_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                Phi_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                Rphi, tphi, mse_phi, mse_phi_nonRT, err_phi, err_phi_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Phi_true, param_hat=Phi_hat, seedint=seedint)
                estimation_sq_error_per_trial[param].append(mse_phi)    
                estimation_sq_error_per_trial_nonRT[param].append(mse_phi_nonRT)
                estimation_error_per_trial[param].append(err_phi)
                estimation_error_per_trial_nonRT[param].append(err_phi_nonRT)        
                params_out_jsonl["mse_phi_RT"] = mse_phi
                params_out_jsonl["mse_phi_nonRT"] = mse_phi_nonRT
                params_out_jsonl["err_phi_RT"] = err_phi
                params_out_jsonl["err_phi_nonRT"] = err_phi_nonRT            
            elif param in ["beta", "alpha"]:
                params_out_jsonl[param] = params_out[param].tolist()
                rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]     
                mse = np.mean(rel_err**2)  
                estimation_sq_error_per_trial[param].append(mse)    
                estimation_sq_error_per_trial_nonRT[param].append(mse)
                estimation_error_per_trial[param].append(np.mean(rel_err))
                estimation_error_per_trial_nonRT[param].append(np.mean(rel_err))   
                params_out_jsonl["mse_{}".format(param)] = mse
                params_out_jsonl["rel_err_{}".format(param)] = np.mean(rel_err)
            else:
                params_out_jsonl[param] = params_out[param]
                rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                mse = rel_err**2
                estimation_sq_error_per_trial[param].append(mse)    
                estimation_sq_error_per_trial_nonRT[param].append(mse)
                estimation_error_per_trial[param].append(rel_err)
                estimation_error_per_trial_nonRT[param].append(rel_err)      
                params_out_jsonl["mse_{}".format(param)] = mse[0]
                params_out_jsonl["rel_err_{}".format(param)] = rel_err[0]
            
            fig_sq_m_over_databatches.add_trace(go.Box(
                    y=estimation_sq_error_per_trial_per_batch[m][param], 
                    x=[param]*len(estimation_sq_error_per_trial_per_batch[m][param]),
                    showlegend=False, boxpoints='outliers'))
            fig_sq_m_over_databatches_nonRT.add_trace(go.Box(
                    y=estimation_sq_error_per_trial_per_batch_nonRT[m][param], 
                    x=[param]*len(estimation_sq_error_per_trial_per_batch_nonRT[m][param]),
                    showlegend=False, boxpoints='outliers'))
            fig_m_over_databatches.add_trace(go.Box(
                    y=estimation_error_per_trial_per_batch[m][param], 
                    x=[param]*len(estimation_error_per_trial_per_batch[m][param]),
                    showlegend=False, boxpoints='outliers'))
            fig_m_over_databatches_nonRT.add_trace(go.Box(
                    y=estimation_error_per_trial_per_batch_nonRT[m][param], 
                    x=[param]*len(estimation_error_per_trial_per_batch_nonRT[m][param]),
                    showlegend=False, boxpoints='outliers'))
            
        pathlib.Path("{}/mle_estimation_plots/".format(data_topdir)).mkdir(parents=True, exist_ok=True)         
        savename = "{}/mle_estimation_plots/mse_trial{}_perparam_unweighted_boxplot.html".format(data_topdir, m)
        fix_plot_layout_and_save(fig_sq_m_over_databatches, savename, xaxis_title="Parameter", yaxis_title="Mean relative Sq. Err Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, print_pdf=False)
        savename = "{}/mle_estimation_plots/mse_nonRT_trial{}_perparam_unweighted_boxplot.html".format(data_topdir, m)
        fix_plot_layout_and_save(fig_sq_m_over_databatches_nonRT, savename, xaxis_title="Parameter", yaxis_title="Mean relative Sq. Err Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, print_pdf=False)
        savename = "{}/mle_estimation_plots/err_trial{}_perparam_unweighted_boxplot.html".format(data_topdir, m)
        fix_plot_layout_and_save(fig_m_over_databatches, savename, xaxis_title="Parameter", yaxis_title="Mean relative Err Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, print_pdf=False)
        savename = "{}/mle_estimation_plots/err_nonRT_trial{}_perparam_unweighted_boxplot.html".format(data_topdir, m)
        fix_plot_layout_and_save(fig_m_over_databatches_nonRT, savename, xaxis_title="Parameter", yaxis_title="Mean relative Err Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, print_pdf=False)
        
        theta_m = np.zeros((parameter_space_dim, ))
        for param in parameter_names:
            theta_m[param_positions_dict[param][0]:param_positions_dict[param][1]] = params_out_jsonl[param] 
        with open("{}/{}/Y.pickle".format(data_topdir, m), "rb") as f:
            Y = pickle.load(f)
        Y = Y.astype(np.int8).reshape((K, J), order="F")  
        loglik, _ = log_full_posterior(Y, theta_m, param_positions_dict, args)
        params_out_jsonl["loglik"] = loglik
        out_file = "{}/params_out_global_theta_hat.jsonl".format(data_location)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write(params_out_jsonl)
    
    # box plot - mse relativised per parameter over trials
    fig_sq_err = go.Figure()
    fig_sq_err_nonRT = go.Figure()
    fig_err = go.Figure()
    fig_err_nonRT = go.Figure()
    for param in parameter_names:
        # df = pd.DataFrame.from_dict({"vals":estimation_error_per_trial[param]})
        # lower_bound = df.vals.quantile(0.05)
        # upper_bound = df.vals.quantile(0.95)
        # df_filtered = df.vals[(df.vals >= lower_bound) & (df.vals <= upper_bound)].dropna()
        fig_sq_err.add_trace(go.Box(
                        y=np.asarray(estimation_sq_error_per_trial[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_sq_error_per_trial[param]), boxpoints='outliers'                                
                    ))
        fig_sq_err_nonRT.add_trace(go.Box(
                        y=np.asarray(estimation_sq_error_per_trial_nonRT[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_sq_error_per_trial_nonRT[param]), boxpoints='outliers'                                
                    ))
        fig_err.add_trace(go.Box(
                        y=np.asarray(estimation_error_per_trial[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_error_per_trial[param]), boxpoints='outliers'                                
                    ))
        fig_err_nonRT.add_trace(go.Box(
                        y=np.asarray(estimation_error_per_trial_nonRT[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_error_per_trial_nonRT[param]), boxpoints='outliers'                                
                    ))
    savename = "{}/mle_estimation_plots/mse_overAllTrials_perparam_weighted_boxplot.html".format(data_topdir)
    fix_plot_layout_and_save(fig_sq_err, savename, xaxis_title="", yaxis_title="Mean relative Sq. Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)
    savename = "{}/mle_estimation_plots/mse_nonRT_overAllTrials_perparam_weighted_boxplot.html".format(data_topdir)
    fix_plot_layout_and_save(fig_sq_err_nonRT, savename, xaxis_title="", yaxis_title="Mean relative Sq. Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)
    savename = "{}/mle_estimation_plots/err_overAllTrials_perparam_weighted_boxplot.html".format(data_topdir)
    fix_plot_layout_and_save(fig_err, savename, xaxis_title="", yaxis_title="Mean relative Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)
    savename = "{}/mle_estimation_plots/err_nonRT_overAllTrials_perparam_weighted_boxplot.html".format(data_topdir)
    fix_plot_layout_and_save(fig_err_nonRT, savename, xaxis_title="", yaxis_title="Mean relative Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)


def collect_mle_results_batchsize_analysis(data_topdir, batchsizes, M, K, J, sigma_e_true, d, parameter_names, param_positions_dict, seedint=1234):
    
    parameter_space_dim = (K+J)*d + J + K + 2    
    params_out_jsonl = dict()
    estimation_error_per_trial = dict()
    estimation_error_per_trial_nonrotated = dict()
    estimation_error_per_trial_per_batch = dict()
    estimation_error_per_trial_per_batch_nonrotated = dict()
    for param in parameter_names:
        estimation_error_per_trial[param] = dict()
        estimation_error_per_trial_nonrotated[param] = dict()
    for batchsize in batchsizes:                    
        print(batchsize)
        for m in range(M):   
            theta_true = np.zeros((parameter_space_dim,))
            with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_topdir, m), "r") as f:
                for result in f.iter(type=dict, skip_invalid=True):
                    for param in parameter_names:
                        theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
            fig_m_over_databatches = go.Figure()
            estimation_error_per_trial_per_batch[m] = dict()
            estimation_error_per_trial_per_batch[m][batchsize] = dict()
            estimation_error_per_trial_per_batch_nonrotated[m] = dict()
            estimation_error_per_trial_per_batch_nonrotated[m][batchsize] = dict()
            for param in parameter_names:            
                estimation_error_per_trial_per_batch[m][batchsize][param] = []       
                estimation_error_per_trial_per_batch_nonrotated[m][batchsize][param] = []                     
            data_location = "{}/{}/{}/".format(data_topdir, m, batchsize)
            params_out, estimation_error_per_trial_per_batch[m][batchsize], estimation_error_per_trial_per_batch_nonrotated[m][batchsize] = \
                                    combine_estimate_variance_rule(data_location, J, K, d, parameter_names, 
                                                                estimation_error_per_trial_per_batch[m][batchsize], 
                                                                estimation_error_per_trial_per_batch_nonrotated[m][batchsize], 
                                                                theta_true, param_positions_dict, seedint=seedint)    
            all_lows = []
            all_highs = []
            for param in parameter_names:
                print(param)
                if batchsize not in estimation_error_per_trial[param].keys():
                    estimation_error_per_trial[param][batchsize] = []
                    estimation_error_per_trial_nonrotated[param][batchsize] = []
                if param == "X":                
                    params_out_jsonl[param] = params_out[param].reshape((d*K,), order="F").tolist()     
                    X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                    X_hat = np.asarray(params_out_jsonl[param]).reshape((d, K), order="F")
                    Rx, tx, mse_x, mse_x_nonRT, err_x, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)
                    estimation_error_per_trial[param][batchsize].append(float(mse_x))
                    estimation_error_per_trial_nonrotated[param][batchsize].append(float(mse_x_nonRT))
                elif param == "Z":
                    params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                    Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                    Z_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                    Rz, tz, mse_z, mse_z_nonRT, err_z, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)
                    estimation_error_per_trial[param][batchsize].append(float(mse_z))  
                    estimation_error_per_trial_nonrotated[param][batchsize].append(float(mse_z_nonRT))            
                elif param == "Phi":            
                    params_out_jsonl[param] = params_out[param].reshape((d*J,), order="F").tolist()
                    Phi_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                    Phi_hat = np.asarray(params_out_jsonl[param]).reshape((d, J), order="F")
                    Rphi, tphi, mse_phi, mse_phi_nonRT, err_phi, err_phi_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Phi_true, param_hat=Phi_hat, seedint=seedint)
                    estimation_error_per_trial[param][batchsize].append(float(mse_phi))
                    estimation_error_per_trial[param][batchsize].append(float(mse_phi_nonRT))
                elif param in ["beta", "alpha"]:
                    params_out_jsonl[param] = params_out[param].tolist()     
                    mse = np.sum(((theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]])**2)/len(params_out_jsonl[param])
                    estimation_error_per_trial[param][batchsize].append(float(mse))
                else:
                    params_out_jsonl[param] = params_out[param]
                    mse = ((theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out_jsonl[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]])**2
                    estimation_error_per_trial[param][batchsize].append(float(mse[0])) 
                
                df = pd.DataFrame.from_dict({"vals": estimation_error_per_trial_per_batch[m][batchsize][param]})
                lower_bound = df.vals.quantile(0.05)
                upper_bound = df.vals.quantile(0.95)
                df_filtered = df.vals[(df.vals >= lower_bound) & (df.vals <= upper_bound)].dropna()
                q1 = df.vals.quantile(0.25)
                q3 = df.vals.quantile(0.75)
                iqr = q3-q1
                _lower_whisker = q1 - iqr
                _upper_whisker = q3 + iqr
                lower_whisker = min([x for x in estimation_error_per_trial_per_batch[m][batchsize][param] if x >= _lower_whisker])
                upper_whisker = max([x for x in estimation_error_per_trial_per_batch[m][batchsize][param] if x <= _upper_whisker])
                fig_m_over_databatches.add_trace(go.Box(
                    y=df_filtered, 
                    name=param,
                    # x=[param]*len(estimation_error_per_trial_per_batch[m][batchsize][param]),
                    showlegend=False,  
                    boxpoints='outliers'
                    ))
                all_lows.append(lower_whisker)
                all_highs.append(upper_whisker)
            fig_m_over_databatches.update_layout(yaxis_range=[min(all_lows), max(all_highs)])
            savename = "{}/mle_estimation_plots/mse_trial{}_perparam_unweighted_boxplot_batchsize{}.html".format(data_topdir, m, batchsize)
            pathlib.Path("{}/mle_estimation_plots/".format(data_topdir)).mkdir(parents=True, exist_ok=True)     
            fix_plot_layout_and_save(fig_m_over_databatches, savename, xaxis_title="Parameter", yaxis_title="MSE Θ", title="", 
                                    showgrid=False, showlegend=True, 
                                    print_png=True, print_html=True, print_pdf=False)
                                  
        out_file = "{}/params_out_global_theta_hat.jsonl".format(data_location)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write(params_out_jsonl)
    # box plot - mse relativised per parameter over trials    
    for param in parameter_names:
        fig = go.Figure()
        all_lows = []
        all_highs = []
        all_lows_nonrot = []
        all_highs_nonrot = []
        for batchsize in batchsizes:
            df = pd.DataFrame.from_dict({"vals":estimation_error_per_trial[param][batchsize]})
            lower_bound = df.vals.quantile(0.05)
            upper_bound = df.vals.quantile(0.95)
            df_filtered = df.vals[(df.vals >= lower_bound) & (df.vals <= upper_bound)].dropna()
            q1 = df.vals.quantile(0.25)
            q3 = df.vals.quantile(0.75)
            iqr = q3-q1
            _lower_whisker = q1 - iqr
            _upper_whisker = q3 + iqr
            lower_whisker = min([x for x in estimation_error_per_trial[param][batchsize] if x >= _lower_whisker])
            upper_whisker = max([x for x in estimation_error_per_trial[param][batchsize] if x <= _upper_whisker])
            all_lows.append(lower_whisker)
            all_highs.append(upper_whisker)
            fig.add_trace(go.Box(
                            y=df_filtered, showlegend=False, name=param,
                            boxpoints='outliers', x=[batchsize]*len(estimation_error_per_trial[param][batchsize])                                
                        ))
            if param in ["X", "Z", "Phi"]:
                df = pd.DataFrame.from_dict({"vals":estimation_error_per_trial_nonrotated[param][batchsize]})
                lower_bound = df.vals.quantile(0.05)
                upper_bound = df.vals.quantile(0.95)
                df_filtered = df.vals[(df.vals >= lower_bound) & (df.vals <= upper_bound)].dropna()
                q1 = df.vals.quantile(0.25)
                q3 = df.vals.quantile(0.75)
                iqr = q3-q1
                _lower_whisker = q1 - iqr
                _upper_whisker = q3 + iqr
                lower_whisker = min([x for x in estimation_error_per_trial_nonrotated[param][batchsize] if x >= _lower_whisker])
                upper_whisker = max([x for x in estimation_error_per_trial_nonrotated[param][batchsize] if x <= _upper_whisker])
                all_lows_nonrot.append(lower_whisker)
                all_highs_nonrot.append(upper_whisker)
                fig.add_trace(go.Box(
                            y=np.asarray(estimation_error_per_trial_nonrotated[param][batchsize]).tolist(), showlegend=True, name="{} - nonRT".format(param),
                            boxpoints='outliers', x=[batchsize]*len(estimation_error_per_trial_nonrotated[param][batchsize])                                
                        ))
        fig.update_layout(boxmode="group")
        all_lows.extend(all_lows_nonrot)
        all_highs.extend(all_highs_nonrot)
        fig.update_layout(yaxis_range=[min(all_lows), max(all_highs)])
        savename = "{}/mle_estimation_plots/mse_overAllTrials_{}_weighted_boxplot.html".format(data_topdir, param)
        pathlib.Path("{}/mle_estimation_plots/".format(data_topdir)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title="MSE Θ", title="", 
                                showgrid=False, showlegend=True, 
                                print_png=True, print_html=True, 
                                print_pdf=False)

####################### MLE #############################



####################### ICM #############################

def create_constraint_functions_icm(n, vector_coordinate=None, param=None, param_positions_dict=None, args=None):
    
    bounds = []
    grid_width_std = 5
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    if param == "alpha":
        bounds.append((-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha))
    elif param == "beta":
        bounds.append((-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta))
    elif param == "gamma":            
        bounds.append((None, None))  ########## MODIFY TO SET NON_ZERO CONSTRAINT?
    elif param == "delta":                    
        bounds.append((None, None))  ########## MODIFY TO SET NON_ZERO CONSTRAINT?        
    elif param == "sigma_e":
        bounds.append((min_sigma_e, grid_width_std*np.sqrt(prior_scale_sigmae)))
    else:
        if d == 1 or elementwise:
            if param == "Phi":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate]))
            elif param == "Z":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate]))
            elif param == "X":
                bounds.append((-grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate]))        
        else:
            raise NotImplementedError("At the moment, we have not implemented diffentiation for multivariate parameter vector.")
    
    return bounds


def log_complement_from_log_cdf_vec(log_cdfx, x, mean, variance, use_jax=False):
     """
     Computes log(1-CDF(x)) given log(CDF(x)) in a numerically stable way.
     """    
     if use_jax:
        if len(log_cdfx.shape)==0 or (isinstance(log_cdfx, jnp.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1): 
            if log_cdfx < -0.693:  #log(0.5)
                # CDF(x) < 0.5, direct computation is stable  
                xinput = -jnp.exp(log_cdfx)
                clipped_xinput = jnp.clip(xinput, a_min=-0.9999999999, a_max=None)
                ret = jnp.log1p(clipped_xinput)
            else: 
                # CDF(x) >= 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
                ret = jax.scipy.stats.norm.logcdf(-x, loc=mean, scale=variance)       
        else:
            ret = jnp.zeros(log_cdfx.shape)    
            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case1 = jnp.argwhere(log_cdfx < -0.693)
                if idx_case1.size > 0:
                    xinput = -jnp.exp(log_cdfx[log_cdfx < -0.693])
                    clipped_xinput = jnp.clip(xinput, a_min=-0.9999999999, a_max=None)
                    ret.at[log_cdfx < -0.693].set(jnp.log1p(clipped_xinput))                  
            else:
                idx_case1 = jnp.argwhere(log_cdfx < -0.693).flatten()       
                if idx_case1.size > 0:
                    xinput = -np.exp(log_cdfx[idx_case1])
                    clipped_xinput = jnp.clip(xinput, a_min=-0.9999999999, a_max=None)
                    ret.at[idx_case1].set(jnp.log1p(clipped_xinput))          

            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case2 = jnp.argwhere(log_cdfx >= -0.693)
                if idx_case2.size > 0:
                    ret.at[log_cdfx >= -0.693].set(jax.scipy.stats.norm.logcdf(-x[log_cdfx >= -0.693], loc=mean, scale=variance))
            else:
                idx_case2 = jnp.argwhere(log_cdfx >= -0.693).flatten()    
                if idx_case2.size > 0:                
                    ret.at[idx_case2].set(jax.scipy.stats.norm.logcdf(-x[idx_case2], loc=mean, scale=variance))       
     else:
        if isinstance(log_cdfx, float) or (isinstance(log_cdfx, np.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1): 
            if log_cdfx < -0.693:  #log(0.5)
                # If CDF(x) < 0.5, direct computation is stable  
                xinput = -np.exp(log_cdfx)
                clipped_xinput = np.clip(xinput, a_min=-0.9999999999, a_max=None)
                ret = np.log1p(clipped_xinput)
            else: 
                # If CDF(x) ≥ 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
                ret = norm.logcdf(-x, loc=mean, scale=variance)       
        else:                          
            ret = np.zeros(log_cdfx.shape)                
            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case1 = np.argwhere(log_cdfx < -0.693)
                if idx_case1.size > 0:
                    xinput = -np.exp(log_cdfx[log_cdfx < -0.693])
                    clipped_xinput = np.clip(xinput, a_min=-0.9999999999, a_max=None)
                    ret[log_cdfx < -0.693] = np.log1p(clipped_xinput)                     
            else:
                idx_case1 = np.argwhere(log_cdfx < -0.693).flatten()       
                if idx_case1.size > 0:
                    xinput = -np.exp(log_cdfx[idx_case1])
                    clipped_xinput = np.clip(xinput, a_min=-0.9999999999, a_max=None)
                    ret[idx_case1] = np.log1p(clipped_xinput)             
 
            if ret.shape[0] > 1 and len(ret.shape)==2 and ret.shape[1] > 1:
                idx_case2 = np.argwhere(log_cdfx >= -0.693)
                if idx_case2.size > 0:
                    ret[log_cdfx >= -0.693] = norm.logcdf(-x[log_cdfx >= -0.693], loc=mean, scale=variance)
            else:
                idx_case2 = np.argwhere(log_cdfx >= -0.693).flatten()    
                if idx_case2.size > 0:                
                    ret[idx_case2] = norm.logcdf(-x[idx_case2], loc=mean, scale=variance)   
 
     return ret

def log_complement_from_log_cdf_vec_fast(log_cdfx, x, mean, variance, use_jax=False, approxfast=False, threshold_std_no=5):
    
    if use_jax:
        return log_complement_from_log_cdf_vec(log_cdfx, x, mean, variance, use_jax=True)

    if log_cdfx.ndim == 0 or (log_cdfx.size == 1):
        if log_cdfx < -0.693:
            return np.log1p(-np.exp(log_cdfx))
        else:
            return norm.logcdf(-x, loc=mean, scale=variance)
    
    ret = np.zeros_like(log_cdfx, dtype=float)
    case1_mask = log_cdfx < -0.693
    case2_mask = ~case1_mask    
    # Case 1: CDF(x) < 0.5
    if np.any(case1_mask):
        ret[case1_mask] = np.log1p(-np.exp(log_cdfx[case1_mask]))    
    # Case 2: CDF(x) >= 0.5
    if np.any(case2_mask):
        adjusted_x = -x[case2_mask]
        ret[case2_mask] = norm.logcdf(adjusted_x, loc=mean, scale=variance)
    
    return ret


def log_complement_from_log_cdf(log_cdfx, x, mean, variance, use_jax=False):
    """
    Computes log(1-CDF(x)) given log(CDF(x)) in a numerically stable way.
    """    
    def get_one_minus_logcdf(logcdfxx):
        logcdfx, xx = logcdfxx
        if logcdfx < -0.693:  #log(0.5)
            # If CDF(x) < 0.5, direct computation is stable  
            if use_jax:
                ret = jnp.log1p(-jnp.exp(logcdfx))
                # ret = jnp.asarray(list(ret))
            else:                
                ret = np.log1p(-np.exp(logcdfx))
                # ret = np.asarray(list(ret))                
        else: 
            # If CDF(x) ≥ 0.5, use the fact that 1-CDF(x) = CDF(-x), hence log(1-CDF(x)) = log(CDF(-x))   
            if use_jax:
                ret = jax.scipy.stats.norm.logcdf(-xx, loc=mean, scale=variance)
            else:
                ret = norm.logcdf(-xx, loc=mean, scale=variance)             
        
        return ret

    if use_jax:
        if len(log_cdfx.shape)==0 or (isinstance(log_cdfx, jnp.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1):        
            return get_one_minus_logcdf((log_cdfx, x))
        else:                
            retvallist = list(map(get_one_minus_logcdf, zip(log_cdfx, x)))        
            return jnp.array(retvallist)
    else:
        if isinstance(log_cdfx, float) or (isinstance(log_cdfx, np.ndarray) and len(log_cdfx.shape)==1 and log_cdfx.shape[0]==1):        
            return get_one_minus_logcdf((log_cdfx, x))
        else:                
            retvallist = list(map(get_one_minus_logcdf, zip(log_cdfx, x)))        
            return np.array(retvallist)
 
@numba.jit(nopython=True, parallel=True, cache=True)
def p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K):
    
    phi = np.zeros((K, Z.shape[1]), dtype=np.float64)    
    for i in prange(K):
        xi = X[:, i]
        phi[i, :] = p_i_arg_numbafast(xi, Z, alpha, beta[i], gamma)
                 
    return phi

@numba.jit(nopython=True, cache=True)
def p_i_arg_numbafast(xi, Z, alpha, betai, gamma):
    
    x_broadcast = xi[:, np.newaxis]    
    diff_xz = x_broadcast - Z        
    dst_xz = np.sum(diff_xz * diff_xz, axis=0)             
    phi = gamma*dst_xz + alpha + betai                   
    
    return phi

@numba.jit(nopython=True, cache=True)
def p_j_arg_numbafast(X, zj, alphaj, beta, gamma):
    
    z_broadcast = zj[:, np.newaxis]    
    diff_xz = z_broadcast - X
    dst_xz = np.sum(diff_xz * diff_xz, axis=0)             
    phi = gamma*dst_xz + alphaj + beta                   
    
    return phi


def p_ij_arg(i, j, theta, J, K, d, parameter_names, dst_func, param_positions_dict, use_jax=False):

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    if use_jax:
        X = jnp.asarray(params_hat["X"]).reshape((d, K), order="F")                     
        Z = jnp.asarray(params_hat["Z"]).reshape((d, J), order="F")      
        if "Phi" in params_hat.keys(): 
            Phi = jnp.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
            delta = jnp.asarray(params_hat["delta"])
        else:
            Phi = jnp.zeros(Z.shape)
            delta = 0
        alpha = jnp.asarray(params_hat["alpha"])
        beta = jnp.asarray(params_hat["beta"])
        gamma = jnp.asarray(params_hat["gamma"])        
        if isinstance(i, int) and isinstance(j, int):            
            phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]            
        else:
            def pairwise_dst_fast(xi_betai):
                xi, betai = xi_betai            
                x_broadcast = xi[:, jnp.newaxis]
                diff_xz = x_broadcast - Z
                dst_xz = jnp.sum(diff_xz * diff_xz, axis=0)
                if "Phi" in params_hat.keys():
                    diff_xphi = x_broadcast - Phi
                    dst_xphi = jnp.sum(diff_xphi * diff_xphi, axis=1)
                else:
                    dst_xphi = 0
                phi = gamma*dst_xz - delta*dst_xphi + alpha + betai     
                return phi
            def pairwise_dst_fast_j(zj_alphaj):                                
                if "Phi" in params_hat.keys():
                    zj, phi_j, alphaj = zj_alphaj  
                else:
                    zj, alphaj = zj_alphaj                            
                z_broadcast = zj[:, jnp.newaxis]
                diff_xz = z_broadcast - X
                dst_xz = jnp.sum(diff_xz * diff_xz, axis=0)
                if "Phi" in params_hat.keys():
                    phi_broadcast = phi_j[:, jnp.newaxis]
                    diff_xphi = phi_broadcast - X
                    dst_xphi = jnp.sum(diff_xphi * diff_xphi, axis=1)
                else:
                    dst_xphi = 0                
                phi = gamma*dst_xz - delta*dst_xphi + alphaj + beta
                return phi
            if isinstance(i, int) and j is None:        
                phi = pairwise_dst_fast((X[:, i], beta[i]))       
            elif i is None and isinstance(j, int):      
                if "Phi" in params_hat.keys():       
                    phi = pairwise_dst_fast_j((Z[:, j], Phi[:, j], alpha[j]))                
                else:
                    phi = pairwise_dst_fast_j((Z[:, j], alpha[j]))             
            elif i is None and j is None:                          
                arr_list = list(map(pairwise_dst_fast, zip(X.transpose(), beta)))
                phi = jnp.vstack(arr_list)
    else:
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                     
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")      
        if "Phi" in params_hat.keys(): 
            Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")     
            delta = params_hat["delta"]
        else:
            Phi = np.zeros(Z.shape)
            delta = 0
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        gamma = params_hat["gamma"]  
        if isinstance(i, int) and isinstance(j, int):
            phi = gamma*dst_func(X[:, i], Z[:, j]) - delta*dst_func(X[:, i], Phi[:, j]) + alpha[j] + beta[i]
        else:
            def pairwise_dst_fast(xi_betai):                
                xi, betai = xi_betai  
                x_broadcast = xi[:, np.newaxis]
                diff_xz = x_broadcast - Z
                dst_xz = np.sum(diff_xz * diff_xz, axis=0)
                if "Phi" in params_hat.keys():
                    diff_xphi = x_broadcast - Phi
                    dst_xphi = np.sum(diff_xphi * diff_xphi, axis=1)
                else:
                    dst_xphi = 0
                phi = gamma*dst_xz - delta*dst_xphi + alpha + betai     
                return phi
            def pairwise_dst_fast_j(zj_alphaj):                                
                if "Phi" in params_hat.keys():
                    zj, phi_j, alphaj = zj_alphaj  
                else:
                    zj, alphaj = zj_alphaj                            
                z_broadcast = zj[:, np.newaxis]
                diff_xz = z_broadcast - X
                dst_xz = np.sum(diff_xz * diff_xz, axis=0)
                if "Phi" in params_hat.keys():
                    phi_broadcast = phi_j[:, np.newaxis]
                    diff_xphi = phi_broadcast - X
                    dst_xphi = np.sum(diff_xphi * diff_xphi, axis=1)
                else:
                    dst_xphi = 0                
                phi = gamma*dst_xz - delta*dst_xphi + alphaj + beta
                return phi
            if isinstance(i, int) and j is None:        
                phi = pairwise_dst_fast((X[:, i], beta[i]))            
            elif i is None and isinstance(j, int):      
                if "Phi" in params_hat.keys():       
                    phi = pairwise_dst_fast_j((Z[:, j], Phi[:, j], alpha[j]))                
                else:
                    phi = pairwise_dst_fast_j((Z[:, j], alpha[j]))                
            elif i is None and j is None:                          
                arr_list = list(map(pairwise_dst_fast, zip(X.transpose(), beta)))
                phi = np.vstack(arr_list)
    
    return phi


def halve_annealing_rate_upd_schedule(N, gamma, delta_n, temperature_rate, temperature_steps, 
                                    all_gammas=None, testparam=None):
       
    tr_idx = temperature_rate.index(delta_n)            
    temperature_rate_upd = temperature_rate[:tr_idx]
    temperature_rate_upd.append(temperature_rate[tr_idx]/2)
    temperature_rate_upd.extend(temperature_rate[(tr_idx+1):])        
    all_gammas = []             
    gamma = 0.01        
    for gidx in range(len(temperature_steps[1:])):
        upperlim = temperature_steps[1+gidx]        
        start = gamma if gidx==0 else all_gammas[-1]        
        all_gammas.extend(np.arange(start, upperlim, temperature_rate_upd[gidx]))            
    N = len(all_gammas)
    print("Annealing schedule: {}".format(N))                
    gamma = all_gammas[0] 
    delta_n = temperature_rate_upd[0] 

    if testparam is not None:
        all_gammas = [1]*N
        gamma = 1

    return gamma, delta_n, temperature_rate_upd, all_gammas, N


def check_convergence(elementwise, theta_curr, theta_prev, param_positions_dict, iteration, parameter_space_dim, d=None, testparam=None, testidx=None, p=0.2, tol=1e-6):
    
    converged = False
    if testparam is None:
        # check full vector for convergence
        delta_theta_se = (theta_curr - theta_prev)**2
        if not np.allclose(np.sum(delta_theta_se), 1e-14):                                   
            delta_theta = delta_theta_se/np.sum(delta_theta_se)
        else:
            delta_theta = delta_theta_se
    else:
        if not elementwise and testidx is not None:
            # check vector portion
            delta_theta_se = (theta_curr[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d] - theta_prev[param_positions_dict[testparam][0]+testidx*d:param_positions_dict[testparam][0]+(testidx+1)*d])**2
            if not np.allclose(np.sum(delta_theta_se), 1e-14):                                   
                delta_theta = delta_theta_se/np.sum(delta_theta_se)
            else:
                delta_theta = delta_theta_se
        elif elementwise:
            delta_theta = np.abs(theta_curr[param_positions_dict[testparam][0]+testidx] - theta_prev[param_positions_dict[testparam][0]+testidx])
    if np.all(delta_theta <= tol):
        converged = True

    random_restart = False
    if testparam is None:
        # check all vector, after a full iteration - add check for iteration to make sure
        if iteration >= parameter_space_dim and (np.sum(delta_theta < tol) > int(p*len(delta_theta))):        
            random_restart = True
    else:                
        # check tested param only, ensure all coordinates have been updated once - delta_theta is already the sub-vector
        if (not elementwise) and (testidx is not None) and (iteration >= param_positions_dict[testparam][1]):
            if (np.sum(delta_theta < tol) > int(p*len(delta_theta))):        
                random_restart = True
        else:
            # no point to check when looking at specific coordinate
            random_restart = False
        
    return converged, delta_theta, random_restart


def parse_timedelta_string(time_str):

    if "day" in time_str:
        day_part, time_part = time_str.split(", ")
        days = int(day_part.split()[0])
    else:
        days = 0
        time_part = time_str

    hours, minutes, seconds = time_part.split(":")
    seconds, microseconds = (seconds.split(".") + ['0'])[:2]
    tdelta = timedelta(
        days=days,
        hours=int(hours),
        minutes=int(minutes),
        seconds=int(seconds),
        microseconds=int(microseconds)
    )

    return tdelta, int(hours), int(minutes), int(seconds), int(microseconds)



def rank_and_plot_solutions(estimated_thetas, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, dst_func, 
                            param_positions_dict, DIR_out, args, data_tempering=False, row_start=None, row_end=None, seedint=1234, get_RT_error=False):

    if efficiency_measures is not None:
        wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = efficiency_measures
    else:
        wall_duration = None 
        avg_total_cpu_util = None
        max_total_cpu_util = None
        avg_total_ram_residentsetsize_MB = None
        max_total_ram_residentsetsize_MB = None
        avg_threads = None
        max_threads = None
        avg_processes = None
        max_processes = None

    best_theta = None
    computed_logfullposterior = []
    for theta_set in estimated_thetas:
        theta = theta_set[0]
        _, posterior = log_full_posterior(Y, theta.copy(), param_positions_dict, args)
        computed_logfullposterior.append(posterior[0])
    # sort in increasing order, i.e. from worst to best solution
    sorted_idx = np.argsort(np.asarray(computed_logfullposterior))
    sorted_idx_lst = sorted_idx.tolist()  
    # save in the order of best to worst solution
    best2worst = list(reversed(sorted_idx_lst))    
    theta_list = []
    posterior_list = []
    for i in best2worst:
        theta = estimated_thetas[i][0]
        theta_list.append(theta)
        best_theta = theta.copy()
        if best2worst.index(i) == 0 and get_RT_error:
            theta_true = args[-1]
            # only for best solution compute expensive error:
            params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
            X_true = np.asarray(params_true["X"]) # d x K    
            X_hat = np.asarray(params_hat["X"]) # d x K         
            Z_true = np.asarray(params_true["Z"]) # d x J       
            Z_hat = np.asarray(params_hat["Z"]) # d x J                           
            params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
            Rx, tx, mse_x_RT, mse_x_nonRT, err_x_RT, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)
            Rz, tz, mse_z_RT, mse_z_nonRT, err_z_RT, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)
        else:
            mse_x_RT = estimated_thetas[i][1]
            mse_z_RT = estimated_thetas[i][2]
            mse_x_nonRT = estimated_thetas[i][3]
            mse_z_nonRT = estimated_thetas[i][4]
            err_x_RT = estimated_thetas[i][5]
            err_z_RT = estimated_thetas[i][6]
            err_x_nonRT = estimated_thetas[i][7]
            err_z_nonRT = estimated_thetas[i][8]
        logposterior = computed_logfullposterior[i]
        posterior_list.append(logposterior)
        params_out = dict()
        params_out["logfullposterior"] = logposterior
        params_out["mse_x_RT"] = mse_x_RT
        params_out["mse_z_RT"] = mse_z_RT
        params_out["mse_x_nonRT"] = mse_x_nonRT
        params_out["mse_z_nonRT"] = mse_z_nonRT
        params_out["err_x_RT"] = err_x_RT
        params_out["err_z_RT"] = err_z_RT
        params_out["err_x_nonRT"] = err_x_nonRT
        params_out["err_z_nonRT"] = err_z_nonRT
        params_out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")        
        params_out["elapsedtime"] = elapsedtime        
        time_obj, hours, mins, sec, microsec = parse_timedelta_string(params_out["elapsedtime"])
        params_out["elapsedtime_hours"] = hours
        params_out["param_positions_dict"] = param_positions_dict
        for param in parameter_names:
            params_out[param] = theta[param_positions_dict[param][0]:param_positions_dict[param][1]]
            if isinstance(params_out[param], np.ndarray):
                params_out[param] = params_out[param].tolist()
        if data_tempering:
            out_file = "{}/params_out_local_theta_hat_{}_{}.jsonl".format(DIR_out, row_start, row_end)
            with open(out_file, 'a') as f:         
                writer = jsonlines.Writer(f)
                writer.write(params_out)
        else:
            out_file = "{}/params_out_global_theta_hat.jsonl".format(DIR_out)
            with open(out_file, 'a') as f:         
                writer = jsonlines.Writer(f)
                writer.write(params_out)

        # plot utilities
        pathlib.Path("{}/solution_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)   
        pij_arg = p_ij_arg(None, None, theta, J, K, d, parameter_names, dst_func, param_positions_dict)
        fig = go.Figure(data=go.Heatmap(
            z=pij_arg,
            x=[str(i) for i in range(pij_arg.shape[1])],
            y=[str(i) for i in range(pij_arg.shape[0])],
            colorscale="sunsetdark",
            showscale=True,   
            colorbar=dict(thickness=10, title='U'),             
        )) 
        fix_plot_layout_and_save(fig, "{}/solution_plots/utilities_solution_index_{}.html".format(DIR_out, sorted_idx_lst.index(i)), 
                                xaxis_title="Leaders", yaxis_title="Followers", title="Utilities with estimated parameters", 
                                showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
      
    if not (efficiency_measures is None):
        out_file = "{}/efficiency_metrics.jsonl".format(DIR_out)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write({"wall_duration": wall_duration, 
                        "avg_total_cpu_util": avg_total_cpu_util, 
                        "max_total_cpu_util": max_total_cpu_util, 
                        "avg_total_ram_residentsetsize_MB": avg_total_ram_residentsetsize_MB, 
                        "max_total_ram_residentsetsize_MB": max_total_ram_residentsetsize_MB,
                        "avg_threads": avg_threads, 
                        "max_threads": max_threads, 
                        "avg_processes": avg_processes, 
                        "max_processes": max_processes})

    # 2D projection of solutions
    # raw_symbols = SymbolValidator().values    
    theta_matrix = np.asarray(theta_list)
    computed_logfullposterior = np.array(posterior_list)    
    if theta_matrix.shape[0] > 1:
        # theta_matrix = theta_matrix[sorted_idx, :]
        pca = PCA(n_components=2)
        components = pca.fit_transform(theta_matrix)
        fig = go.Figure()
        for i in range(components.shape[0]):
            opacity = (components.shape[0]-i)/(components.shape[0])
            fig.add_trace(go.Scatter(x=[components[i, 0]], y=[components[i, 1]], marker_color="green",
                                    marker_symbol="circle", name="Rank {}".format(i+1), marker_opacity=opacity,
                                    text="pc2/(pc1+pc2) = {:.2f}/({:.2f}+{:.2f}) = {:.2f}"                                                      
                                    "<br>full posterior = {:.3f}".format(components[i, 1], components[i, 0], components[i, 1],
                                                                components[i, 1]/(components[i, 0] + components[i, 1]),
                                                                computed_logfullposterior[i])))
        fig.update(layout_yaxis_range = [np.min(components[:,1])-1,np.max(components[:,1])+1])
        fix_plot_layout_and_save(fig, "{}/solution_plots/project_solutions_2D.html".format(DIR_out, 
                                sorted_idx_lst.index(i)), xaxis_title="PC1", yaxis_title="PC2", 
                                title="s1 = {:.3f}, s2 = {:.3f}".format(pca.singular_values_[0], pca.singular_values_[1]), 
                                showgrid=False, showlegend=True, print_png=True, print_html=True, print_pdf=False)
        # fig.show()
    
    return best_theta
 

def sample_theta_curr_init(parameter_space_dim, base2exponent, param_positions_dict, args, samples_list=None, idx_all=None, rng=None):

    try:
        DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
            _, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
            gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    except:
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method,\
            parameter_names, J, K, d, N, dst_func, niter, _, m, penalty_weight_Z,\
                constant_Z, retries, parallel, min_sigma_e, prior_loc_x, prior_scale_x, prior_loc_z,\
                    prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha,\
                        prior_scale_alpha, prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta,\
                            prior_loc_sigmae, prior_scale_sigmae, _, rng, batchsize, theta_true = args

    if samples_list is None and parameter_space_dim <= 21201:
        sampler = qmc.Sobol(d=parameter_space_dim, scramble=False)   
        samples_list = list(sampler.random_base2(m=base2exponent))
        idx_all = np.arange(0, len(samples_list), 1).tolist()       
    elif samples_list is None and parameter_space_dim > 21201:
        if d > 2:
            raise NotImplementedError("In {}-dimensional space for the ideal points, find a way to generate random initial solutions.")
        sampler = qmc.Sobol(d=21201, scramble=False)    
        base2exponent = 15
        samples = sampler.random_base2(m=base2exponent)
        samplesreshape = samples[1:,:].reshape((-1,1))
        samplesselect = rng.choice(samplesreshape, 15*parameter_space_dim, replace=False)
        samples = samplesselect.reshape((15, parameter_space_dim))
        samples_list = list(samples)       
        idx_all = np.arange(0, len(samples_list), 1).tolist()
    
    idx = np.random.choice(idx_all, size=1, replace=False)[0]
    theta_curr = np.asarray(samples_list[idx]).reshape((1, parameter_space_dim))
    idx_all.remove(idx)       

    lbounds = np.zeros((parameter_space_dim,))
    ubounds = np.ones((parameter_space_dim,))
    for param in parameter_names:
        if param == "X":
            # assume homogeneous variance
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_x[0, 0])+prior_loc_x[0]
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_x[0, 0])+prior_loc_x[0]
        elif param == "Z":
            # assume homogeneous variance
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_z[0, 0])+prior_loc_z[0]
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_z[0, 0])+prior_loc_z[0]
        elif param == "Phi":
            # assume homogeneous variance
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_phi[0, 0])+prior_loc_phi[0]
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_phi[0, 0])+prior_loc_phi[0]
        elif param == "alpha":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_alpha)+prior_loc_alpha
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_alpha)+prior_loc_alpha
        elif param == "beta":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_beta)+prior_loc_beta
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_beta)+prior_loc_beta
        elif param == "gamma":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_gamma)+prior_loc_gamma
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_gamma)+prior_loc_gamma
        elif param == "delta":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = -5*np.sqrt(prior_scale_delta)+prior_loc_delta
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_delta)+prior_loc_delta        
        elif param == "sigma_e":
            lbounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = min_sigma_e
            ubounds[param_positions_dict[param][0]:param_positions_dict[param][1]] = 5*np.sqrt(prior_scale_sigmae) #+prior_loc_gamma
    
    theta_curr = qmc.scale(theta_curr, lbounds, ubounds).reshape((parameter_space_dim,))

    return theta_curr, samples_list, idx_all


def data_annealing_init_theta_given_theta_prev(theta_curr, theta_prev, K, J, d, param_positions_dict, parameter_names, annealing_prev, diff_elementwise=False):
 
    for param in parameter_names:
        if (not diff_elementwise) and param in ["X", "beta"]:
            print("not init from prev")
            continue
        elif diff_elementwise and param in ["X", "beta"]:
            # if param=="X":
            theta_curr[annealing_prev[param][0]:annealing_prev[param][1]] = theta_prev[annealing_prev[param][0]:annealing_prev[param][1]].copy()
            print(param, theta_prev[annealing_prev[param][0]:annealing_prev[param][1]], np.allclose(theta_prev[annealing_prev[param][0]:annealing_prev[param][1]], theta_curr[annealing_prev[param][0]:annealing_prev[param][1]]))
        else:
            theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]] = theta_prev[annealing_prev[param][0]:annealing_prev[param][1]].copy()
            print(param, theta_prev[annealing_prev[param][0]:annealing_prev[param][1]], np.allclose(theta_prev[annealing_prev[param][0]:annealing_prev[param][1]], theta_curr[param_positions_dict[param][0]:param_positions_dict[param][1]]))
        
    return theta_curr


def update_annealing_temperature(gamma_prev, total_iter, temperature_rate, temperature_steps, all_gammas=None):
    
    delta_n = None
    if all_gammas is None:        
        if (gamma_prev >= temperature_steps[0] and gamma_prev <= temperature_steps[1]):
            delta_n = temperature_rate[0]
        elif (gamma_prev > temperature_steps[1] and gamma_prev <= temperature_steps[2]):
            delta_n = temperature_rate[1]
        elif (gamma_prev > temperature_steps[2] and gamma_prev <= temperature_steps[3]):
            delta_n = temperature_rate[2]
        elif (gamma_prev > temperature_steps[3]):
            delta_n = temperature_rate[3]        
        gamma = gamma_prev + delta_n        
        # print("Delta_{} = {}".format(total_iter, delta_n))            
    elif isinstance(all_gammas, list):  
        idx = total_iter % len(all_gammas)
        gamma = all_gammas[idx]                                    
        if (gamma >= temperature_steps[0] and gamma <= temperature_steps[1]):
            tr_idx = 0            
        elif (gamma > temperature_steps[1] and gamma <= temperature_steps[2]):
            tr_idx = 1            
        elif (gamma > temperature_steps[2] and gamma <= temperature_steps[3]):
            tr_idx = 2            
        elif (gamma > temperature_steps[3]):
            tr_idx = 3     
        delta_n = temperature_rate[tr_idx]             

    return gamma, delta_n


def get_min_achievable_mse_under_rotation_trnsl(param_true, param_hat, seedint):

    """
    Minimizes ||Y - (XR + t)||_F where ||.||_F is the Frobenius norm and R a rotation matrix
    
    """
    if param_true.shape != param_hat.shape:
        raise ValueError("Input matrices X and Y must have the same shape")
    
    X_mean = np.mean(param_hat, axis=0)
    Y_mean = np.mean(param_true, axis=0)
    
    X_centered = param_hat - X_mean
    Y_centered = param_true - Y_mean
    
    # Compute the covariance
    H = X_centered.T @ Y_centered
    
    # SVD
    try:
        U, S, Vt = np.linalg.svd(H)
    except:
        svd_out = ca_svd.compute_svd(
            X=H,
            n_iter=2,
            n_components=2,
            random_state=seedint,
            engine='sklearn',
        )
        U, S, Vt = svd_out.U, svd_out.s, svd_out.V

    # Construct the rotation matrix
    # Handle reflection by ensuring proper rotation, i.e. det(R) = 1
    d = np.linalg.det(Vt.T @ U.T)
    M = np.eye(U.shape[1])
    M[-1, -1] = d
    R = Vt.T @ M @ U.T
    
    # Step 5: Compute the optimal translation
    t = Y_mean - X_mean @ R
    
    # Compute the residual error
    # error = np.linalg.norm(param_true - (param_hat @ R + t), 'fro')
    # error = np.sum((param_true - (param_hat @ R + t))**2)/(param_true.shape[0]*param_true.shape[1])
    # relative error
    reconstructed = param_hat @ R + t
    rec = reconstructed.reshape((param_true.shape[0]*param_true.shape[1],), order="F")
    true_reshaped = param_true.reshape((param_true.shape[0]*param_true.shape[1],), order="F")
    rel_err = (true_reshaped - rec)/true_reshaped
    sq_err = rel_err**2
    meanrelerror = np.mean(rel_err)
    meansquarederror = np.mean(sq_err)
    # non-rotated/non-translated relative error
    hat_reshaped = param_hat.reshape((param_hat.shape[0]*param_hat.shape[1],), order="F")
    rel_err_nonRT = (true_reshaped - hat_reshaped)/true_reshaped
    sq_err_nonRT = rel_err_nonRT**2
    meanrelerror_nonRT = np.mean(rel_err_nonRT)
    meansquarederror_nonRT = np.mean(sq_err_nonRT)
        
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(R.shape[0]))    
    det_is_one = np.abs(np.linalg.det(R) - 1.0) < 1e-10    
    t_shape_correct = t.shape == (param_hat.shape[1],)
    if not (orthogonality_error < 1e-10 and det_is_one and t_shape_correct):
        print("Orthogonality error: {}".format(orthogonality_error))
        # raise AttributeError("Error in solving projection problem?")

    return R, t, meansquarederror, meansquarederror_nonRT, meanrelerror, meanrelerror_nonRT


def compute_and_plot_mse(theta_true, theta_hat, fullscan, iteration, args, param_positions_dict,
                        plot_online=True, mse_theta_full=[], err_theta_full=[], fig_x=None, fig_z=None, fig_x_err=None, fig_z_err=None, mse_x_list=[], mse_z_list=[],
                        mse_x_nonRT_list=[], mse_z_nonRT_list=[], err_x_list=[], err_z_list=[], err_x_nonRT_list=[], err_z_nonRT_list=[],
                        per_param_sq_ers=dict(), per_param_ers=dict(), per_param_heats=dict(), xbox=[], plot_restarts=[], fastrun=False, 
                        target_param=None, subset_coord2plot=None, seedint=1234):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true = args
    
    # compute with full theta vector - relative error and relative squared error
    rel_err  = (theta_true - theta_hat)/theta_true
    rel_se = rel_err**2    
    if fastrun is False:
        per_param_heats["theta_sq_e"].append(rel_se)
        per_param_heats["theta_e"].append(rel_err)
    # mean relative error
    mse_theta_full.append(float(np.mean(rel_se)))
    err_theta_full.append(float(np.mean(rel_err)))
    
    ###############################
    if plot_online and (fastrun is False):  
        fig = go.Figure(data=go.Heatmap(z=per_param_heats["theta_sq_e"], colorscale = 'Viridis'))
        savename = "{}/theta_heatmap/theta_full_relativised_squarederror.html".format(DIR_out)
        pathlib.Path("{}/theta_heatmap/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig, savename, xaxis_title="Coordinate", yaxis_title="Iteration", 
                                title="", showgrid=False, showlegend=True, print_png=True, 
                                print_html=False, print_pdf=False)        
        fig = go.Figure(data=go.Heatmap(z=per_param_heats["theta_e"], colorscale = 'Viridis'))
        savename = "{}/theta_heatmap/theta_full_relativised_error.html".format(DIR_out)
        pathlib.Path("{}/theta_heatmap/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
        fix_plot_layout_and_save(fig, savename, xaxis_title="Coordinate", yaxis_title="Iteration", 
                                title="", showgrid=False, showlegend=True, print_png=True, 
                                print_html=False, print_pdf=False)        
    ###############################

    params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)
    X_true = np.asarray(params_true["X"]) # d x K       
    Z_true = np.asarray(params_true["Z"]) # d x J                         
    params_hat = optimisation_dict2params(theta_hat, param_positions_dict, J, K, d, parameter_names)

    for param in parameter_names:        
        if param in ["gamma", "delta", "sigma_e"]:
            # scalars
            rel_err = (params_true[param] - params_hat[param])/params_true[param]
            rel_se = rel_err**2
            # time series plots
            per_param_ers[param].append(float(rel_err))
            per_param_sq_ers[param].append(float(rel_se))
            if plot_online and (fastrun is False):
                fig_sq = make_subplots(specs=[[{"secondary_y": True}]])   
                fig = make_subplots(specs=[[{"secondary_y": True}]])   
                fig_sq.add_trace(go.Scatter(
                                        y=per_param_sq_ers[param], 
                                        showlegend=False,
                                        x=np.arange(iteration)                                    
                                    ),  secondary_y=False)
                fig_sq.add_trace(go.Scatter(
                                        y=mse_theta_full, showlegend=True,
                                        x=np.arange(iteration), 
                                        line_color="red", name="Θ Mean relative Sq. Err"                                
                                    ),  secondary_y=True)
                fig.add_trace(go.Scatter(
                                        y=per_param_ers[param], 
                                        showlegend=False,
                                        x=np.arange(iteration)                                    
                                    ),  secondary_y=False)
                fig.add_trace(go.Scatter(
                                        y=err_theta_full, showlegend=True,
                                        x=np.arange(iteration), 
                                        line_color="red", name="Θ Mean relative Err"                                
                                    ),  secondary_y=True)
                ###############################
                for itm in plot_restarts:
                    scanrep, totaliterations, halvedgammas, restarted = itm
                    if halvedgammas:
                        vcolor = "red"
                    else:
                        vcolor = "green"
                    if restarted=="fullrestart":
                        fig_sq.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                        fig.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                    else:
                        # partial restart
                        fig_sq.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                        fig.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                ###############################
                savename = "{}/timeseries_plots/squared_err/{}_squarederror.html".format(DIR_out, param)
                pathlib.Path("{}/timeseries_plots/squared_err/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                fix_plot_layout_and_save(fig_sq, savename, xaxis_title="Annealing iterations", yaxis_title="Relative Squared error", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=False, print_pdf=False)
                savename = "{}/timeseries_plots/err/{}_error.html".format(DIR_out, param)
                pathlib.Path("{}/timeseries_plots/err/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                fix_plot_layout_and_save(fig, savename, xaxis_title="Annealing iterations", yaxis_title="Relative error", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=False, print_pdf=False)                
        else:
            if param == "X":
                X_hat = np.asarray(params_hat[param]) # d x K       
                X_hat_vec = np.asarray(params_hat[param]).reshape((d*K,), order="F")       
                X_true_vec = np.asarray(params_true[param]).reshape((d*K,), order="F")       
                rel_err = (X_true_vec - X_hat_vec)/X_true_vec
                sq_err = rel_err**2
            elif param == "Z":
                Z_hat = np.asarray(params_hat[param]) # d x J          
                Z_hat_vec = np.asarray(params_hat[param]).reshape((d*J,), order="F")         
                Z_true_vec = np.asarray(params_true[param]).reshape((d*J,), order="F")       
                rel_err = (Z_true_vec - Z_hat_vec)/Z_true_vec
                sq_err = rel_err**2
            else:
                rel_err = (params_true[param] - params_hat[param])/params_true[param]
                sq_err = rel_err**2            
            if fastrun is False:
                per_param_heats["{}_sq_e".format(param)].append(sq_err)   
                per_param_heats["{}_e".format(param)].append(rel_err)               
            # mean relative error
            per_param_sq_ers[param].append(float(np.mean(sq_err)))
            per_param_ers[param].append(float(np.mean(rel_err)))           
            if plot_online and (fastrun is False):                
                pathlib.Path("{}/params_heatmap/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                fig = go.Figure(data=go.Heatmap(z=per_param_heats["{}_sq_e".format(param)], colorscale = 'Viridis'))
                savename = "{}/params_heatmap/{}_relativised_squarederror.html".format(DIR_out, param)                
                fix_plot_layout_and_save(fig, savename, xaxis_title="Coordinate", yaxis_title="Iteration", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                        print_pdf=False)            
                fig = go.Figure(data=go.Heatmap(z=per_param_heats["{}_e".format(param)], colorscale = 'Viridis'))
                savename = "{}/params_heatmap/{}_relativised_error.html".format(DIR_out, param)                
                fix_plot_layout_and_save(fig, savename, xaxis_title="Coordinate", yaxis_title="Iteration", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                        print_pdf=False)                            
                for param2plot in ["{}_sq_e".format(param), "{}_e".format(param)]:
                    parray = np.stack(per_param_heats[param2plot])
                    # timeseries plots
                    for pidx in range(parray.shape[1]):
                        if ((subset_coord2plot is not None) and (pidx not in subset_coord2plot)):
                            print("Plotting subset of coordinates...")
                            continue
                        fig = make_subplots(specs=[[{"secondary_y": True}]]) 
                        fig.add_trace(go.Scatter(
                                        y=parray[:, pidx], showlegend=False,
                                        x=np.arange(iteration)                                    
                                    ), secondary_y=False)
                        if "sq_e" in param2plot:
                            fig.add_trace(go.Scatter(
                                            y=mse_theta_full, showlegend=True,
                                            x=np.arange(iteration), 
                                            line_color="red", name="Θ Mean relative Sq. Err"                                
                                        ),  secondary_y=True)
                        else:
                            fig.add_trace(go.Scatter(
                                            y=err_theta_full, showlegend=True,
                                            x=np.arange(iteration), 
                                            line_color="red", name="Θ Mean relative Err"                                
                                        ),  secondary_y=True)
                        ###############################
                        for itm in plot_restarts:
                            scanrep, totaliterations, halvedgammas, restarted = itm
                            if halvedgammas:
                                vcolor = "red"
                            else:
                                vcolor = "green"
                            if restarted == "fullrestart":
                                fig.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                            label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                            font=dict(size=16, family="Times New Roman"),),)
                            else:
                                # partial restart
                                fig.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                            label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                            font=dict(size=16, family="Times New Roman"),),)
                        ###############################
                        if "sq_e" in param2plot:
                            savename = "{}/timeseries_plots/squared_err/{}_idx_{}_relativised_squarederror.html".format(DIR_out, param, pidx)
                            pathlib.Path("{}/timeseries_plots/squared_err/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                            fix_plot_layout_and_save(fig, savename, xaxis_title="Annealing iterations", yaxis_title="Relative Squared error", title="", 
                                                    showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                                    print_pdf=False)
                        else:
                            savename = "{}/timeseries_plots/err/{}_idx_{}_relativised_error.html".format(DIR_out, param, pidx)
                            pathlib.Path("{}/timeseries_plots/err/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
                            fix_plot_layout_and_save(fig, savename, xaxis_title="Annealing iterations", yaxis_title="Relative error", title="", 
                                                    showgrid=False, showlegend=True, print_png=True, print_html=False, 
                                                    print_pdf=False)     
    if fastrun is False:
        if fig_x is None:
            fig_x = go.Figure()  
        if fig_z is None:
            fig_z = go.Figure()  
        if fig_x_err is None:
            fig_x_err = go.Figure()  
        if fig_z_err is None:
            fig_z_err = go.Figure()      
        if fullscan not in xbox:
            xbox.extend([fullscan]*2)
        if target_param == "X" or target_param is None:
            # mean error over all elements of the matrices  
            Rx, tx, mse_x, mse_x_nonRT, meanrelerror_x, meanrelerror_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)
            mse_x_list.append(mse_x)
            mse_x_nonRT_list.append(mse_x_nonRT)
            err_x_list.append(meanrelerror_x)
            err_x_nonRT_list.append(meanrelerror_x_nonRT)
            per_param_sq_ers["X_rot_translated_mseOverMatrix"].append(mse_x)                         
            per_param_sq_ers["X_mseOverMatrix"].append(mse_x_nonRT)
            per_param_ers["X_rot_translated_errOverMatrix"].append(meanrelerror_x)
            per_param_ers["X_errOverMatrix"].append(meanrelerror_x_nonRT)
        else:
            if len(mse_x_list) > 0:
                mse_x_list.append(mse_x_list[-1])
                mse_x_nonRT_list.append(mse_x_nonRT_list[-1])
                err_x_list.append(err_x_list[-1])
                err_x_nonRT_list.append(err_x_nonRT_list[-1])
                per_param_sq_ers["X_rot_translated_mseOverMatrix"].append(per_param_sq_ers["X_rot_translated_mseOverMatrix"][-1])                         
                per_param_sq_ers["X_mseOverMatrix"].append(per_param_sq_ers["X_mseOverMatrix"][-1])
                per_param_ers["X_rot_translated_errOverMatrix"].append(per_param_ers["X_rot_translated_errOverMatrix"][-1])                         
                per_param_ers["X_errOverMatrix"].append(per_param_ers["X_errOverMatrix"][-1])
            else:
                mse_x_list.append(None)
                mse_x_nonRT_list.append(None)
                err_x_list.append(None)
                err_x_nonRT_list.append(None)
                per_param_sq_ers["X_rot_translated_mseOverMatrix"].append(None)
                per_param_sq_ers["X_mseOverMatrix"].append(None)     
                per_param_ers["X_rot_translated_errOverMatrix"].append(None)
                per_param_ers["X_errOverMatrix"].append(None)     
        if plot_online:       
            fig_x.add_trace(go.Box(
                            y=np.asarray([xxx for xxx in mse_x_list if xxx is not None]).flatten().tolist(), 
                            showlegend=False, x=xbox, 
                            name="X - total iter. {}".format(iteration),
                            boxpoints="outliers", line=dict(color="blue")
                            ))
            fig_x.add_trace(go.Box(
                            y=np.asarray([xxx for xxx in mse_x_nonRT_list if xxx is not None]).flatten().tolist(), 
                            opacity=0.5, showlegend=False, x=xbox, 
                            name="X (nonRT) - total iter. {}".format(iteration),
                            boxpoints="outliers", line=dict(color="green")
                            ))
            fig_x.update_layout(boxmode="group")   
            savename = "{}/xz_boxplots/relative_mse_x.html".format(DIR_out)
            pathlib.Path("{}/xz_boxplots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
            fix_plot_layout_and_save(fig_x, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=True,
                                print_png=True, print_html=True, print_pdf=False)            
            fig_x_err.add_trace(go.Box(
                            y=np.asarray([xxx for xxx in err_x_list if xxx is not None]).flatten().tolist(), 
                            showlegend=False, x=xbox, 
                            name="X - total iter. {}".format(iteration),
                            boxpoints="outliers", line=dict(color="blue")
                            ))
            fig_x_err.add_trace(go.Box(
                            y=np.asarray([xxx for xxx in err_x_nonRT_list if xxx is not None]).flatten().tolist(), 
                            opacity=0.5, showlegend=False, x=xbox, 
                            name="X (nonRT) - total iter. {}".format(iteration),
                            boxpoints="outliers", line=dict(color="green")
                            ))
            fig_x_err.update_layout(boxmode="group")   
            savename = "{}/xz_boxplots/relative_meanerr_x.html".format(DIR_out)
            pathlib.Path("{}/xz_boxplots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
            fix_plot_layout_and_save(fig_x_err, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=True,
                                print_png=True, print_html=True, print_pdf=False)            
            figX_sq = make_subplots(specs=[[{"secondary_y": True}]]) 
            figX_sq.add_trace(go.Scatter(
                                    y=per_param_sq_ers["X_mseOverMatrix"], 
                                    x=np.arange(iteration),
                                    name="X - min Mean relative Sq. Err", 
                                    line_color="green"
                                ),  secondary_y=False)
            figX_sq.add_trace(go.Scatter(
                                    y=per_param_sq_ers["X_rot_translated_mseOverMatrix"], 
                                    x=np.arange(iteration),  
                                    line=dict(color="blue"),
                                    name="X - min Mean relative Sq. Err<br>(under rot/transl)"
                                ),  secondary_y=False)
            figX_sq.add_trace(go.Scatter(
                                    y=mse_theta_full, 
                                    x=np.arange(iteration), 
                                    line_color="red", 
                                    name="Θ Mean relative Sq. Err"                                
                                ),  secondary_y=True)
            figX = make_subplots(specs=[[{"secondary_y": True}]]) 
            figX.add_trace(go.Scatter(
                                    y=per_param_ers["X_errOverMatrix"], 
                                    x=np.arange(iteration),
                                    name="X - min Mean relative Err", line_color="green"
                                ),  secondary_y=False)
            figX.add_trace(go.Scatter(
                                    y=per_param_ers["X_rot_translated_errOverMatrix"], 
                                    x=np.arange(iteration), line=dict(color="blue"),
                                    name="X - min Mean relative Err<br>(under rot/transl)"
                                ),  secondary_y=False)
            figX.add_trace(go.Scatter(
                                    y=err_theta_full, 
                                    x=np.arange(iteration), 
                                    line_color="red", 
                                    name="Θ Mean relative Err"                                
                                ),  secondary_y=True)
        if target_param == "Z" or target_param is None:
            Rz, tz, mse_z, mse_z_nonRT, meanrelerror_z, meanrelerror_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)
            mse_z_list.append(mse_z)   
            mse_z_nonRT_list.append(mse_z_nonRT)
            err_z_list.append(meanrelerror_z)
            err_z_nonRT_list.append(meanrelerror_z_nonRT)
            per_param_sq_ers["Z_rot_translated_mseOverMatrix"].append(mse_z)            
            per_param_sq_ers["Z_mseOverMatrix"].append(mse_z_nonRT)   
            per_param_ers["Z_rot_translated_errOverMatrix"].append(meanrelerror_z)
            per_param_ers["Z_errOverMatrix"].append(meanrelerror_z_nonRT)   
        else:
            if len(mse_z_list) > 0:
                mse_z_list.append(mse_z_list[-1])
                mse_z_nonRT_list.append(mse_z_nonRT_list[-1])
                err_z_list.append(err_z_list[-1])
                err_z_nonRT_list.append(err_z_nonRT_list[-1])
                per_param_sq_ers["Z_rot_translated_mseOverMatrix"].append(per_param_sq_ers["Z_rot_translated_mseOverMatrix"][-1])                         
                per_param_sq_ers["Z_mseOverMatrix"].append(per_param_sq_ers["Z_mseOverMatrix"][-1])
                per_param_ers["Z_rot_translated_errOverMatrix"].append(per_param_ers["Z_rot_translated_errOverMatrix"][-1])                         
                per_param_ers["Z_errOverMatrix"].append(per_param_ers["Z_errOverMatrix"][-1])
            else:
                mse_z_list.append(None)
                mse_z_nonRT_list.append(None)
                err_z_list.append(None)
                err_z_nonRT_list.append(None)
                per_param_sq_ers["Z_rot_translated_mseOverMatrix"].append(None)
                per_param_sq_ers["Z_mseOverMatrix"].append(None)        
                per_param_ers["Z_rot_translated_errOverMatrix"].append(None)
                per_param_ers["Z_errOverMatrix"].append(None)        
        if plot_online:
            fig_z.add_trace(go.Box(
                                y=np.asarray([zzz for zzz in mse_z_list if zzz is not None]).flatten().tolist(), 
                                showlegend=False, x=xbox, 
                                name="Z - total iter. {}".format(iteration),
                                boxpoints="outliers", line=dict(color="blue")
                                ))
            fig_z.add_trace(go.Box(
                                y=np.asarray([zzz for zzz in mse_z_nonRT_list if zzz is not None]).flatten().tolist(), 
                                opacity=0.5, showlegend=False, x=xbox, 
                                name="Z (nonRT) - total iter. {}".format(iteration),
                                boxpoints="outliers", line=dict(color="green")
                                ))
            fig_z.update_layout(boxmode="group")               
            savename = "{}/xz_boxplots/relative_mse_z.html".format(DIR_out)
            pathlib.Path("{}/xz_boxplots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
            fix_plot_layout_and_save(fig_z, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=True,
                                print_png=True, print_html=True, print_pdf=False)
            fig_z_err.add_trace(go.Box(
                                y=np.asarray([zzz for zzz in err_z_list if zzz is not None]).flatten().tolist(), 
                                showlegend=False, x=xbox, 
                                name="Z - total iter. {}".format(iteration),
                                boxpoints="outliers", line=dict(color="blue")
                                ))
            fig_z_err.add_trace(go.Box(
                                y=np.asarray([zzz for zzz in err_z_nonRT_list if zzz is not None]).flatten().tolist(), 
                                opacity=0.5, showlegend=False, x=xbox, 
                                name="Z (nonRT) - total iter. {}".format(iteration),
                                boxpoints="outliers", line=dict(color="green")
                                ))
            fig_z_err.update_layout(boxmode="group")               
            savename = "{}/xz_boxplots/relative_meanerr_z.html".format(DIR_out)
            pathlib.Path("{}/xz_boxplots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
            fix_plot_layout_and_save(fig_z_err, savename, xaxis_title="", yaxis_title="", title="", showgrid=False, showlegend=True,
                                print_png=True, print_html=True, print_pdf=False)
            figZ_sq = make_subplots(specs=[[{"secondary_y": True}]]) 
            figZ_sq.add_trace(go.Scatter(
                                    y=per_param_sq_ers["Z_rot_translated_mseOverMatrix"], 
                                    x=np.arange(iteration), line=dict(color="blue"),
                                    name="Z - min Mean relative Sq. Err<br>(under rot/trl)"
                                ),  secondary_y=False)
            figZ_sq.add_trace(go.Scatter(
                                    y=per_param_sq_ers["Z_mseOverMatrix"], 
                                    x=np.arange(iteration),
                                    name="Z - min Mean relative Sq. Err", 
                                    line=dict(color="green")
                                ),  secondary_y=False)
            figZ_sq.add_trace(go.Scatter(
                                    y=mse_theta_full, 
                                    x=np.arange(iteration), line_color="red", 
                                    name="Θ Mean relative Sq. Err"                                
                                ),  secondary_y=True)
            figZ = make_subplots(specs=[[{"secondary_y": True}]]) 
            figZ.add_trace(go.Scatter(
                                    y=per_param_ers["Z_rot_translated_errOverMatrix"], 
                                    x=np.arange(iteration), line=dict(color="blue"),
                                    name="Z - min Mean relative Err<br>(under rot/trl)"
                                ),  secondary_y=False)
            figZ.add_trace(go.Scatter(
                                    y=per_param_ers["Z_errOverMatrix"], 
                                    x=np.arange(iteration), 
                                    name="Z - min Mean relative Err", 
                                    line=dict(color="green")
                                ),  secondary_y=False)
            figZ.add_trace(go.Scatter(
                                    y=err_theta_full, 
                                    x=np.arange(iteration), 
                                    line_color="red", 
                                    name="Θ Mean relative Err"                                
                                ),  secondary_y=True)
        ###############################
        if plot_online:
            for itm in plot_restarts:            
                scanrep, totaliterations, halvedgammas, restarted = itm
                if halvedgammas:
                    vcolor = "red"
                else:
                    vcolor = "green"
                if restarted=="fullrestart":
                    if target_param == "X" or target_param is None:
                        figX_sq.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                        figX.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                    label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                    font=dict(size=16, family="Times New Roman"),),)
                    elif target_param == "Z" or target_param is None:
                        figZ_sq.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                font=dict(size=16, family="Times New Roman"),),)
                        figZ.add_vline(x=totaliterations, opacity=1, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                font=dict(size=16, family="Times New Roman"),),)
                else:
                    # partial restart
                    if target_param == "X" or target_param is None:
                        figX_sq.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                font=dict(size=16, family="Times New Roman"),),)
                        figX.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                font=dict(size=16, family="Times New Roman"),),)
                    elif target_param == "Z" or target_param is None:
                        figZ_sq.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                font=dict(size=16, family="Times New Roman"),),)
                        figZ.add_vline(x=totaliterations, opacity=0.5, line_width=2, line_dash="dash", line_color=vcolor, showlegend=False, 
                                label=dict(text="l={}, total_iter={}".format(scanrep, totaliterations), textposition="top left",
                                font=dict(size=16, family="Times New Roman"),),)
            ###############################
            pathlib.Path("{}/timeseries_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)     
            if target_param == "X" or target_param is None:
                pathlib.Path("{}/timeseries_plots/".format(DIR_out)).mkdir(parents=True, exist_ok=True)                                 
                savenameX = "{}/timeseries_plots/X_rot_translated_relative_mse.html".format(DIR_out)                
                fix_plot_layout_and_save(figX_sq, savenameX, xaxis_title="", yaxis_title="Mean relative Sq. Err", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=True, print_pdf=False)
                savenameX = "{}/timeseries_plots/X_rot_translated_relative_err.html".format(DIR_out)                
                fix_plot_layout_and_save(figX, savenameX, xaxis_title="", yaxis_title="Mean relative Err", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=True, print_pdf=False)
            elif target_param == "Z" or target_param is None:                
                savenameZ = "{}/timeseries_plots/Z_rot_translated_relative_mse.html".format(DIR_out)                
                fix_plot_layout_and_save(figZ_sq, savenameZ, xaxis_title="", yaxis_title="Mean relative Sq. Err", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=True, print_pdf=False)
                savenameZ = "{}/timeseries_plots/Z_rot_translated_relative_err.html".format(DIR_out)                
                fix_plot_layout_and_save(figZ, savenameZ, xaxis_title="", yaxis_title="Mean relative Err", title="", 
                                        showgrid=False, showlegend=True, print_png=True, print_html=True, print_pdf=False)

    return mse_theta_full, err_theta_full, mse_x_list, mse_z_list, mse_x_nonRT_list, mse_z_nonRT_list, \
                fig_x, fig_z, per_param_sq_ers, per_param_ers, per_param_heats, xbox,\
                    err_x_list, err_z_list, err_x_nonRT_list, err_z_nonRT_list, fig_x_err, fig_z_err


def get_parameter_name_and_vector_coordinate(param_positions_dict, i, d):

    for param in param_positions_dict.keys():
        if param_positions_dict[param][0] <= i and param_positions_dict[param][1] > i: # not equal to upper limit
            target_param = param
            index_in_flat_vector = i - param_positions_dict[param][0]            
            if target_param in ["X", "Z", "Phi"]:                
                column = index_in_flat_vector // d
                row = index_in_flat_vector % d
                # print(target_param, i, column, row)
                return target_param, column, row
            else:
                # alpha, beta are vectors, not flattened matrices
                return target_param, index_in_flat_vector, index_in_flat_vector
        else:
            continue
    raise AttributeError("Should never reach this point.")


def get_posterior_for_optimisation_vec(param, Y, idx, vector_index_in_param_matrix, vector_coordinate, theta, gamma, param_positions_dict, args):

    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    if param == "X":
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_x_il(x, vector_coordinate, vector_index_in_param_matrix, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, 
                                                                  prior_loc_x[vector_coordinate], prior_scale_x[vector_coordinate, vector_coordinate], gamma)
        else:
            post2optim = lambda x: log_conditional_posterior_x_vec(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_x, prior_scale_x, gamma)  
    elif param == "Z":
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_z_jl(x, vector_coordinate, vector_index_in_param_matrix, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z[vector_coordinate], 
                                                                prior_scale_z[vector_coordinate, vector_coordinate], gamma, constant_Z, penalty_weight_Z)
        else:
            post2optim = lambda x: log_conditional_posterior_z_vec(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z, prior_scale_z, gamma, 
                                                                constant_Z, penalty_weight_Z)
    elif param == "Phi":            
        if elementwise and isinstance(vector_coordinate, int):
            post2optim = lambda x: log_conditional_posterior_phi_jl(x, vector_coordinate, vector_index_in_param_matrix, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi[vector_coordinate], 
                                                                    prior_scale_phi[vector_coordinate, vector_coordinate], gamma)
        else:
            post2optim = lambda x: log_conditional_posterior_phi_vec(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_phi, prior_scale_phi, gamma)
    elif param == "beta":
        if elementwise and isinstance(vector_coordinate, int):
            idx = vector_coordinate
        post2optim = lambda x: log_conditional_posterior_beta_i(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_beta, prior_scale_beta, gamma)
    elif param == "alpha":
        if elementwise and isinstance(vector_coordinate, int):
            idx = vector_coordinate
        post2optim = lambda x: log_conditional_posterior_alpha_j(x, idx, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_alpha, prior_scale_alpha, gamma)
    elif param == "gamma":
        post2optim = lambda x: log_conditional_posterior_gamma(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=gamma, prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma) 
    elif param == "delta":
        post2optim = lambda x: log_conditional_posterior_delta(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma, prior_loc_delta, prior_scale_delta)        
    elif param == "sigma_e":
        post2optim = lambda x: log_conditional_posterior_sigma_e(x, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma, prior_loc_sigmae, prior_scale_sigmae, min_sigma_e) 
        
    return post2optim


def get_evaluation_grid(param, vector_coordinate, args, gridpoints_num_plot=None):

    grid_width_std = 5    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    if gridpoints_num_plot is not None:
        gridpoints_num = gridpoints_num_plot
    
    gridpoints_num_alpha_beta = gridpoints_num**2
    xx_ = None

    if param == "alpha":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, grid_width_std*np.sqrt(prior_scale_alpha)+prior_loc_alpha, gridpoints_num_alpha_beta).tolist()
    elif param == "beta":
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, grid_width_std*np.sqrt(prior_scale_beta)+prior_loc_beta, gridpoints_num_alpha_beta).tolist()
    elif param == "gamma":            
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, grid_width_std*np.sqrt(prior_scale_gamma)+prior_loc_gamma, gridpoints_num_alpha_beta).tolist()
        if 0.0 in grid:
            grid.remove(0.0)
    elif param == "delta":                    
        grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, grid_width_std*np.sqrt(prior_scale_delta)+prior_loc_delta, gridpoints_num_alpha_beta).tolist()
        if 0.0 in grid:
            grid.remove(0.0)            
    elif param == "sigma_e":
        grid = np.linspace(min_sigma_e, grid_width_std*np.sqrt(prior_scale_sigmae)+prior_loc_sigmae, gridpoints_num_alpha_beta).tolist()
    else:           
        if d == 1 or elementwise and vector_coordinate is not None:
            if param == "Phi":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], grid_width_std*np.sqrt(prior_scale_phi[vector_coordinate, vector_coordinate])+prior_loc_phi[vector_coordinate], gridpoints_num).tolist()                
            elif param == "Z":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], grid_width_std*np.sqrt(prior_scale_z[vector_coordinate, vector_coordinate])+prior_loc_z[vector_coordinate], gridpoints_num).tolist()                
            elif param == "X":
                grid = np.linspace(-grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], grid_width_std*np.sqrt(prior_scale_x[vector_coordinate, vector_coordinate])+prior_loc_x[vector_coordinate], gridpoints_num).tolist()                            
        elif (d > 1 and d <= 5) and not elementwise:
            if param == "Phi":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
                xx_ = np.linspace(-grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], grid_width_std*np.sqrt(prior_scale_phi[0,0])+prior_loc_phi[0], gridpoints_num).tolist()
            elif param == "Z":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
                xx_ = np.linspace(-grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], grid_width_std*np.sqrt(prior_scale_z[0,0])+prior_loc_z[0], gridpoints_num).tolist()
            elif param == "X":
                unidimensional_grid = [np.linspace(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], gridpoints_num).tolist() for i in range(d)]
                grid = itertools.product(*unidimensional_grid)
                xx_ = np.linspace(-grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], grid_width_std*np.sqrt(prior_scale_x[0,0])+prior_loc_x[0], gridpoints_num).tolist()
        else:
            raise NotImplementedError("Use a Sobol sequence to generate a grid in such high dimensional space.")
    if xx_ is None:
        xx_ = grid.copy()

    return grid, xx_


def plot_posterior_elementwise(outdir, param, Y, idx, vector_coordinate, theta_curr, gamma, param_positions_dict, args, 
                            true_param=None, hat_param=None, iteration=None, fig_in=None, plot_arrows=False, all_theta=None):
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args

    f = get_posterior_for_optimisation_vec(param=param, Y=Y, idx=idx, vector_index_in_param_matrix=idx, vector_coordinate=vector_coordinate, theta=theta_true.copy(), 
                                           gamma=gamma, param_positions_dict=param_positions_dict, args=args)
    
    gridpoints_num_plot = 20

    xx, xx_ = get_evaluation_grid(param, vector_coordinate, args, gridpoints_num_plot=gridpoints_num_plot)      
    if vector_coordinate is None:        
        xxlist = [ix for ix in xx]                
    else:        
        xxlist = xx_    
    yy = np.asarray(list(map(f, xxlist)))

    # plot : all_theta[param][idx] for X, Phi, Z or all_theta[param][vector_coordinate] for alpha, beta, or all_theta[param] for scalars
    if isinstance(idx, int) and vector_coordinate is None:
        fig = go.Figure()
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])   
    # density plot
    if vector_coordinate is not None:
        fig.add_trace(go.Scatter(
                        x=xx_,
                        y=yy,
                        mode='lines',
                        name="{}: cond. posterior".format(param),
                        line=dict(
                            color='royalblue',
                            width=2
                        ),
                        hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ), secondary_y=False,)
    else:        
        fig.add_trace(go.Contour(
                        x=xx_,
                        y=xx_,
                        z=yy,
                        colorscale='Hot',
                        contours=dict(showlabels=True),
                        colorbar=dict(
                            title='Cond. posterior',
                            titleside='right'
                        )
                    )
                )
        fig.update_layout(xaxis_title='x1', yaxis_title='x2')

    if (idx is None and vector_coordinate==0 and true_param is not None) or (isinstance(idx, int) and isinstance(vector_coordinate, int)):
        # scalar param        
        fig.add_vline(x=true_param, line_width=3, line_dash="dash", line_color="green", name="True θ", showlegend=True)
    elif isinstance(idx, int) and vector_coordinate is None:
        # surface plot
        fig.add_trace(go.Scatter(x=[true_param[0]], y=[true_param[1]], name="True θ", mode="markers", marker_symbol="star", marker_color="green"))

    # if only plotting density
    if all_theta is None:
        fig.update_layout(hovermode='x unified', legend=dict(orientation="h"))         
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)    
        savename = "{}/{}_idx_{}_vector_coord_{}.html".format(outdir, param, idx, vector_coordinate)
        if param in ["X", "Z", "Phi"] and vector_coordinate is None:
            yaxistitle = ""
        else:
            yaxistitle = "Conditional posterior"
        fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title=yaxistitle, title="", 
                                showgrid=False, showlegend=True, print_png=True, print_html=False, print_pdf=False)
        return fig

    if param in ["X", "Z", "Phi"] and vector_coordinate is None:
        # 2d plot
        if d != 2:
            raise NotImplementedError("Only implemented for 2D ideal points space.")
        try:
            y1 = np.asarray([itm[2] for itm in all_theta[param][idx][0]])
            y2 = np.asarray([itm[2] for itm in all_theta[param][idx][1]])
        except:
            y1 = []
            y2 = []
            print("Not enough data for both coordinates yet.")            
            pass
        if len(y1) > 0 and len(y2) > 0:
            if len(y1) > len(y2):
                all_theta[param][idx][1].append(all_theta[param][idx][1][-1])
                y2 = np.asarray([itm[2] for itm in all_theta[param][idx][1]])
            elif len(y2) > len(y1):
                all_theta[param][idx][0].append(all_theta[param][idx][0][-1])
                y1 = np.asarray([itm[2] for itm in all_theta[param][idx][0]])
            opacitylevels = np.linspace(0.1, 1, len(y1))
            text_i = np.asarray(["i: {}, total i: {}, γ = {:.3f}".format(itm[0], itm[1], gamma) for itm in all_theta[param][idx][0]])
            for iii in range(len(y1)):
                fig.add_trace(
                    go.Scatter(
                        x=[y1[iii]],
                        y=[y2[iii]],
                        mode="markers", marker_symbol="square", marker_color="blue",            
                        marker_size=8,
                        showlegend=False,
                        text=text_i,
                        hoverinfo="text"
                    )
                )   
            for iii in range(len(y1)-1):            
                fig.add_annotation(
                    x=y1[iii+1],
                    y=y2[iii+1],
                    ax=y1[iii],
                    ay=y2[iii],
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    opacity=opacitylevels[iii], 
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='blue'
                )        
    elif param in ["X", "Z", "Phi"] and (isinstance(idx, int) and isinstance(vector_coordinate, int)):
        # per coord plot of vector param
        opacitylevels = np.linspace(0.1, 1, len(all_theta[param][idx][vector_coordinate]))
        opc = 0
        for itm in all_theta[param][idx][vector_coordinate]:
            fig.add_vline(x=itm[2], opacity=opacitylevels[opc], line_width=2, line_dash="dash", line_color="red", showlegend=False) 
                        # , label=dict(text="{}, γ = {:.3f}".format(itm[1], gamma), textposition="top left",
                        # font=dict(size=16, family="Times New Roman"),),)
            fig.add_trace(go.Scatter(x=[itm[2]], y=[itm[3][0]], text="{}, γ = {}".format(itm[1], gamma), showlegend=False, 
                    mode="markers+lines", marker_symbol="square", marker_color="red", 
                    name="step: {}".format(itm[1])), secondary_y=True,)
            opc += 1
        x = np.asarray([itm[2] for itm in all_theta[param][idx][vector_coordinate]])
        y = np.asarray([itm[3][0] for itm in all_theta[param][idx][vector_coordinate]])
        for iii in range(len(x)-1):        
            fig.add_annotation(
                x=x[iii+1],
                y=y[iii+1],
                ax=x[iii],
                ay=y[iii],
                xref='x',
                yref='y2',
                axref='x',
                ayref='y2',
                opacity=opacitylevels[iii],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )       
    elif param in ["alpha", "beta"]:
        # per coord plot of vector param
        opacitylevels = np.linspace(0.1, 1, len(all_theta[param][vector_coordinate]))
        opc = 0
        for itm in all_theta[param][vector_coordinate]:
            fig.add_vline(x=itm[2], opacity=opacitylevels[opc], line_width=2, line_dash="dash", line_color="red", showlegend=False) 
                        # , label=dict(text="{}, γ = {:.3f}".format(itm[1], gamma), textposition="top left",
                        # font=dict(size=16, family="Times New Roman"),),)
            fig.add_trace(go.Scatter(x=[itm[2]], y=[itm[3][0]], text="{}, γ = {}".format(itm[1], gamma),
                    mode="markers+lines", marker_symbol="square", marker_color="red", showlegend=False, 
                    name="step: {}".format(itm[1])), secondary_y=True,)
            opc += 1
        x = np.asarray([itm[2] for itm in all_theta[param][vector_coordinate]])
        y = np.asarray([itm[3][0] for itm in all_theta[param][vector_coordinate]])
        for iii in range(len(x)-1):       
            fig.add_annotation(
                x=x[iii+1],
                y=y[iii+1],
                ax=x[iii],
                ay=y[iii],
                xref='x',
                yref='y2',
                axref='x',
                ayref='y2',
                opacity=opacitylevels[iii],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )        
    else:
        # scalar plot
        opacitylevels = np.linspace(0.2, 1, len(all_theta[param]))
        opc = 0
        for itm in all_theta[param]:
            fig.add_vline(x=itm[2], opacity=opacitylevels[opc], line_width=2, line_dash="dash", line_color="red", showlegend=False), 
                        # , label=dict(text="{}, γ = {:.3f}".format(itm[1], gamma), textposition="top left",
                        # font=dict(size=16, family="Times New Roman"),),)
            fig.add_trace(go.Scatter(x=[itm[2]], y=[itm[3][0]], text="{}, γ = {}".format(itm[1], gamma),
                    mode="markers+lines", marker_symbol="square", marker_color="red", showlegend=False,
                    name="step: {}".format(itm[1])), secondary_y=True,)
            opc += 1
        x = np.asarray([itm[2] for itm in all_theta[param]])
        y = np.asarray([itm[3][0] for itm in all_theta[param]])
        for iii in range(len(x)-1):    
            fig.add_annotation(
                x=x[iii+1],
                y=y[iii+1],
                ax=x[iii],
                ay=y[iii],
                xref='x',
                yref='y2',
                axref='x',
                ayref='y2',
                opacity=opacitylevels[iii],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )       
    
    fig.update_layout(hovermode='x unified', legend=dict(orientation="h")) 
    if not (isinstance(idx, int) and vector_coordinate is None):
        fig.update_layout(yaxis2=dict(title="Data loglikelihood")) 
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)    
    savename = "{}/{}_idx_{}_vector_coord_{}.html".format(outdir, param, idx, vector_coordinate)
    if param in ["X", "Z", "Phi"] and vector_coordinate is None:
        yaxistitle = ""
    else:
        yaxistitle = "Conditional posterior"
    fix_plot_layout_and_save(fig, savename, xaxis_title="", yaxis_title=yaxistitle, title="", 
                            showgrid=False, showlegend=True, print_png=True, print_html=False, print_pdf=False)

    return fig


def plot_posteriors_during_estimation(Y, iteration, plotting_thetas, theta_curr, vect_iter, fig_posteriors, 
                                    fig_posteriors_annealed, gamma, param_positions_dict, args, plot_arrows=False, 
                                    testparam=None, testidx=None, testvec=None):
    
    
    DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true  = args
    
    data_loglik, data_fullposterior = log_full_posterior(Y, theta_curr.copy(), param_positions_dict, args)

    # non-annealed posterior
    for theta_i in range(parameter_space_dim):       

        # if theta_i not in [10, 75, 82, 115, 120, 121]: ####################################################
        #     continue
        
        if testparam is not None and elementwise and testidx is not None and param_positions_dict[testparam][0] + testidx != theta_i:
            continue

        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=theta_i, d=d)   

        if testparam is not None and testparam != target_param:
            continue

        if testparam is not None and not elementwise and testvec != vector_index_in_param_matrix:
            continue

        if target_param in ["gamma", "delta", "sigma_e"]:
            if len(plotting_thetas[target_param]) == 0:                
                plotting_thetas[target_param].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
            elif len(plotting_thetas[target_param]) >= 1 and plotting_thetas[target_param][-1][2] != theta_curr[theta_i]:
                # plot if parameter value has moved
                plotting_thetas[target_param].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
        else:            
            if isinstance(plotting_thetas[target_param], list) and len(plotting_thetas[target_param]) == 0:                
                plotting_thetas[target_param] = dict()
                if target_param in ["X", "beta"]:                    
                    for kk in range(K):
                        if target_param == "X":
                            plotting_thetas[target_param][kk] = dict()
                            for dd in range(d):
                                plotting_thetas[target_param][kk][dd] = []                            
                        else:
                            plotting_thetas[target_param][kk] = []                            
                elif target_param in ["Phi", "Z", "alpha"]:
                    for jj in range(J):
                        if target_param in ["Z", "Phi"]:
                            plotting_thetas[target_param][jj] = dict()
                            for dd in range(d):
                                plotting_thetas[target_param][jj][dd] = []                              
                        else:
                            plotting_thetas[target_param][jj] = []                                  
                if target_param in ["X", "Z", "Phi"]:
                    plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
                else:
                    plotting_thetas[target_param][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
            else:
                if target_param in ["X", "Z", "Phi"]:                    
                    if len(plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate]) == 0:                        
                        plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
                    elif len(plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate]) >= 1 and \
                        plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate][-1][2] != theta_curr[theta_i]:
                        # plot if parameter value has moved
                        plotting_thetas[target_param][vector_index_in_param_matrix][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
                else:
                    if len(plotting_thetas[target_param][vector_coordinate]) == 0:                        
                        plotting_thetas[target_param][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))
                    elif len(plotting_thetas[target_param][vector_coordinate]) >= 1 and \
                        plotting_thetas[target_param][vector_coordinate][-1][2] != theta_curr[theta_i]:
                        # plot if parameter value has moved
                        plotting_thetas[target_param][vector_coordinate].append((vect_iter, iteration, theta_curr[theta_i], data_fullposterior))

        if elementwise or (not elementwise and target_param in ["alpha", "beta", "gamma", "delta", "sigma_e"]):                                         
            fig_posteriors[target_param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=target_param, 
                        Y=Y, idx=vector_index_in_param_matrix, vector_coordinate=vector_coordinate, 
                        theta_curr=theta_curr.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[theta_i], 
                        hat_param=theta_curr[theta_i], iteration=iteration,
                        fig_in=fig_posteriors[target_param], plot_arrows=plot_arrows, all_theta=plotting_thetas)     
    # vector 2D plots
    if not elementwise:
        for param in parameter_names:
            if testparam is not None and (testparam != param):
                continue
            if param in ["X"]:
                for i in range(K):         
                    if testparam is not None and testvec != i:         
                        continue
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=i, vector_coordinate=None, 
                        theta_curr=theta_curr.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d].copy(), 
                        hat_param=theta_curr[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d].copy(), iteration=iteration,
                        fig_in=fig_posteriors[param], plot_arrows=plot_arrows, all_theta=plotting_thetas)                    
            elif param in ["Z", "Phi"]:
                for j in range(J):                                    
                    if testparam is not None and testvec != j:         
                        continue
                    fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors/".format(DIR_out), param=param, Y=Y, idx=j, vector_coordinate=None, 
                        theta_curr=theta_curr.copy(), gamma=1, param_positions_dict=param_positions_dict, args=args, 
                        true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d].copy(), 
                        hat_param=theta_curr[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d].copy(), iteration=iteration, 
                        fig_in=fig_posteriors[param], plot_arrows=plot_arrows, all_theta=plotting_thetas)    
        
    if gamma != 1:
        # annealed posterior
        for theta_i in range(parameter_space_dim):     


            # if theta_i not in [10, 75, 82, 115, 120, 121]: ####################################################
            #     continue


            if testidx is not None and testparam is not None and param_positions_dict[testparam][0] + testidx != theta_i:
                continue  
            target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=theta_i, d=d)     
            if testparam is not None and testparam != target_param:
                continue
            if testparam is not None and not elementwise and testvec != vector_index_in_param_matrix:
                continue
            keyname = "param_{}_vindexParammatrix_{}_veccord_{}_gamma_{}".format(target_param, vector_index_in_param_matrix, vector_coordinate, gamma)   
            if keyname in fig_posteriors_annealed.keys():
                fig = fig_posteriors_annealed[keyname]             
            else:
                fig = go.Figure()   
            if elementwise or (not elementwise and target_param in ["alpha", "beta", "gamma", "delta", "sigma_e"]):     
                fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=target_param, 
                            Y=Y, idx=vector_index_in_param_matrix, vector_coordinate=vector_coordinate, 
                            theta_curr=theta_curr.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                            true_param=theta_true[theta_i], 
                            hat_param=theta_curr[theta_i], iteration=iteration,
                            fig_in=fig, plot_arrows=plot_arrows, all_theta=plotting_thetas)     
        # vector 2D plots
        if not elementwise:
            for param in parameter_names:
                if testparam is not None and (testparam != param): # or testidx is not None):
                    continue
                keyname = "param_{}_vindexParammatrix_{}_gamma_{}".format(param, vector_index_in_param_matrix, gamma)   
                if keyname in fig_posteriors_annealed.keys():
                    fig = fig_posteriors_annealed[keyname]             
                else:
                    fig = go.Figure()
                if param in ["X"]:
                    for i in range(K):    
                        if testparam is not None and testvec != i:         
                            continue                               
                        fig_posteriors_annealed[keyname] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, idx=i, vector_coordinate=None, 
                            theta_curr=theta_curr.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                            true_param=theta_true[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], 
                            hat_param=theta_curr[param_positions_dict[param][0]+i*d:param_positions_dict[param][0]+(i+1)*d], iteration=iteration,
                            fig_in=fig, plot_arrows=plot_arrows, all_theta=plotting_thetas)                
                elif param in ["Z", "Phi"]:
                    for j in range(J):   
                        if testparam is not None and testvec != j:         
                            continue
                        fig_posteriors[param] = plot_posterior_elementwise(outdir="{}/estimation_posteriors_annealed_gamma_{}/".format(DIR_out, gamma), param=param, Y=Y, idx=j, vector_coordinate=None, 
                            theta_curr=theta_curr.copy(), gamma=gamma, param_positions_dict=param_positions_dict, args=args, 
                            true_param=theta_true[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], 
                            hat_param=theta_curr[param_positions_dict[param][0]+j*d:param_positions_dict[param][0]+(j+1)*d], iteration=iteration, 
                            fig_in=fig, plot_arrows=plot_arrows, all_theta=plotting_thetas)               
                
    return fig_posteriors, fig_posteriors_annealed, plotting_thetas


def error_polarisation_plots(datain, estimation_folder, M, K, J, d=2):

    popularities_rel_err = []
    Zrel_err = []
    alpharel_err = []
    for m in range(M):
        # load true utilities
        with open("{}/{}/Utilities.pickle".format(datain, m), "rb") as f:
            pij = pickle.load(f)   
        popularity_js = np.sum(pij, axis=0)
        # decreasing order
        sorted_indices = np.argsort(popularity_js)[::-1]
        sorted_popularity = popularity_js[sorted_indices]
        # load estimated parameters - best solution, i.e. first line in the file
        with jsonlines.open("{}/{}/{}/params_out_global_theta_hat.jsonl".format(datain, m, estimation_folder), mode="r") as f:                    
            for result in f.iter(type=dict, skip_invalid=True):
                Xhat = np.asarray(result["X"]).reshape((d, K), order="F")     
                Zhat = np.asarray(result["Z"]).reshape((d, J), order="F")     
                gammahat = result["gamma"][0]
                alphahat = np.asarray(result["alpha"])
                betahat  = np.asarray(result["beta"])                
                sigma_ehat  = result["sigma_e"][0]                
                pijshat = p_ij_arg_numbafast(Xhat, Zhat, alphahat, betahat, gammahat, K)    
                popularity_js_hat_m = np.sum(pijshat, axis=0)
                rel_err = (sorted_popularity - popularity_js_hat_m[sorted_indices])/sorted_popularity
                popularities_rel_err.append(rel_err)
                break
        # parameters        
        with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(datain, m), mode="r") as f:                    
            for paramtrue in f.iter(type=dict, skip_invalid=True):
                X = np.asarray(paramtrue["X"]).reshape((d, K), order="F")     
                Z = np.asarray(paramtrue["Z"]).reshape((d, J), order="F")     
                gamma = paramtrue["gamma"]
                alpha = np.asarray(paramtrue["alpha"])
                beta  = np.asarray(paramtrue["beta"])                
                sigma_e  = paramtrue["sigma_e"]
                break
            # reorder columns of Z, alpha according to true rank
            Ztrue_rank = Z[:, sorted_indices]
            Ztrue_rank_vec = Ztrue_rank.reshape((d*J,), order="F")   
            Zhat_rank = Zhat[:, sorted_indices]  
            Zhat_rank_vec = Zhat_rank.reshape((d*J,), order="F")   
            rel_err_Z = (Ztrue_rank_vec - Zhat_rank_vec)/Ztrue_rank_vec
            Zrel_err.append(rel_err_Z)

            alpha_true_rank = alpha[sorted_indices]
            alpha_hat_rank = alphahat[sorted_indices]
            rel_err_alpha = (alpha_true_rank - alpha_hat_rank)/alpha_true_rank
            alpharel_err.append(rel_err_alpha)

    popularities_all_rel_err = np.stack(popularities_rel_err)
    fig = go.Figure()
    for j in range(J):
        fig.add_trace(go.Box(
                y=popularities_all_rel_err[:, j].flatten().tolist(), 
                showlegend=False, x=[j],                 
                boxpoints="outliers", line=dict(color="salmon")
                ))
    savename = "{}/popularity_error_pj_plot.html".format(datain)
    fix_plot_layout_and_save(fig, savename, xaxis_title="Popularity (most to least popular)", yaxis_title="p.j relative error", 
                            title="", showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)

    Z_all_rel_err = np.stack(Zrel_err)
    fig = go.Figure()
    for j in range(J):
        fig.add_trace(go.Box(
                y=Z_all_rel_err[:, j].flatten().tolist(), 
                showlegend=False, x=[j],                 
                boxpoints="outliers", line=dict(color="salmon")
                ))
    savename = "{}/popularity_error_Z_plot.html".format(datain)
    fix_plot_layout_and_save(fig, savename, xaxis_title="Popularity (most to least popular)", yaxis_title="Zj relative error", 
                            title="", showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
    
    alpha_all_rel_err = np.stack(alpharel_err)
    fig = go.Figure()
    for j in range(J):
        fig.add_trace(go.Box(
                y=alpha_all_rel_err[:, j].flatten().tolist(), 
                showlegend=False, x=[j],                 
                boxpoints="outliers", line=dict(color="salmon")
                ))
    savename = "{}/popularity_error_alpha_plot.html".format(datain)
    fix_plot_layout_and_save(fig, savename, xaxis_title="Popularity (most to least popular)", yaxis_title="alphaj relative error", 
                            title="", showgrid=False, showlegend=False, print_png=True, print_html=True, print_pdf=False)
    
    

def get_data_tempering_variance_combined_solution(parameter_names, M, d, K, J, DIR_base, 
                                                theta_true_per_m, param_positions_dict, 
                                                topdir="/tmp/", seedint=1234):
    

    estimation_sq_error_per_trial = dict()
    estimation_sq_error_per_trial_nonRT = dict()
    estimation_error_per_trial = dict()
    estimation_error_per_trial_nonRT = dict()  
    for param in parameter_names:   
        estimation_sq_error_per_trial[param] = []
        estimation_sq_error_per_trial_nonRT[param] = []
        estimation_error_per_trial[param] = []
        estimation_error_per_trial_nonRT[param] = []  
    
    for m in range(M):
        theta_true = theta_true_per_m[m]

        local_mse = dict()
        local_mse_nonrotated = dict()
        local_err = dict()
        local_err_nonrotated = dict()
        params_out = dict()
        params_out["X"] = np.zeros((d*K,))
        params_out["beta"] = np.zeros((K,))    
        for param in parameter_names:     
            local_mse[param] = []
            local_mse_nonrotated[param] = []
            local_err[param] = []
            local_err_nonrotated[param] = []      
            theta = None
            all_estimates = []
            path = pathlib.Path(DIR_base)  
            subdatasets_names = [file.name for file in path.iterdir() if not file.is_file()]                    
            for dataset_index in range(len(subdatasets_names)):                    
                subdataset_name = subdatasets_names[dataset_index]                        
                DIR_read = "{}/{}/".format(DIR_base, subdataset_name)
                path = pathlib.Path(DIR_read)  
                estimates_names = [file.name for file in path.iterdir() if file.is_file() and "_best" in file.name]
                if len(estimates_names) > 1:
                    raise AttributeError("Should have 1 output estimation file.")
                for estim in estimates_names:
                    with jsonlines.open("{}/{}".format(DIR_read, estim), mode="r") as f: 
                        for result in f.iter(type=dict, skip_invalid=True):
                            if param in ["X", "beta"]:
                                # single estimate per data split
                                theta = result[param]
                                namesplit = estim.split("_")
                                start = int(namesplit[5])
                                end   = int(namesplit[6].replace(".jsonl", ""))
                                if param == "X":
                                    params_out[param][start*d:end*d] = theta
                                    X_true = np.asarray(theta_true[param_positions_dict[param][0]+start*d:param_positions_dict[param][0]+end*d]).reshape((d, end-start), order="F")
                                    X_hat = np.asarray(theta).reshape((d, end-start), order="F")
                                    Rx, tx, mse, mse_nonrotated, err, err_nonrotated = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)
                                else:
                                    params_out[param][start:end] = theta 
                                    rel_err = (theta_true[param_positions_dict[param][0]+start:param_positions_dict[param][0]+end] - theta)/theta_true[param_positions_dict[param][0]+start:param_positions_dict[param][0]+end]                      
                                    err = np.mean(rel_err)
                                    mse = np.mean(rel_err**2)
                            else:                                                        
                                theta = result[param]
                                all_estimates.append(theta)
                                if param == "Z":
                                    Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")  
                                    Z_hat = np.asarray(theta).reshape((d, J), order="F")
                                    Rz, tz, mse, mse_nonrotated, err, err_nonrotated = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)
                                else:
                                    rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - theta)/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                                    err = rel_err
                                    mse = np.mean(rel_err**2)
                
                local_mse[param].append(mse)
                if param in ["X", "Z", "Phi"]:
                    local_mse_nonrotated[param].append(mse_nonrotated)
                else: 
                    # for parameters that are not rotated in any case, just add the same estimate
                    local_mse_nonrotated[param].append(mse)

                local_err[param].append(err)
                if param in ["X", "Z", "Phi"]:
                    local_err_nonrotated[param].append(err_nonrotated)
                else: 
                    # for parameters that are not rotated in any case, just add the same estimate
                    local_err_nonrotated[param].append(err)

            if param in ["X", "beta"]:
                params_out[param] = params_out[param].tolist()       
            else:             
                all_estimates = np.stack(all_estimates)
                if param not in ["Z", "Phi", "alpha"]:
                    all_estimates = all_estimates.flatten()
                # compute variance over columns
                column_variances = np.var(all_estimates, axis=0)
                # sum acrocs each coordinate's weight
                all_weights_sum = np.sum(column_variances, axis=0)
                all_weights_norm = column_variances/all_weights_sum
                assert np.allclose(np.sum(all_weights_norm, axis=0), np.ones(all_weights_sum.shape))
                # element-wise multiplication
                weighted_estimate = np.sum(all_weights_norm*all_estimates, axis=0)
                params_out[param] = weighted_estimate.tolist()
        

        for param in parameter_names:
            if param == "X":                 
                X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                X_hat = np.asarray(params_out[param]).reshape((d, K), order="F")
                Rx, tx, mse_x, mse_x_nonRT, err_x, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seedint)

                estimation_sq_error_per_trial[param].append(mse_x)
                estimation_sq_error_per_trial_nonRT[param].append(mse_x_nonRT)
                estimation_error_per_trial[param].append(err_x)
                estimation_error_per_trial_nonRT[param].append(err_x_nonRT)
                params_out["mse_x_RT"] = mse_x
                params_out["mse_x_nonRT"] = mse_x_nonRT
                params_out["err_x_RT"] = err_x
                params_out["err_x_nonRT"] = err_x_nonRT
            elif param == "Z":
                Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                Z_hat = np.asarray(params_out[param]).reshape((d, J), order="F")
                Rz, tz, mse_z, mse_z_nonRT, err_z, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seedint)

                estimation_sq_error_per_trial[param].append(mse_z)
                estimation_sq_error_per_trial_nonRT[param].append(mse_z_nonRT)
                estimation_error_per_trial[param].append(err_z)
                estimation_error_per_trial_nonRT[param].append(err_z_nonRT)  
                params_out["mse_z_RT"] = mse_z
                params_out["mse_z_nonRT"] = mse_z_nonRT
                params_out["err_z_RT"] = err_z
                params_out["err_z_nonRT"] = err_z_nonRT               
            elif param == "Phi":            
                Phi_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                Phi_hat = np.asarray(params_out[param]).reshape((d, J), order="F")
                Rphi, tphi, mse_phi, mse_phi_nonRT, err_phi, err_phi_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Phi_true, param_hat=Phi_hat, seedint=seedint)

                estimation_sq_error_per_trial[param].append(mse_phi)    
                estimation_sq_error_per_trial_nonRT[param].append(mse_phi_nonRT)
                estimation_error_per_trial[param].append(err_phi)
                estimation_error_per_trial_nonRT[param].append(err_phi_nonRT)                    
            elif param in ["beta", "alpha"]:
                rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]     
                mse = np.mean(rel_err**2)  

                estimation_sq_error_per_trial[param].append(mse)    
                estimation_sq_error_per_trial_nonRT[param].append(mse)
                estimation_error_per_trial[param].append(np.mean(rel_err))
                estimation_error_per_trial_nonRT[param].append(np.mean(rel_err))      

                params_out["mse_{}".format(param)] = mse
                params_out["rel_err_{}".format(param)] = np.mean(rel_err)
            else:
                rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                mse = rel_err**2

                estimation_sq_error_per_trial[param].append(mse)    
                estimation_sq_error_per_trial_nonRT[param].append(mse)
                estimation_error_per_trial[param].append(rel_err)
                estimation_error_per_trial_nonRT[param].append(rel_err)      

                params_out["mse_{}".format(param)] = mse[0]
                params_out["rel_err_{}".format(param)] = rel_err[0]


        # out_file = "{}/efficiency_metrics.jsonl".format(DIR_out)
        # with jsonlines.open(out_file, mode='r') as f:         
        #     for result in f.iter(type=dict, skip_invalid=True):
        #         wall_duration = result["wall_duration"]
        #         avg_total_cpu_util = result["avg_total_cpu_util"]
        #         max_total_cpu_util = result["max_total_cpu_util"]
        #         avg_total_ram_residentsetsize_MB = result["avg_total_ram_residentsetsize_MB"]
        #         max_total_ram_residentsetsize_MB = result["max_total_ram_residentsetsize_MB"]
        #         avg_threads = result["avg_threads"]
        #         max_threads = result["max_threads"]
        #         avg_processes = result["avg_processes"]
        #         ax_processes = result["max_processes"]
        
        out_file = "{}/params_out_global_theta_hat.jsonl".format(DIR_base)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write(params_out)
    
    fig_sq_err = go.Figure()
    fig_sq_err_nonRT = go.Figure()
    fig_err = go.Figure()
    fig_err_nonRT = go.Figure()
    pathlib.Path("{}/icm_data_tempering_estimation_plots/".format(topdir)).mkdir(parents=True, exist_ok=True)     
    for param in parameter_names:
        # df = pd.DataFrame.from_dict({"vals":estimation_error_per_trial[param]})
        # lower_bound = df.vals.quantile(0.05)
        # upper_bound = df.vals.quantile(0.95)
        # df_filtered = df.vals[(df.vals >= lower_bound) & (df.vals <= upper_bound)].dropna()
        fig_sq_err.add_trace(go.Box(
                        y=np.asarray(estimation_sq_error_per_trial[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_sq_error_per_trial[param]), boxpoints='outliers'                                
                    ))
        fig_sq_err_nonRT.add_trace(go.Box(
                        y=np.asarray(estimation_sq_error_per_trial_nonRT[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_sq_error_per_trial_nonRT[param]), boxpoints='outliers'                                
                    ))
        fig_err.add_trace(go.Box(
                        y=np.asarray(estimation_error_per_trial[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_error_per_trial[param]), boxpoints='outliers'                                
                    ))
        fig_err_nonRT.add_trace(go.Box(
                        y=np.asarray(estimation_error_per_trial_nonRT[param]).tolist(), showlegend=True, name=param,
                        x=[param]*len(estimation_error_per_trial_nonRT[param]), boxpoints='outliers'                                
                    ))
    savename = "{}/icm_data_tempering_estimation_plots/mse_overAllTrials_perparam_weighted_boxplot.html".format(topdir)
    fix_plot_layout_and_save(fig_sq_err, savename, xaxis_title="", yaxis_title="Mean relative Sq. Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)
    savename = "{}/icm_data_tempering_estimation_plots/mse_nonRT_overAllTrials_perparam_weighted_boxplot.html".format(topdir)
    fix_plot_layout_and_save(fig_sq_err_nonRT, savename, xaxis_title="", yaxis_title="Mean relative Sq. Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)
    savename = "{}/icm_data_tempering_estimation_plots/err_overAllTrials_perparam_weighted_boxplot.html".format(topdir)
    fix_plot_layout_and_save(fig_err, savename, xaxis_title="", yaxis_title="Mean relative Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)
    savename = "{}/icm_data_tempering_estimation_plots/err_nonRT_overAllTrials_perparam_weighted_boxplot.html".format(topdir)
    fix_plot_layout_and_save(fig_err_nonRT, savename, xaxis_title="", yaxis_title="Mean relative Err Θ", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=True, print_html=True, 
                            print_pdf=False)      
        
    return params_out, local_mse, local_mse_nonrotated, local_err, local_err_nonrotated



class TruncatedInverseGamma:
    """
    Implementation of a truncated Inverse Gamma distribution.
    The distribution is truncated to the interval [lower, upper].
    
    Parameters:
    -----------
    alpha : float
        Shape parameter of the inverse gamma distribution
    beta : float
        Scale parameter of the inverse gamma distribution
    lower : float
        Lower bound of the truncation interval
    upper : float
        Upper bound of the truncation interval
    """
    
    def __init__(self, alpha, beta, lower, upper):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        if lower >= upper:
            raise ValueError("lower bound must be less than upper bound")
        if lower <= 0:
            raise ValueError("lower bound must be positive")
            
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        
        # Calculate normalization constant
        self.norm_constant = self._get_normalization_constant()
        self.log_norm_constant = np.log(self.norm_constant)
        
    def _get_normalization_constant(self):
        """Calculate the normalization constant for truncation."""
        # Integrate the inverse gamma from lower to upper
        lower_cdf = invgamma.cdf(self.lower, a=self.alpha, loc=0, scale=self.beta)
        upper_cdf = invgamma.cdf(self.upper, a=self.alpha, loc=0, scale=self.beta)
        return upper_cdf - lower_cdf
    
    def pdf(self, x):
        """
        Probability density function of the truncated inverse gamma.
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the PDF
            
        Returns:
        --------
        array_like
            PDF values at x
        """
        x = np.asarray(x)
        # Return 0 for values outside the truncation interval
        pdf = np.zeros_like(x, dtype=float)
        mask = (x >= self.lower) & (x <= self.upper)
        
        # Calculate PDF for values within bounds
        pdf[mask] = invgamma.pdf(x[mask], a=self.alpha, loc=0, scale=self.beta) / self.norm_constant
        return pdf

    def logcdf(self, x):
        """
        Log of the cumulative distribution function of the truncated inverse gamma.
        Computed in a numerically stable way.
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the log CDF
            
        Returns:
        --------
        array_like
            Log CDF values at x
        """
        x = np.asarray(x)
        logcdf = np.full_like(x, -np.inf, dtype=float)
        
        # For x < lower, logcdf is -inf
        # For x > upper, logcdf is 0
        upper_mask = x > self.upper
        logcdf[upper_mask] = 0.0
        
        # For values in truncation interval
        mask = (x >= self.lower) & (x <= self.upper)
        if np.any(mask):
            # Compute unnormalized CDF using the complementary incomplete gamma function
            # P(X ≤ x) = 1 - P(X > x) = 1 - gammaincc(alpha, beta/x)
            z = self.beta / x[mask]
            unnorm_cdf = -np.log1p(-gammaincc(self.alpha, z))
            
            # Normalize by the truncation interval
            lower_cdf = invgamma.cdf(self.lower, a=self.alpha, loc=0, scale=self.beta)
            logcdf[mask] = unnorm_cdf - np.log(self.norm_constant)
        
        return logcdf
    
    def rvs(self, size=1, random_state=None):
        """
        Random variates from the truncated inverse gamma distribution.
        
        Parameters:
        -----------
        size : int
            Number of random variates to generate
        random_state : int or RandomState, optional
            Random state for reproducibility
            
        Returns:
        --------
        array_like
            Random variates from the truncated distribution
        """
        rng = np.random.RandomState(random_state)
        
        # Use rejection sampling
        samples = []
        while len(samples) < size:
            # Sample from the original inverse gamma
            candidate = invgamma.rvs(self.alpha, loc=0, scale=self.beta, 
                                   size=size, random_state=rng)
            
            # Accept samples that fall within bounds
            valid_samples = candidate[(candidate >= self.lower) & 
                                   (candidate <= self.upper)]
            samples.extend(valid_samples[:size - len(samples)])
            
        return np.array(samples[:size])
    
    def logpdf(self, x):
        """
        Log probability density function of the truncated inverse gamma.
        This method is numerically more stable than taking the log of pdf().
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the log PDF
            
        Returns:
        --------
        array_like
            Log PDF values at x
        """
        x = np.asarray(x)
        # Initialize with -inf for values outside the truncation interval
        logpdf = np.full_like(x, -np.inf, dtype=float)
        mask = (x >= self.lower) & (x <= self.upper)
        
        # Calculate logpdf for values within bounds
        # log(pdf) = log(beta^alpha) - log(Gamma(alpha)) - (alpha + 1)log(x) - beta/x - log(norm_constant)
        valid_x = x[mask]
        if len(valid_x) > 0:
            logpdf[mask] = (self.alpha * np.log(self.beta) - 
                            gammaln(self.alpha) -
                           (self.alpha + 1) * np.log(valid_x) -
                           self.beta / valid_x -
                           self.log_norm_constant)
        
        return logpdf

def log_full_posterior(Y, theta_curr, param_positions_dict, args):

    try:
        DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol, \
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, \
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, \
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, \
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true = args
    except:
        DIR_out, data_location, subdataset_name, dataset_index, optimisation_method,\
            parameter_names, J, K, d, N, dst_func, niter, _, m, penalty_weight_Z,\
                constant_Z, retries, parallel, min_sigma_e, prior_loc_x, prior_scale_x, prior_loc_z,\
                    prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha,\
                        prior_scale_alpha, prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta,\
                            prior_loc_sigmae, prior_scale_sigmae, _, rng, batchsize, theta_true = args

    # if False:
    if K*J <= 10e5:
        loglik = -negative_loglik(theta_curr, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False)
    else:
        loglik = -negative_loglik_parallel(theta_curr, Y, J, K, d, parameter_names, dst_func, param_positions_dict, penalty_weight_Z, constant_Z, debug=False)
            
    params_hat = optimisation_dict2params(theta_curr, param_positions_dict, J, K, d, parameter_names)

    full_posterior = 0.0
    full_posterior += loglik
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")
    full_posterior += multivariate_normal.logpdf(X.T, mean=prior_loc_x, cov=prior_scale_x).sum()
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F") 
    full_posterior += multivariate_normal.logpdf(Z.T, mean=prior_loc_z, cov=prior_scale_z).sum()
    if "Phi" in parameter_names:
        Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F") 
        full_posterior += multivariate_normal.logpdf(Phi.T, mean=prior_loc_phi, cov=prior_scale_phi).sum()
    alpha = params_hat["alpha"]
    full_posterior += norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha).sum()
    beta = params_hat["beta"]
    full_posterior += norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta).sum()
    gamma = params_hat["gamma"]
    full_posterior += norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma)
    if "delta" in parameter_names:
        delta = params_hat["delta"]
        full_posterior += norm.logpdf(delta, loc=prior_loc_delta, scale=prior_scale_delta)
    sigma_e = params_hat["sigma_e"]
    tig = TruncatedInverseGamma(alpha=prior_loc_sigmae, beta=prior_scale_sigmae, lower=min_sigma_e, upper=10*np.sqrt(prior_scale_sigmae)+prior_scale_sigmae) 
    full_posterior += tig.logpdf(sigma_e)

    return loglik, full_posterior

def log_conditional_posterior_x_vec(xi, i, Y, theta, J, K, d, parameter_names, dst_func, 
                                param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1, debug=False, numbafast=True):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[:, i] = xi
    theta_test = theta.copy()
    theta_test[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpx_i = 0   
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)                  
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)        
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)        
            _logpx_i += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(xi, mean=prior_loc_x, cov=prior_scale_x)
    
    if numbafast:
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        betai = params_hat["beta"][i]
        pijs = p_i_arg_numbafast(X[:, i], Z, alpha, betai, gamma)
    else:
        pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict) 
    
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)
    log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)   
    if debug:
        log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
        assert np.allclose(log1mcdfs, log1mcdfsbase)       
    
    logpx_i = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(xi, mean=prior_loc_x, cov=prior_scale_x))
    if debug:
        assert(np.allclose(logpx_i, _logpx_i))
    
    return logpx_i*gamma

def log_conditional_posterior_x_il(x_il, l, i, Y, theta, J, K, d, parameter_names, dst_func, 
                                param_positions_dict, prior_loc_x=0, prior_scale_x=1, gamma=1, 
                                debug=False, numbafast=True):
    # l denotes the coordinate of vector x_i

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    X = np.asarray(params_hat["X"]).reshape((d, K), order="F")                         
    X[l, i] = x_il
    theta_test = theta.copy()
    theta_test[param_positions_dict["X"][0]:param_positions_dict["X"][1]] = X.reshape((d*K,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpx_il = 0
        for j in range(J):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpx_il += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(x_il, loc=prior_loc_x, scale=prior_scale_x)
    
    if numbafast:
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        betai = params_hat["beta"][i]
        pijs = p_i_arg_numbafast(X[:, i], Z, alpha, betai, gamma)
    else:
        pijs = p_ij_arg(i, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)      
    
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)   
    if debug:
        log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
        assert np.allclose(log1mcdfs, log1mcdfsbase)       

    logpx_il = np.sum(Y[i, :]*logcdfs + (1-Y[i, :])*log1mcdfs + multivariate_normal.logpdf(x_il, mean=prior_loc_x, cov=prior_scale_x))
    if debug:
        assert(np.allclose(logpx_il, _logpx_il))
             
    return logpx_il*gamma

def log_conditional_posterior_phi_vec(phi_j, j, Y, theta, J, K, d, parameter_names, dst_func, 
                                      param_positions_dict, prior_loc_phi=0, prior_scale_phi=1, 
                                      gamma=1, debug=False, numbafast=True, block_size_rows=10):
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")                         
    Phi[:, j] = phi_j
    theta_test = theta.copy()
    theta_test[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpphi_j = 0
        for i in range(K):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpphi_j += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(phi_j, mean=prior_loc_phi, cov=prior_scale_phi)

    pijs = p_ij_arg(None, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpphi_j = np.sum(Y[:, j]*logcdfs + (1-Y[:, j])*log1mcdfs + multivariate_normal.logpdf(phi_j, mean=prior_loc_phi, cov=prior_scale_phi))
    if debug:
        assert(np.allclose(logpphi_j, _logpphi_j))
             
    return logpphi_j*gamma

def log_conditional_posterior_phi_jl(phi_jl, l, j, Y, theta, J, K, d, parameter_names, dst_func, 
                                    param_positions_dict, prior_loc_phi=0, prior_scale_phi=1, gamma=1, 
                                    debug=False, numbafast=True, block_size_rows=10):
    # l denotes the coordinate of vector phi_j

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Phi = np.asarray(params_hat["Phi"]).reshape((d, J), order="F")                         
    Phi[l, j] = phi_jl
    theta_test = theta.copy()
    theta_test[param_positions_dict["Phi"][0]:param_positions_dict["Phi"][1]] = Phi.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpphi_jl = 0
        for i in range(K):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpphi_jl += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(phi_jl, loc=prior_loc_phi, scale=prior_scale_phi)
    
    pijs = p_ij_arg(None, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
    logpphi_jl = np.sum(Y[:, j]*logcdfs + (1-Y[:, j])*log1mcdfs + multivariate_normal.logpdf(phi_jl, mean=prior_loc_phi, cov=prior_scale_phi))
    if debug:
        assert(np.allclose(logpphi_jl, _logpphi_jl))

    return logpphi_jl*gamma

def log_conditional_posterior_z_vec(zj, j, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z=0, 
                                    prior_scale_z=1, gamma=1, constant_Z=0, penalty_weight_Z=100, 
                                    debug=False, numbafast=True, block_size_rows=500):
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         
    Z[:, j] = zj
    theta_test = theta.copy()
    theta_test[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpz_j = 0
        for i in range(K):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpz_j += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + multivariate_normal.logpdf(zj, mean=prior_loc_z, cov=prior_scale_z)
    
    if K*J <= 10e5:
        if numbafast:
            X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
            gamma = params_hat["gamma"][0]
            alphaj = params_hat["alpha"][j]
            beta = params_hat["beta"]
            pijs = p_j_arg_numbafast(X, Z[:, j], alphaj, beta, gamma)
        else:
            pijs = p_ij_arg(None, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)              
        logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
        log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
        logpz_j = np.sum(Y[:, j]*logcdfs + (1-Y[:, j])*log1mcdfs + multivariate_normal.logpdf(zj, mean=prior_loc_z, cov=prior_scale_z))
        if debug:
            log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
            assert np.allclose(log1mcdfs, log1mcdfsbase)
            assert(np.allclose(logpz_j, _logpz_j))
    else:
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        gamma = params_hat["gamma"][0]
        beta = params_hat["beta"]
        row_blocks = (K + block_size_rows - 1) // block_size_rows
        block_size_cols = j
        col_blocks = -1
        logpz_j = process_blocks_parallel(Y, X, Z, params_hat["alpha"], 
                                            beta, gamma, K, J, mu_e, sigma_e, 
                                            row_blocks, col_blocks, block_size_rows, block_size_cols, 
                                            prior=multivariate_normal.logpdf(zj, mean=prior_loc_z, cov=prior_scale_z))
        if debug:
            assert(np.allclose(logpz_j, _logpz_j))

    # if abs(penalty_weight_Z) > 1e-10:
    #     sum_Z_J_vectors = np.sum(Z, axis=1)    
    #     obj = logpz_j + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)
    # else:
    #     obj = logpz_j
    
    return logpz_j*gamma

def log_conditional_posterior_z_jl(z_jl, l, j, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, prior_loc_z=0, 
                                   prior_scale_z=1, gamma=1, constant_Z=0, penalty_weight_Z=100, 
                                   debug=False, numbafast=True, block_size_rows=500):
    # l denotes the coordinate of vector z_j
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                         
    Z[l, j] = z_jl
    theta_test = theta.copy()
    theta_test[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]] = Z.reshape((d*J,), order="F")
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    if debug:
        _logpz_jl = 0
        for i in range(K):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpz_jl += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(z_jl, loc=prior_loc_z, scale=prior_scale_z)

    # if False:
    if K*J <= 10e5:
        if numbafast:
            X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
            gamma = params_hat["gamma"][0]
            alphaj = params_hat["alpha"][j]
            beta = params_hat["beta"]
            pijs = p_j_arg_numbafast(X, Z[:, j], alphaj, beta, gamma)
            # _pij_arg = p_ij_arg_numbafast(X, Z[:, j], alphaj, beta, gamma, K)  
        else:
            pijs = p_ij_arg(None, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
        logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)    
        log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)   
        logpz_jl = np.sum(Y[:, j]*logcdfs + (1-Y[:, j])*log1mcdfs + multivariate_normal.logpdf(z_jl, mean=prior_loc_z, cov=prior_scale_z))
        if debug:
            log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
            assert np.allclose(log1mcdfs, log1mcdfsbase)       
            assert(np.allclose(logpz_jl, _logpz_jl))    

        # print(pijs[0])
        # pijs = p_ij_arg(None, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
        # print(pijs[0])
        

    else:
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        gamma = params_hat["gamma"][0]
        beta = params_hat["beta"]
        row_blocks = (K + block_size_rows - 1) // block_size_rows
        block_size_cols = j
        col_blocks = -1
        logpz_jl = process_blocks_parallel(Y, X, Z, params_hat["alpha"], 
                                            beta, gamma, K, J, mu_e, sigma_e, row_blocks, col_blocks, 
                                            block_size_rows, block_size_cols, 
                                            prior=multivariate_normal.logpdf(z_jl, mean=prior_loc_z, cov=prior_scale_z))
        if debug:
            assert(np.allclose(logpz_jl, _logpz_jl))

    # if abs(penalty_weight_Z) > 1e-10:
    #     sum_Z_J_vectors = np.sum(Z, axis=1)    
    #     obj = logpz_jl + penalty_weight_Z * np.sum((sum_Z_J_vectors-np.asarray([constant_Z]*d))**2)
    # else:
    #     obj = logpz_jl

    return logpz_jl*gamma

def log_conditional_posterior_alpha_j(alpha, idx, Y, theta, J, K, d, parameter_names, dst_func, 
                                    param_positions_dict, prior_loc_alpha=0, prior_scale_alpha=1, gamma=1, 
                                    debug=False, numbafast=True, block_size_rows=500):    
    
    # Assuming independent, Gaussian alphas.
    # Hence, even when evaluating with vector parameters, we use the uni-dimensional posterior for alpha.
    # Only computes the part of the posterior that impacts optimisation.

    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()    
    theta_test[param_positions_dict["alpha"][0] + idx] = alpha     
    if debug:        
        _logpalpha_j = 0
        # for j in range(J):        
        j = idx
        for i in range(K):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)            
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpalpha_j += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha)

    # if False:
    if K*J <= 10e5:
        if numbafast:
            params_hat = optimisation_dict2params(theta_test, param_positions_dict, J, K, d, parameter_names)
            X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
            Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
            gamma = params_hat["gamma"][0]        
            beta_nbfast = params_hat["beta"]            
            pijs = p_j_arg_numbafast(X, Z[:, idx], alpha, beta_nbfast, gamma)
            # alpha_nbfast = params_hat["alpha"]
            # pijs = p_ij_arg_numbafast(X, Z, alpha_nbfast, beta_nbfast, gamma, K)  
            # assert(np.allclose(pijs[:, idx], p_ij_arg_numbafast(X, Z[:, idx].reshape((d, 1)), alpha, beta_nbfast, gamma, K).flatten()))        
        else:
            pijs = p_ij_arg(None, idx, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)      
        logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)   
        log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e) 
        logpalpha_j = np.sum(Y[:, idx]*logcdfs + (1-Y[:, idx])*log1mcdfs + norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha))
        if debug:     
            log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)
            assert np.allclose(log1mcdfs, log1mcdfsbase)
            assert np.allclose(logpalpha_j, _logpalpha_j)
    else:
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
        gamma = params_hat["gamma"][0]        
        row_blocks = (K + block_size_rows - 1) // block_size_rows
        block_size_cols = idx
        col_blocks = -1    
        logpalpha_j = process_blocks_parallel(Y, X, Z, theta_test[param_positions_dict["alpha"][0]:param_positions_dict["alpha"][1]].copy(), 
                                                params_hat["beta"], gamma, K, J, mu_e, sigma_e, 
                                                row_blocks, col_blocks, block_size_rows, block_size_cols, 
                                                prior=norm.logpdf(alpha, loc=prior_loc_alpha, scale=prior_scale_alpha))
        if debug:
            assert(np.allclose(logpalpha_j, _logpalpha_j))
    
    return logpalpha_j*gamma

def log_conditional_posterior_beta_i(beta, idx, Y, theta, J, K, d, parameter_names, dst_func, 
                            param_positions_dict, prior_loc_beta=0, prior_scale_beta=1, gamma=1, 
                            debug=False, numbafast=True):
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()
    theta_test[param_positions_dict["beta"][0] + idx] = beta
    if debug:
        _logpbeta_k = 0
        i = idx
        for j in range(J):            
            # for i in range(K):
            pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)
            philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
            log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
            _logpbeta_k += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta)

    if numbafast:
        params_hat = optimisation_dict2params(theta_test, param_positions_dict, J, K, d, parameter_names)
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
        gamma = params_hat["gamma"][0]
        alpha_nbfast = params_hat["alpha"]        
        pijs = p_i_arg_numbafast(X[:, idx], Z, alpha_nbfast, beta, gamma)
        # beta_nbfast = params_hat["beta"]
        # pijs = p_ij_arg_numbafast(X, Z, alpha_nbfast, beta_nbfast, gamma, K)     
        # assert(np.allclose(pijs[idx, :], p_i_arg_numbafast(X[:, idx], Z, alpha_nbfast, beta, gamma)))        
    else:
        pijs = p_ij_arg(idx, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)
        
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)   
    if debug:
        log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
        assert np.allclose(log1mcdfs, log1mcdfsbase)

    logpbeta_k = np.sum(Y[idx, :]*logcdfs + (1-Y[idx, :])*log1mcdfs + norm.logpdf(beta, loc=prior_loc_beta, scale=prior_scale_beta))
    if debug:
        assert(np.allclose(logpbeta_k, _logpbeta_k))
    
    return logpbeta_k*gamma

def log_conditional_posterior_gamma(gamma, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma_annealing=1, prior_loc_gamma=0, 
                                    prior_scale_gamma=1, debug=False, numbafast=True, block_size_rows=500, block_size_cols=100):    
        
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()
    theta_test[param_positions_dict["gamma"][0]] = gamma
    if debug:
        _logpgamma = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpgamma += Y[i, j]*philogcdf + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma)

    # if False:
    if K*J <= 10e5:
        if numbafast:            
            X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
            Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")                 
            alpha = params_hat["alpha"]
            beta = params_hat["beta"]
            pijs = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K)     
        else:
            pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)        
        logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
        log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)       
        logpgamma = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma))
        if debug:
            log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
            assert np.allclose(log1mcdfs, log1mcdfsbase)
            assert(np.allclose(logpgamma, _logpgamma))
    else:
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")             
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        row_blocks = (K + block_size_rows - 1) // block_size_rows
        col_blocks = (J + block_size_cols - 1) // block_size_cols    
        logpgamma = process_blocks_parallel(Y, X, Z, alpha, beta, gamma, K, J, mu_e, sigma_e, 
                                    row_blocks, col_blocks, block_size_rows, block_size_cols, 
                                    prior=norm.logpdf(gamma, loc=prior_loc_gamma, scale=prior_scale_gamma))
        if debug:
            assert(np.allclose(logpgamma, _logpgamma))

    return logpgamma*gamma_annealing

def log_conditional_posterior_delta(delta, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, 
                                    prior_loc_delta=0, prior_scale_delta=1, debug=False):    
    
    params_hat = optimisation_dict2params(theta, param_positions_dict, J, K, d, parameter_names)
    mu_e = 0
    sigma_e = params_hat["sigma_e"]
    theta_test = theta.copy()
    theta_test[param_positions_dict["delta"][0]] = delta
    if debug:
        _logpdelta = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)           
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpdelta += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf + norm.logpdf(delta, loc=prior_loc_delta, scale=prior_scale_delta)

    pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)    
    logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
    log1mcdfs = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
    logpdelta = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + norm.logpdf(delta, loc=prior_loc_delta, scale=prior_scale_delta))   
    if debug:
        assert(np.allclose(logpdelta, _logpdelta))     
        
    return logpdelta**gamma

def log_conditional_posterior_sigma_e(sigma_e, Y, theta, J, K, d, parameter_names, dst_func, param_positions_dict, gamma=1, 
                                    prior_loc_sigmae=0, prior_scale_sigmae=1, min_sigma_e=0.0001, 
                                    debug=False, numbafast=True, block_size_rows=500, block_size_cols=100):    
    
    tig = TruncatedInverseGamma(alpha=prior_loc_sigmae, beta=prior_scale_sigmae, lower=min_sigma_e, upper=10*np.sqrt(prior_scale_sigmae)+prior_scale_sigmae)    
    mu_e = 0
    theta_test = theta.copy()
    theta_test[param_positions_dict["sigma_e"][0]] = sigma_e
    if debug:
        _logpsigma_e = 0
        for j in range(J):
            for i in range(K):
                pij = p_ij_arg(i, j, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)            
                philogcdf = norm.logcdf(pij, loc=mu_e, scale=sigma_e)
                log_one_minus_cdf = log_complement_from_log_cdf(philogcdf, pij, mean=mu_e, variance=sigma_e)
                _logpsigma_e += Y[i, j]*philogcdf  + (1-Y[i, j])*log_one_minus_cdf + tig.logpdf(sigma_e)
    
    # if False:
    if K*J <= 10e5:
        if numbafast:
            params_hat = optimisation_dict2params(theta_test, param_positions_dict, J, K, d, parameter_names)
            X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
            Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
            gamma = params_hat["gamma"][0]
            alpha = params_hat["alpha"]
            beta = params_hat["beta"]
            pijs = p_ij_arg_numbafast(X, Z, alpha, beta, gamma, K)     
        else:
            pijs = p_ij_arg(None, None, theta_test, J, K, d, parameter_names, dst_func, param_positions_dict)    
        
        logcdfs = norm.logcdf(pijs, loc=mu_e, scale=sigma_e)        
        log1mcdfs = log_complement_from_log_cdf_vec_fast(logcdfs, pijs, mean=mu_e, variance=sigma_e)
        logpsigma_e = np.sum(Y*logcdfs + (1-Y)*log1mcdfs + tig.logpdf(sigma_e))   
        if debug:
            log1mcdfsbase = log_complement_from_log_cdf_vec(logcdfs, pijs, mean=mu_e, variance=sigma_e)    
            assert np.allclose(log1mcdfs, log1mcdfsbase)
            assert(np.allclose(logpsigma_e, _logpsigma_e)) 
    else:
        params_hat = optimisation_dict2params(theta_test, param_positions_dict, J, K, d, parameter_names)
        X = np.asarray(params_hat["X"]).reshape((d, K), order="F")     
        Z = np.asarray(params_hat["Z"]).reshape((d, J), order="F")     
        gamma = params_hat["gamma"][0]
        alpha = params_hat["alpha"]
        beta = params_hat["beta"]
        row_blocks = (K + block_size_rows - 1) // block_size_rows
        col_blocks = (J + block_size_cols - 1) // block_size_cols    
        logpsigma_e = process_blocks_parallel(Y, X, Z, alpha, beta, gamma, K, J, mu_e, sigma_e, 
                                                row_blocks, col_blocks, block_size_rows, 
                                                block_size_cols, prior=tig.logpdf(sigma_e))
        if debug:
            assert(np.allclose(logpsigma_e, _logpsigma_e))
        
    return logpsigma_e*gamma

####################### ICM #############################



####################### CA #############################

def clean_up_data_matrix(Y, K, J, d, theta_true, parameter_names, param_positions_dict):

    # uninformative users 
    k_idx = np.argwhere(np.all(Y<1, axis=1))
    # uninformative lead users
    j_idx = np.argwhere(np.all(Y<1, axis=0))

    Y_new = Y[:, ~np.all(Y < 1, axis=0)]
    Y_new = Y_new[~np.all(Y_new < 1, axis=1), :]

    K_new = K - len(k_idx.flatten())
    J_new = J - len(j_idx.flatten())

    parameter_space_dim_new = (K_new+J_new)*d + J_new + K_new + 2
    param_positions_dict_new = dict()            
    k = 0
    for param in parameter_names:
        if param == "X":
            param_positions_dict_new[param] = (k, k + K_new*d)                       
            k += K_new*d    
        elif param in ["Z"]:
            param_positions_dict_new[param] = (k, k + J_new*d)                                
            k += J_new*d
        elif param in ["Phi"]:            
            param_positions_dict_new[param] = (k, k + J_new*d)                                
            k += J_new*d
        elif param == "beta":
            param_positions_dict_new[param] = (k, k + K_new)                                   
            k += K_new
        elif param == "alpha":
            param_positions_dict_new[param] = (k, k + J_new)                                       
            k += J_new
        elif param == "gamma":
            param_positions_dict_new[param] = (k, k + 1)                                
            k += 1
        elif param == "delta":
            param_positions_dict_new[param] = (k, k + 1)                                
            k += 1
        elif param == "sigma_e":
            param_positions_dict_new[param] = (k, k + 1)                                
            k += 1

    theta_true_new = theta_true.tolist().copy()
    for kidx in k_idx.flatten().tolist():
        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=kidx, d=d)
        # assert target_param == "X"
        del theta_true_new[param_positions_dict["X"][0]+kidx*d:param_positions_dict["X"][0]+(kidx+1)*d]
        del theta_true_new[param_positions_dict["beta"][0]-d+kidx:param_positions_dict["beta"][0]-d+kidx+1] # due to removal of X vector, shift all indices d positions to the left
    for jidx in j_idx.flatten().tolist():
        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=jidx, d=d)
        # assert target_param == "Z"
        del theta_true_new[param_positions_dict["Z"][0]+jidx*d:param_positions_dict["Z"][0]+(jidx+1)*d]
        if "Phi" in parameter_names:
            del theta_true_new[param_positions_dict["Phi"][0]+jidx*d:param_positions_dict["Phi"][0]+(jidx+1)*d]
        if "Phi" in parameter_names:            
            del theta_true_new[param_positions_dict["alpha"][0]-2*d+jidx:param_positions_dict["alpha"][0]-2*d+jidx+1]
        else:
            del theta_true_new[param_positions_dict["alpha"][0]-d+jidx:param_positions_dict["alpha"][0]-d+jidx+1]

    print("Dropped {} users, {} lead users, new parameter space size: {}.".format(K-K_new, J-J_new, parameter_space_dim_new))

    i = 0
    ii = 0
    theta_true = theta_true.tolist()
    while i < (K+J)*d + J + K + 2:
        target_param, vector_index_in_param_matrix, vector_coordinate = get_parameter_name_and_vector_coordinate(param_positions_dict, i=i, d=d)
        if vector_index_in_param_matrix in k_idx.flatten().tolist():
            i += 1            
        elif vector_index_in_param_matrix in j_idx.flatten().tolist():
            i += 1
        else:
            assert np.allclose(theta_true[i], theta_true_new[ii])
            i += 1
            ii += 1
       

    return Y_new, K_new, J_new, np.asarray(theta_true_new), param_positions_dict_new, parameter_space_dim_new

####################### CA #############################