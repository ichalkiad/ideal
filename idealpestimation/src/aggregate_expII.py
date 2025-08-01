import sys
import ipdb
import pathlib
import pickle
import numpy as np
import random
import math
import pandas as pd
from idealpestimation.src.utils import time, timedelta, fix_plot_layout_and_save, rank_and_plot_solutions, \
                                                            print_threadpool_info, jsonlines, combine_estimate_variance_rule, \
                                                                optimisation_dict2params, clean_up_data_matrix, get_data_tempering_variance_combined_solution,\
                                                                get_min_achievable_mse_under_rotation_trnsl, go
from plotly.subplots import make_subplots



def huber_weighted_mean(x, delta=1.345, tol=1e-6, max_iter=100):
    """
    Compute the Huber weighted mean of a 1D numpy array `x`.
    
    Parameters:
        x (array-like): 1D array of values.
        delta (float): Huber threshold (in units of MAD). Default is 1.345.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: The robust Huber-weighted mean.
    """
    x = np.asarray(x)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = delta * mad if mad > 0 else delta * np.std(x)  # fallback if MAD = 0

    mu = median
    for _ in range(max_iter):
        residuals = x - mu
        abs_res = np.abs(residuals)
        weights = np.where(abs_res <= scale, 1.0, scale / abs_res)
        mu_new = np.sum(weights * x) / np.sum(weights)
        if np.abs(mu - mu_new) < tol:
            break
        mu = mu_new
    return mu


if __name__ == "__main__":

    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    Ks = [10000, 50000, 100000]
    Js = [100]
    sigma_es = [0.01]
    M = 10
    batchsize = [304, 1504, 3004]
    d = 2
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]
    dataspace = "/linkhome/rech/genpuz01/umi36fq/"       #"/mnt/hdd2/ioannischalkiadakis/"
    dir_in = "{}/idealdata_rsspaper/".format(dataspace)
    dir_out = "{}/rsspaper_expII/".format(dataspace)
    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 

    algorithms = ["icmd"]
    colors = {Ks[0]:"Crimson", Ks[1]:"ForestGreen", Ks[2]:"Maroon"}
    param_err_fig = {}
    param_sqerr_fig = {}
    for param in parameter_names:
        if param in ["X", "Z"]:
            param_err_fig["{}_RT".format(param)] = go.Figure()
            param_sqerr_fig["{}_RT".format(param)] = go.Figure()
        param_err_fig[param] = go.Figure()
        param_sqerr_fig[param] = go.Figure()
    time_fig = make_subplots(specs=[[{"secondary_y": True}]])
    ram_fig_max = go.Figure()
    cpu_fig_max = go.Figure()
    ram_fig_avg = go.Figure()
    cpu_fig_avg = go.Figure()
    for K in Ks:                
        theta_err = {}
        theta_sqerr = {}
        theta_err_RT = {}
        theta_sqerr_RT = {}
        for param in parameter_names:
            theta_err[param] = []
            theta_sqerr[param] = []
            theta_err_RT[param] = []
            theta_sqerr_RT[param] = []
        ram = {"max":[], "avg":[]}
        cpu_util = {"max":[], "avg":[]}
        runtimes = []
        dataloglik = []
        estimation_sq_error_per_trial_per_batch = dict()
        estimation_sq_error_per_trial_per_batch_nonRT = dict()
        estimation_error_per_trial_per_batch = dict()
        estimation_error_per_trial_per_batch_nonRT = dict()
        for J in Js:
            parameter_space_dim = (K+J)*d + J + K + 2
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
            for sigma_e in sigma_es:                
                for algo in algorithms:
                    res_path = "{}/data_K{}_J{}_sigmae{}/".format(dir_in, K, J, str(sigma_e).replace(".", ""))                    
                    for trial in range(M):              
                        theta_true = np.zeros((parameter_space_dim,))
                        # load true param vector - same over all trials but store in each trial folder for convenience
                        with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(res_path, trial), "r") as f:
                            for result in f.iter(type=dict, skip_invalid=True):
                                for param in parameter_names:
                                    theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
                        
                        if algo == "mle":
                            trial_path = "{}/{}/{}/".format(res_path, trial, batchsize)
                        elif algo == "ca":
                            trial_path = "{}/{}/estimation_CA/".format(res_path, trial)
                        elif algo == "icmd":
                            trial_path = "{}/{}/estimation_ICM_data_annealing_evaluate_posterior_elementwise/".format(res_path, trial)
                        elif algo == "icmp":
                            trial_path = "{}/{}/estimation_ICM_evaluate_posterior_elementwise/".format(res_path, trial)
                        
                        if algo == "ca":   
                            # only assess ideal points for X, Z                         
                            # load data    
                            with open("{}/{}/Y.pickle".format(res_path, trial), "rb") as f:
                                Y = pickle.load(f)
                            Y = Y.astype(np.int8).reshape((K, J), order="F")   
                            _, K_new, J_new, theta_true_ca, _, _ , k_idx, j_idx = clean_up_data_matrix(Y, K, J, d, theta_true, parameter_names, param_positions_dict)

                            with jsonlines.open("{}/params_out_global_theta_hat.jsonl".format(trial_path), mode="r") as f: 
                                for result in f.iter(type=dict, skip_invalid=True):                                    
                                    dataloglik.append(result["logfullposterior"])
                                    theta_err["X"].append(result["err_x_nonRT"])
                                    theta_err["Z"].append(result["err_z_nonRT"])
                                    theta_err_RT["X"].append(result["err_x_RT"])
                                    theta_err_RT["Z"].append(result["err_z_RT"])
                                    theta_sqerr["X"].append(result["mse_x_nonRT"])
                                    theta_sqerr["Z"].append(result["mse_z_nonRT"])
                                    theta_sqerr_RT["X"].append(result["mse_x_RT"])
                                    theta_sqerr_RT["Z"].append(result["mse_z_RT"])                                    
                                    param_positions_dict_ca = result["param_positions_dict"]                                                                 
                            with jsonlines.open("{}/efficiency_metrics.jsonl".format(trial_path), mode="r") as f: 
                                for result in f.iter(type=dict, skip_invalid=True):     
                                    runtimes.append(result["wall_duration"]) # in seconds
                                    cpu_util["avg"].append(result["avg_total_cpu_util"])
                                    cpu_util["max"].append(result["max_total_cpu_util"])
                                    ram["avg"].append(result["avg_total_ram_residentsetsize_MB"]/1000)
                                    ram["max"].append(result["max_total_ram_residentsetsize_MB"]/1000)
                        elif algo == "icmp":         
                            readinfile = "{}/params_out_global_theta_hat.jsonl".format(trial_path)
                            precomputed_errors = False
                            if pathlib.Path("{}/params_out_global_theta_hat_upd_with_computed_err.jsonl".format(trial_path)).exists():
                                precomputed_errors = True
                                readinfile = "{}/params_out_global_theta_hat_upd_with_computed_err.jsonl".format(trial_path)             
                            with jsonlines.open(readinfile, mode="r") as f: 
                                for result in f.iter(type=dict, skip_invalid=True):                                    
                                    dataloglik.append(result["logfullposterior"])                                    
                                    param_positions_dict = result["param_positions_dict"]                                         
                                    for param in parameter_names:                                        
                                        param_true = theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                                        param_hat = result[param]                                        
                                        if param == "X":                                                                
                                            if not precomputed_errors:
                                                X_true = np.asarray(param_true).reshape((d, K), order="F")
                                                X_hat = np.asarray(param_hat).reshape((d, K), order="F")
                                                Rx, tx, mse_x, mse_x_nonRT, err_x, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, 
                                                                                                                                            param_hat=X_hat, 
                                                                                                                                            seedint=seed_value)
                                                result["err_x_nonRT"] = err_x_nonRT
                                                result["err_x_RT"] = err_x
                                                result["mse_x_nonRT"] = mse_x_nonRT
                                                result["mse_x_RT"] = mse_x
                                            else:
                                                err_x_nonRT = result["err_x_nonRT"]
                                                err_x = result["err_x_RT"]
                                                mse_x_nonRT = result["mse_x_nonRT"]
                                                mse_x = result["mse_x_RT"]
                                            theta_err_RT[param].append(err_x)
                                            theta_err[param].append(err_x_nonRT)
                                            theta_sqerr_RT[param].append(mse_x)
                                            theta_sqerr[param].append(mse_x_nonRT)
                                        elif param == "Z":      
                                            if not precomputed_errors:                                       
                                                Z_true = np.asarray(param_true).reshape((d, J), order="F")
                                                Z_hat = np.asarray(param_hat).reshape((d, J), order="F")
                                                Rz, tz, mse_z, mse_z_nonRT, err_z, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, 
                                                                                                                                            param_hat=Z_hat, 
                                                                                                                                            seedint=seed_value)
                                                result["err_z_nonRT"] = err_z_nonRT
                                                result["err_z_RT"] = err_z
                                                result["mse_z_nonRT"] = mse_z_nonRT
                                                result["mse_z_RT"] = mse_z        
                                            else:        
                                                err_z_nonRT = result["err_z_nonRT"]
                                                err_z = result["err_z_RT"]
                                                mse_z_nonRT = result["mse_z_nonRT"]
                                                mse_z = result["mse_z_RT"]                 
                                            theta_err_RT[param].append(err_z)
                                            theta_err[param].append(err_z_nonRT)
                                            theta_sqerr_RT[param].append(mse_z)
                                            theta_sqerr[param].append(mse_z_nonRT)    
                                        elif param in ["gamma", "delta", "sigma_e"]:
                                            # scalars
                                            rel_err = (param_true - param_hat)/param_true
                                            rel_se = rel_err**2
                                            theta_err[param].append(rel_err[0])
                                            theta_sqerr[param].append(rel_se[0])
                                        else:
                                            rel_err = (np.asarray(param_true) - np.asarray(param_hat))/np.asarray(param_true)
                                            sq_err = rel_err**2            
                                            theta_err[param].append(float(np.mean(rel_err)))    
                                            theta_sqerr[param].append(float(np.mean(sq_err)))
                                    if not precomputed_errors:
                                        # save updated file
                                        out_file = "{}/params_out_global_theta_hat_upd_with_computed_err.jsonl".format(trial_path)
                                        with open(out_file, 'a') as f:         
                                            writer = jsonlines.Writer(f)
                                            writer.write(result)
                                    # only consider the best solution
                                    break
                            with jsonlines.open("{}/efficiency_metrics.jsonl".format(trial_path), mode="r") as f: 
                                for result in f.iter(type=dict, skip_invalid=True):     
                                    runtimes.append(result["wall_duration"]/60) # in minutes
                                    cpu_util["avg"].append(result["avg_total_cpu_util"])
                                    cpu_util["max"].append(result["max_total_cpu_util"])
                                    ram["avg"].append(result["avg_total_ram_residentsetsize_MB"]/1000)
                                    ram["max"].append(result["max_total_ram_residentsetsize_MB"]/1000)
                                    break
                        elif algo == "mle":
                            m = trial
                            estimation_sq_error_per_trial_per_batch[m] = dict()
                            estimation_sq_error_per_trial_per_batch_nonRT[m] = dict()
                            estimation_error_per_trial_per_batch[m] = dict()
                            estimation_error_per_trial_per_batch_nonRT[m] = dict()
                            for param in parameter_names:            
                                estimation_sq_error_per_trial_per_batch[m][param] = []
                                estimation_sq_error_per_trial_per_batch_nonRT[m][param] = []
                                estimation_error_per_trial_per_batch[m][param] = []
                                estimation_error_per_trial_per_batch_nonRT[m][param] = []
                            data_location = trial_path
                            readinfile = "{}/params_out_combined_theta_hat.jsonl".format(trial_path)
                            precomputed_errors = False
                            if pathlib.Path(readinfile).exists():
                                precomputed_errors = True
                                ffop = open(readinfile, mode="r")
                                reader = jsonlines.Reader(ffop)
                                for item in reader:
                                    print("DMLE - loaded precomputed data.")
                                params_out = item 
                            else:    
                                params_out, estimation_sq_error_per_trial_per_batch[m], estimation_sq_error_per_trial_per_batch_nonRT[m],\
                                estimation_error_per_trial_per_batch[m], estimation_error_per_trial_per_batch_nonRT[m] = \
                                        combine_estimate_variance_rule(data_location, J, K, d, parameter_names, 
                                                                    estimation_sq_error_per_trial_per_batch[m], 
                                                                    estimation_sq_error_per_trial_per_batch_nonRT[m],
                                                                    estimation_error_per_trial_per_batch[m], 
                                                                    estimation_error_per_trial_per_batch_nonRT[m],
                                                                    theta_true, param_positions_dict, seedint=seed_value)    
                            for param in parameter_names:
                                if param == "X":   
                                    if not precomputed_errors:      
                                        param_hat = params_out[param].reshape((d*K,), order="F").tolist()     
                                        X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                                        X_hat = np.asarray(param_hat).reshape((d, K), order="F")
                                        Rx, tx, mse_x, mse_x_nonRT, err_x, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, 
                                                                                                                                    param_hat=X_hat, 
                                                                                                                                    seedint=seed_value)
                                        params_out["err_x_nonRT"] = err_x_nonRT
                                        params_out["err_x_RT"] = err_x
                                        params_out["mse_x_nonRT"] = mse_x_nonRT
                                        params_out["mse_x_RT"] = mse_x
                                        params_out[param] = param_hat.copy()
                                    else:
                                        err_x_nonRT = params_out["err_x_nonRT"]
                                        err_x = params_out["err_x_RT"]
                                        mse_x_nonRT = params_out["mse_x_nonRT"]
                                        mse_x = params_out["mse_x_RT"]
                                    theta_err_RT[param].append(err_x)
                                    theta_err[param].append(err_x_nonRT)
                                    theta_sqerr_RT[param].append(mse_x)
                                    theta_sqerr[param].append(mse_x_nonRT)
                                elif param == "Z":
                                    if not precomputed_errors:     
                                        param_hat = params_out[param].reshape((d*J,), order="F").tolist()     
                                        Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                                        Z_hat = np.asarray(param_hat).reshape((d, J), order="F")
                                        Rz, tz, mse_z, mse_z_nonRT, err_z, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, 
                                                                                                                                    param_hat=Z_hat, 
                                                                                                                                    seedint=seed_value)  
                                        params_out["err_z_nonRT"] = err_z_nonRT
                                        params_out["err_z_RT"] = err_z
                                        params_out["mse_z_nonRT"] = mse_z_nonRT
                                        params_out["mse_z_RT"] = mse_z 
                                        params_out[param] = param_hat.copy()
                                    else:
                                        err_z_nonRT = params_out["err_z_nonRT"]
                                        err_z = params_out["err_z_RT"]
                                        mse_z_nonRT = params_out["mse_z_nonRT"]
                                        mse_z = params_out["mse_z_RT"]                 
                                    theta_err_RT[param].append(err_z)
                                    theta_err[param].append(err_z_nonRT)
                                    theta_sqerr_RT[param].append(mse_z)
                                    theta_sqerr[param].append(mse_z_nonRT)    
                                elif param in ["beta", "alpha"]:
                                    if not isinstance(params_out[param], list):
                                        param_hat = params_out[param].tolist()
                                    else:
                                        param_hat = params_out[param]
                                    rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - param_hat)/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]     
                                    mse = np.mean(rel_err**2)  
                                    theta_err[param].append(float(np.mean(rel_err)))    
                                    theta_sqerr[param].append(float(mse))
                                    params_out[param] = param_hat.copy()
                                else:
                                    param_hat = params_out[param]
                                    rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - param_hat)/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                                    mse = rel_err**2                                    
                                    theta_err[param].append(rel_err[0])
                                    theta_sqerr[param].append(mse[0])
                            if not precomputed_errors:
                                # save updated file
                                out_file = "{}/params_out_combined_theta_hat.jsonl".format(trial_path)
                                with open(out_file, 'a') as f:         
                                    writer = jsonlines.Writer(f)
                                    writer.write(params_out)
                            else:
                                reader.close()
                                ffop.close()
                                
                            # for efficiency metrics in dmle: provide averages/max over all batches and make remark in paper   
                            subdatasets_names = [file.name for file in pathlib.Path(trial_path).iterdir() if not file.is_file() and "dataset_" in file.name]      
                            batch_runtimes = []
                            batch_cpu_util_avg = []
                            batch_cpu_util_max = []
                            batch_ram_avg = []
                            batch_ram_max = []
                            for dataset_index in range(len(subdatasets_names)):                    
                                subdataset_name = subdatasets_names[dataset_index]                        
                                DIR_read = "{}/{}/estimation/".format(trial_path, subdataset_name)
                                with jsonlines.open("{}/efficiency_metrics.jsonl".format(DIR_read), mode="r") as f: 
                                    for result in f.iter(type=dict, skip_invalid=True):     
                                        batch_runtimes.append(result["wall_duration"]) # in seconds
                                        batch_cpu_util_avg.append(result["avg_total_cpu_util"])
                                        batch_cpu_util_max.append(result["max_total_cpu_util"])
                                        batch_ram_avg.append(result["avg_total_ram_residentsetsize_MB"]/1000)
                                        batch_ram_max.append(result["max_total_ram_residentsetsize_MB"]/1000)
                                        break
                            runtimes.append(np.mean(batch_runtimes)) # in seconds
                            cpu_util["avg"].append(np.mean(batch_cpu_util_avg))
                            cpu_util["max"].append(np.mean(batch_cpu_util_max))
                            ram["avg"].append(np.mean(batch_ram_avg))
                            ram["max"].append(np.mean(batch_ram_max))
                        elif algo == "icmd":
                            DIR_base = trial_path
                            readinfile = "{}/params_out_combined_theta_hat.jsonl".format(trial_path)
                            precomputed_errors = False
                            if pathlib.Path(readinfile).exists(): 
                                precomputed_errors = True
                                ffop = open(readinfile, mode="r")
                                reader = jsonlines.Reader(ffop)
                                for item in reader:
                                    print("ICM-D - loaded precomputed data.")
                                    break
                                params_out = item 
                            else:                                
                                params_out = dict()
                                params_out["X"] = np.zeros((d*K,))
                                params_out["beta"] = np.zeros((K,))    
                                for param in parameter_names:       
                                    theta = None
                                    all_estimates = []
                                    path = pathlib.Path(DIR_base)  
                                    subdatasets_names = [file.name for file in pathlib.Path(trial_path).iterdir() if not file.is_file()]                    
                                    for dataset_index in range(len(subdatasets_names)):                    
                                        subdataset_name = subdatasets_names[dataset_index]    
                                        # print(subdataset_name)            
                                        DIR_read = "{}/{}/".format(DIR_base, subdataset_name)
                                        path = pathlib.Path(DIR_read)  
                                        estimates_names = [file.name for file in pathlib.Path(path).iterdir() if file.is_file() and "_best" in file.name]
                                        if len(estimates_names) > 1:
                                            raise AttributeError("Should have 1 output estimation file.")
                                        elif len(estimates_names) == 0:
                                            print(trial_path)
                                            continue
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
                                                            params_out[param][start*d:end*d] = theta.copy()
                                                        else:
                                                            params_out[param][start:end] = theta.copy()
                                                    else:                                                 
                                                        theta = result[param]
                                                        all_estimates.append(theta)
                                                    # only consider best solution
                                                    break
                                    if param in ["X", "beta"]:
                                        params_out[param] = params_out[param].tolist()       
                                    else:
                                        all_estimates = np.stack(all_estimates)
                                        if param not in ["Z", "Phi", "alpha"]:
                                            all_estimates = all_estimates.flatten()
                                        if len(np.nonzero(np.diff(all_estimates))[0]) > 1:
                                            if param in ["Z", "Phi", "alpha"]:
                                                # compute variance over columns
                                                column_variances = np.var(all_estimates, ddof=1, axis=0)
                                                column_variances[np.argwhere(abs(column_variances)<1e-14)] = 1
                                                weights = 1/column_variances
                                                if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                                                    raise NotImplementedError("Perhaps estimation issue with: {}/{}".format(DIR_read, estim))
                                                # sum acrocs each coordinate's weight
                                                all_weights_sum = np.sum(weights, axis=0)
                                                try:
                                                    all_weights_norm = weights/all_weights_sum
                                                    assert np.allclose(np.sum(all_weights_norm, axis=0), np.ones(all_weights_sum.shape))
                                                except:
                                                    raise NotImplementedError("Perhaps estimation issue with: {}/{}".format(DIR_read, estim))                                                
                                                # element-wise multiplication
                                                weighted_estimate = np.sum(all_weights_norm*all_estimates, axis=0)
                                            else:
                                                # gamma, sigma_e: get median
                                                weighted_estimate = np.asarray(np.percentile(all_estimates, 50, method="lower"))
                                        else:
                                            if param in ["Z", "Phi", "alpha"]:
                                                # same estimate, set uniform weighting
                                                all_weights_norm = 1/len(all_estimates)     
                                                weighted_estimate = np.sum(all_weights_norm*all_estimates, axis=0)    
                                            else:
                                                weighted_estimate = np.asarray([all_estimates[0]])
                                        params_out[param] = weighted_estimate.tolist()                              
                            for param in parameter_names:
                                if param == "X":       
                                    if not precomputed_errors:          
                                        X_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, K), order="F")
                                        X_hat = np.asarray(params_out[param]).reshape((d, K), order="F")
                                        Rx, tx, mse_x, mse_x_nonRT, err_x, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, 
                                                                                                                                    seedint=seed_value)
                                        params_out["err_x_nonRT"] = err_x_nonRT
                                        params_out["err_x_RT"] = err_x
                                        params_out["mse_x_nonRT"] = mse_x_nonRT
                                        params_out["mse_x_RT"] = mse_x      
                                    else:
                                        err_x_nonRT = params_out["err_x_nonRT"]
                                        err_x = params_out["err_x_RT"]
                                        mse_x_nonRT = params_out["mse_x_nonRT"]
                                        mse_x = params_out["mse_x_RT"]
                                    theta_err_RT[param].append(err_x)
                                    theta_err[param].append(err_x_nonRT)
                                    theta_sqerr_RT[param].append(mse_x)
                                    theta_sqerr[param].append(mse_x_nonRT)
                                elif param == "Z":
                                    if not precomputed_errors:  
                                        Z_true = np.asarray(theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]).reshape((d, J), order="F")
                                        Z_hat = np.asarray(params_out[param]).reshape((d, J), order="F")
                                        Rz, tz, mse_z, mse_z_nonRT, err_z, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, 
                                                                                                                                    seedint=seed_value)
                                        params_out["err_z_nonRT"] = err_z_nonRT
                                        params_out["err_z_RT"] = err_z
                                        params_out["mse_z_nonRT"] = mse_z_nonRT
                                        params_out["mse_z_RT"] = mse_z      
                                    else:
                                        err_z_nonRT = params_out["err_z_nonRT"]
                                        err_z = params_out["err_z_RT"]
                                        mse_z_nonRT = params_out["mse_z_nonRT"]
                                        mse_z = params_out["mse_z_RT"]                                   
                                    theta_err_RT[param].append(err_z)
                                    theta_err[param].append(err_z_nonRT)
                                    theta_sqerr_RT[param].append(mse_z)
                                    theta_sqerr[param].append(mse_z_nonRT)           
                                elif param in ["beta", "alpha"]:
                                    rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]     
                                    mse = np.mean(rel_err**2)  
                                    theta_err[param].append(float(np.mean(rel_err)))    
                                    theta_sqerr[param].append(float(mse))
                                else:
                                    rel_err = (theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] - params_out[param])/theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]]
                                    mse = rel_err**2
                                    theta_err[param].append(float(rel_err[0]))
                                    theta_sqerr[param].append(float(mse[0]))      
                            if not precomputed_errors:
                                # save updated file
                                out_file = "{}/params_out_combined_theta_hat.jsonl".format(trial_path)
                                with open(out_file, 'a') as f:         
                                    writer = jsonlines.Writer(f)
                                    writer.write(params_out)
                            else:
                                reader.close()
                                ffop.close()

                            # for efficiency in icmd: provide averages/max over all batches and make remark in paper   
                            batch_runtimes = []
                            batch_cpu_util_avg = []
                            batch_cpu_util_max = []
                            batch_ram_avg = []
                            batch_ram_max = []
                            with jsonlines.open("{}/efficiency_metrics.jsonl".format(trial_path), mode="r") as f: 
                                for result in f.iter(type=dict, skip_invalid=True):     
                                    batch_runtimes.append(result["wall_duration"]/60) # in minutes
                                    batch_cpu_util_avg.append(result["avg_total_cpu_util"])
                                    batch_cpu_util_max.append(result["max_total_cpu_util"])
                                    batch_ram_avg.append(result["avg_total_ram_residentsetsize_MB"]/1000)
                                    batch_ram_max.append(result["max_total_ram_residentsetsize_MB"]/1000)
                                    break
                            runtimes.append(np.mean(batch_runtimes)) # in minutes
                            cpu_util["avg"].append(np.mean(batch_cpu_util_avg))
                            cpu_util["max"].append(np.mean(batch_cpu_util_max))
                            ram["avg"].append(np.mean(batch_ram_avg))
                            ram["max"].append(np.mean(batch_ram_max)) 
                    
        
        if K == 10000:
            plotname = "10,000"
            sec_y = False
        elif K == 50000:
            plotname = "50,000"
            sec_y = False
        elif K == 100000:
            plotname = "100,000"
            sec_y = False
        time_fig.add_trace(go.Box(
            y=runtimes, showlegend=True, name=plotname,
            boxpoints='outliers', line=dict(color=colors[K]),                                
        ), secondary_y=sec_y)
        ram_fig_max.add_trace(go.Box(
            y=ram["max"], showlegend=True, name="{}-max".format(plotname),
            boxpoints='outliers', line=dict(color=colors[K])                          
        ))
        ram_fig_avg.add_trace(go.Box(
            y=ram["avg"], showlegend=True, name="{}-avg".format(plotname),
            boxpoints='outliers', line=dict(color=colors[K])                          
        ))
        cpu_fig_max.add_trace(go.Box(
            y=cpu_util["max"], showlegend=True, name="{}-max".format(plotname),
            boxpoints='outliers', line=dict(color=colors[K])                          
        ))
        cpu_fig_avg.add_trace(go.Box(
            y=cpu_util["avg"], showlegend=True, name="{}-avg".format(plotname),
            boxpoints='outliers', line=dict(color=colors[K])                          
        ))
        for param in parameter_names:
            if param in ["X", "Z"]:
                param_err_fig[param].add_trace(go.Box(
                    y=theta_err[param], showlegend=True, name="{}".format(plotname),
                    boxpoints='outliers', line=dict(color=colors[K])                          
                ))
                param_sqerr_fig[param].add_trace(go.Box(
                    y=theta_sqerr[param], showlegend=True, name="{}".format(plotname),
                    boxpoints='outliers', line=dict(color=colors[K])                          
                ))
                param_err_fig["{}_RT".format(param)].add_trace(go.Box(
                    y=theta_err_RT[param], showlegend=True, name="{}-RT".format(plotname),
                    boxpoints='outliers', line=dict(color=colors[K])                          
                ))
                param_sqerr_fig["{}_RT".format(param)].add_trace(go.Box(
                    y=theta_sqerr_RT[param], showlegend=True, name="{}-RT".format(plotname),
                    boxpoints='outliers', line=dict(color=colors[K])                          
                ))
            else:
                param_err_fig[param].add_trace(go.Box(
                    y=theta_err[param], showlegend=True, name="{}".format(plotname),
                    boxpoints='outliers', line=dict(color=colors[K])                          
                ))
                param_sqerr_fig[param].add_trace(go.Box(
                    y=theta_sqerr[param], showlegend=True, name="{}".format(plotname),
                    boxpoints='outliers', line=dict(color=colors[K])                          
                ))
                
    # save figures per K
    savename = "{}/time_J{}_sigmae_{}.html".format(dir_out, J, str(sigma_e).replace(".", ""))        
    fix_plot_layout_and_save(time_fig, savename, xaxis_title="Users sample size", yaxis_title="Duration (in minutes)", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
    fix_plot_layout_and_save(time_fig, savename, xaxis_title="Users sample size", yaxis_title="Duration (in minutes)", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 
    savename = "{}/ram_max_J{}_sigmae_{}.html".format(dir_out, J, str(sigma_e).replace(".", ""))    
    fix_plot_layout_and_save(ram_fig_max, savename, xaxis_title="Users sample size", yaxis_title="Maximum RAM consumption (in GB)", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
    fix_plot_layout_and_save(ram_fig_max, savename, xaxis_title="Users sample size", yaxis_title="Maximum RAM consumption (in GB)", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 
    savename = "{}/ram_avg_J{}_sigmae_{}.html".format(dir_out, J, str(sigma_e).replace(".", ""))    
    fix_plot_layout_and_save(ram_fig_avg, savename, xaxis_title="Users sample size", yaxis_title="Average RAM consumption (in GB)", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
    fix_plot_layout_and_save(ram_fig_avg, savename, xaxis_title="Users sample size", yaxis_title="Average RAM consumption (in GB)", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 
    savename = "{}/cpu_max_J{}_sigmae_{}.html".format(dir_out, J, str(sigma_e).replace(".", ""))    
    fix_plot_layout_and_save(cpu_fig_max, savename, xaxis_title="Users sample size", yaxis_title="Maximum CPU utilisation (% usage of 1 core)", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 
    fix_plot_layout_and_save(cpu_fig_max, savename, xaxis_title="Users sample size", yaxis_title="Maximum CPU utilisation (% usage of 1 core)", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
    savename = "{}/cpu_avg_J{}_sigmae_{}.html".format(dir_out, J, str(sigma_e).replace(".", ""))    
    fix_plot_layout_and_save(cpu_fig_avg, savename, xaxis_title="Users sample size", yaxis_title="Average CPU utilisation (% usage of 1 core)", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
    fix_plot_layout_and_save(cpu_fig_avg, savename, xaxis_title="Users sample size", yaxis_title="Average CPU utilisation (% usage of 1 core)", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False)
    for param in parameter_names:
        savename = "{}/rel_err_{}_J{}_sigmae_{}.html".format(dir_out, param, J, str(sigma_e).replace(".", ""))    
        if param in ["X", "Z"]:
            savename = "{}/rel_err_RT_{}_J{}_sigmae_{}.html".format(dir_out, param, J, str(sigma_e).replace(".", ""))
            fix_plot_layout_and_save(param_err_fig["{}_RT".format(param)], 
                                savename, xaxis_title="Users sample size", 
                                yaxis_title="Mean relative error (under rotation/scaling)", 
                                title="", showgrid=False, showlegend=False, 
                                print_png=True, print_html=False, 
                                print_pdf=False) 
            fix_plot_layout_and_save(param_err_fig["{}_RT".format(param)], 
                                savename, xaxis_title="Users sample size", 
                                yaxis_title="Mean relative error (under rotation/scaling)", 
                                title="", showgrid=False, showlegend=True, 
                                print_png=False, print_html=True, 
                                print_pdf=False) 
        savename = "{}/rel_err_{}_J{}_sigmae_{}.html".format(dir_out, param, J, str(sigma_e).replace(".", ""))
        fix_plot_layout_and_save(param_err_fig[param], savename, xaxis_title="Users sample size", yaxis_title="Mean relative error", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False)    
        fix_plot_layout_and_save(param_err_fig[param], savename, xaxis_title="Users sample size", yaxis_title="Mean relative error", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 
        if param in ["X", "Z"]:
            savename = "{}/rel_sqerr_RT_{}_J{}_sigmae_{}.html".format(dir_out, param, J, str(sigma_e).replace(".", "")) 
            fix_plot_layout_and_save(param_sqerr_fig["{}_RT".format(param)], savename, 
                            xaxis_title="Users sample size", 
                            yaxis_title="Mean relative squared error (under rotation/scaling)", 
                            title="", showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
            fix_plot_layout_and_save(param_sqerr_fig["{}_RT".format(param)], savename, 
                            xaxis_title="Users sample size", 
                            yaxis_title="Mean relative squared error (under rotation/scaling)", 
                            title="", showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 
        savename = "{}/rel_sqerr_{}_J{}_sigmae_{}.html".format(dir_out, param, J, str(sigma_e).replace(".", "")) 
        fix_plot_layout_and_save(param_sqerr_fig[param], savename, xaxis_title="Users sample size", yaxis_title="Mean relative squared error", title="", 
                            showgrid=False, showlegend=False, 
                            print_png=True, print_html=False, 
                            print_pdf=False) 
        fix_plot_layout_and_save(param_sqerr_fig[param], savename, xaxis_title="Users sample size", yaxis_title="Mean relative squared error", title="", 
                            showgrid=False, showlegend=True, 
                            print_png=False, print_html=True, 
                            print_pdf=False) 

                       