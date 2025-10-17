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
                                                                get_min_achievable_mse_under_rotation_trnsl, go, p_ij_arg_numbafast
from plotly.subplots import make_subplots

def get_polarisation_data(sum_over_users_sorted, X_true, Z_true, alpha_true, X_hat, Z_hat, alpha_hat, beta_hat, gamma_hat, K, sort_most2least_liked, 
                        pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, pij_sumi_median, 
                        pij_sumi_median_RT, theta_err, theta_err_RT, seed_value):

    pijs_est = p_ij_arg_numbafast(X_hat, Z_hat, alpha_hat, beta_hat, gamma_hat, K)
    sum_over_users_est = pijs_est.sum(axis=0)
    sum_over_users_est_sorted = sum_over_users_est[sort_most2least_liked]
    err = (sum_over_users_est_sorted-sum_over_users_sorted)/sum_over_users_sorted
    err_pijsumi_median = np.percentile(err, q=50, method="lower")
    pij_err.append(err)                            
    pij_sumi_mean.append(np.mean(err))
    pij_sumi_median.append(err_pijsumi_median)
    Z_hat_flat = Z_hat.reshape((Z_hat.shape[0]*Z_hat.shape[1],), order="F")
    Z_true_flat = Z_true.reshape((Z_true.shape[0]*Z_true.shape[1],), order="F")
    theta_err["Z"].append((Z_hat_flat[sort_most2least_liked]-Z_true_flat[sort_most2least_liked])/Z_true_flat[sort_most2least_liked])
    if alpha_true is not None:
        theta_err["alpha"].append((alpha_hat[sort_most2least_liked]-alpha_true[sort_most2least_liked])/alpha_true[sort_most2least_liked])
    # RT
    Rx, tx, mse_x_RT, mse_x_nonRT, err_x_RT, err_x_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=X_true, param_hat=X_hat, seedint=seed_value)            
    Rz, tz, mse_z_RT, mse_z_nonRT, err_z_RT, err_z_nonRT = get_min_achievable_mse_under_rotation_trnsl(param_true=Z_true, param_hat=Z_hat, seedint=seed_value) 
    reconstructedX = X_hat @ Rx + tx
    reconstructedZ = Z_hat @ Rz + tz
    pijs_est = p_ij_arg_numbafast(reconstructedX, reconstructedZ, alpha_hat, beta_hat, gamma_hat, K)
    sum_over_users_est = pijs_est.sum(axis=0)
    sum_over_users_est_sorted = sum_over_users_est[sort_most2least_liked]
    err = (sum_over_users_est_sorted-sum_over_users_sorted)/sum_over_users_sorted
    err_pijsumi_median = np.percentile(err, q=50, method="lower")
    pij_err_RT.append(err)
    pij_sumi_mean_RT.append(np.mean(err))
    pij_sumi_median_RT.append(err_pijsumi_median)
    reconstructedZ_flat = reconstructedZ.reshape((reconstructedZ.shape[0]*reconstructedZ.shape[1],), order="F")    
    theta_err_RT["Z"].append((reconstructedZ_flat[sort_most2least_liked]-Z_true_flat[sort_most2least_liked])/Z_true_flat[sort_most2least_liked])

    return pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT




if __name__ == "__main__":

    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    Ks = [50000]
    Js = [100]
    sigma_es = [0.01, 0.1, 0.5, 1.0, 5.0]
    M = 10
    batchsize = 1504 # 50k
    d = 2
    parameter_names = ["X", "Z", "alpha", "beta", "gamma" , "sigma_e"]
    dataspace = "/linkhome/rech/genpuz01/umi36fq/"       #"/mnt/hdd2/ioannischalkiadakis/"
    dir_in = "{}/idealdata_rsspaper_expIupd/".format(dataspace)
    dir_out = "{}/rsspaper_expI_pij_nooutliers/".format(dataspace)
    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True) 

    algorithms = ["icmp", "icmd", "ca", "mle"]
    colors = {"mle":"Crimson", "ca":"Tomato", "icmd":"ForestGreen", "icmp":"Maroon"}
    for K in Ks:
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
                param_pijmean_fig = go.Figure()
                param_pijmean_fig_RT = go.Figure()
                z_fig = go.Figure()
                z_fig_RT = go.Figure()
                alpha_fig = go.Figure()
                alpha_fig_RT = go.Figure()                
                for algo in algorithms:
                    res_path = "{}/data_K{}_J{}_sigmae{}/".format(dir_in, K, J, str(sigma_e).replace(".", ""))
                    # boxplots for all J
                    pij_err = []
                    pij_err_RT = []
                    # avg, median sumpij
                    pij_sumi_mean = []
                    pij_sumi_median = []
                    pij_sumi_mean_RT = []
                    pij_sumi_median_RT = []
                    theta_err = {}
                    theta_err_RT = {}
                    for param in ["Z", "alpha"]:
                        theta_err[param] = []
                        theta_err_RT[param] = []
                    
                    estimation_sq_error_per_trial_per_batch = dict()
                    estimation_sq_error_per_trial_per_batch_nonRT = dict()
                    estimation_error_per_trial_per_batch = dict()
                    estimation_error_per_trial_per_batch_nonRT = dict()
                    for trial in range(M):              
                        theta_true = np.zeros((parameter_space_dim,))
                        # load true param vector - same over all trials but store in each trial folder for convenience
                        with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(res_path, trial), "r") as f:
                            for result in f.iter(type=dict, skip_invalid=True):
                                for param in parameter_names:
                                    theta_true[param_positions_dict[param][0]:param_positions_dict[param][1]] = result[param] 
                        
                        with open("{}/{}/Utilities.pickle".format(res_path, trial), "rb") as f:
                            pijs_true = pickle.load(f)
                        # sum over i and sort over j in decreasing order
                        sum_over_users = pijs_true.sum(axis=0)
                        sort_most2least_liked = np.argsort(sum_over_users)[::-1]
                        sum_over_users_sorted = sum_over_users[sort_most2least_liked]

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
                            _, K_new, J_new, theta_true_ca, _, _ , k_idx, j_idx = clean_up_data_matrix(Y, K, J, d, theta_true, 
                                                                                                    parameter_names, param_positions_dict, verify=False)
                            if len(j_idx) > 0:
                                raise NotImplementedError("Drop ground truth columns when comparing...no J dropping at the moment though, check!")
                            with jsonlines.open("{}/params_out_global_theta_hat.jsonl".format(trial_path), mode="r") as f: 
                                for result in f.iter(type=dict, skip_invalid=True):                                    
                                    param_positions_dict_ca = result["param_positions_dict"]      
                                    X_hat = np.asarray(result["X"]).reshape((d, K_new), order="F")
                                    Z_hat = np.asarray(result["Z"]).reshape((d, J_new), order="F")
                            X_true = np.asarray(theta_true_ca[param_positions_dict_ca["X"][0]:param_positions_dict_ca["X"][1]]).reshape((d, K_new), order="F")
                            Z_true = np.asarray(theta_true_ca[param_positions_dict_ca["Z"][0]:param_positions_dict_ca["Z"][1]]).reshape((d, J_new), order="F")
                            # only for CA since estimation does not return these parameters!
                            alpha_true = theta_true_ca[param_positions_dict_ca["alpha"][0]:param_positions_dict_ca["alpha"][1]]
                            beta_true = theta_true_ca[param_positions_dict_ca["beta"][0]:param_positions_dict_ca["beta"][1]]
                            gamma_true = theta_true_ca[param_positions_dict_ca["gamma"][0]:param_positions_dict_ca["gamma"][1]]
                            alpha_hat = alpha_true
                            beta_hat = beta_true
                            gamma_hat = gamma_true      
                            loadpath = "{}/{}/pij_plottingdata_{}.pickle".format(res_path, trial, algo)
                            if pathlib.Path(loadpath).exists():
                                with open(loadpath, "rb") as f:
                                    pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                        pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = pickle.load(f)                  
                            else:
                                pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                    pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = get_polarisation_data(sum_over_users_sorted, X_true, Z_true, None, X_hat, 
                                                                                                                    Z_hat, alpha_hat, beta_hat, 
                                                                                                                    gamma_hat, K_new, sort_most2least_liked, 
                                                                                                                    pij_err, pij_err_RT, pij_sumi_mean, 
                                                                                                                    pij_sumi_mean_RT, pij_sumi_median, 
                                                                                                                    pij_sumi_median_RT, theta_err, 
                                                                                                                    theta_err_RT, seed_value)
                                with open(loadpath, "wb") as f:
                                    pickle.dump((pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT), f)

                        elif algo == "icmp":         
                            readinfile = "{}/params_out_global_theta_hat.jsonl".format(trial_path)
                            precomputed_errors = True
                            if not pathlib.Path(readinfile).exists():
                                print("Did not find: {}".format(readinfile))                                    
                                continue
                            # if pathlib.Path("{}/params_out_global_theta_hat_upd_with_computed_err.jsonl".format(trial_path)).exists():
                            #     precomputed_errors = True
                            #     readinfile = "{}/params_out_global_theta_hat_upd_with_computed_err.jsonl".format(trial_path)           
                            X_true = np.asarray(theta_true[param_positions_dict["X"][0]:param_positions_dict["X"][1]]).reshape((d, K), order="F")
                            Z_true = np.asarray(theta_true[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]]).reshape((d, J), order="F")  
                            alpha_true = np.asarray(theta_true[param_positions_dict["alpha"][0]:param_positions_dict["alpha"][1]])
                            loadpath = "{}/{}/pij_plottingdata_{}.pickle".format(res_path, trial, algo)
                            if pathlib.Path(loadpath).exists():
                                with open(loadpath, "rb") as f:
                                    pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                        pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = pickle.load(f)                  
                            else:
                                with jsonlines.open(readinfile, mode="r") as f: 
                                    for result in f.iter(type=dict, skip_invalid=True):                                                                        
                                        param_positions_dict = result["param_positions_dict"]         

                                        X_hat = np.asarray(result["X"]).reshape((d, K), order="F")
                                        Z_hat = np.asarray(result["Z"]).reshape((d, J), order="F")
                                        alpha_hat = np.asarray(result["alpha"])
                                        beta_hat = np.asarray(result["beta"])
                                        gamma_hat = np.asarray(result["gamma"])
                                            
                                        # if not precomputed_errors:
                                        #     # save updated file
                                        #     out_file = "{}/params_out_global_theta_hat_upd_with_computed_err.jsonl".format(trial_path)
                                        #     with open(out_file, 'a') as f:         
                                        #         writer = jsonlines.Writer(f)
                                        #         writer.write(result)
                                        # only consider the best solution
                                        break
                                pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                        pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = get_polarisation_data(sum_over_users_sorted, X_true, Z_true, alpha_true, X_hat, 
                                                                                                                        Z_hat, alpha_hat, beta_hat, 
                                                                                                                        gamma_hat, K, sort_most2least_liked, 
                                                                                                                        pij_err, pij_err_RT, pij_sumi_mean, 
                                                                                                                        pij_sumi_mean_RT, pij_sumi_median, 
                                                                                                                        pij_sumi_median_RT, theta_err, 
                                                                                                                        theta_err_RT, seed_value)
                                with open(loadpath, "wb") as f:
                                    pickle.dump((pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT), f)
                            
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
                            X_true = np.asarray(theta_true[param_positions_dict["X"][0]:param_positions_dict["X"][1]]).reshape((d, K), order="F")
                            Z_true = np.asarray(theta_true[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]]).reshape((d, J), order="F")  
                            alpha_true = np.asarray(theta_true[param_positions_dict["alpha"][0]:param_positions_dict["alpha"][1]])
                            X_hat = np.asarray(params_out["X"]).reshape((d, K), order="F")
                            Z_hat = np.asarray(params_out["Z"]).reshape((d, J), order="F")
                            alpha_hat = np.asarray(params_out["alpha"])
                            beta_hat = np.asarray(params_out["beta"])
                            gamma_hat = np.asarray(params_out["gamma"])
                            loadpath = "{}/{}/pij_plottingdata_{}.pickle".format(res_path, trial, algo)
                            if pathlib.Path(loadpath).exists():
                                with open(loadpath, "rb") as f:
                                    pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                        pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = pickle.load(f)                  
                            else:
                                pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = get_polarisation_data(sum_over_users_sorted, X_true, Z_true, alpha_true, X_hat, 
                                                                                                                Z_hat, alpha_hat, beta_hat, 
                                                                                                                gamma_hat, K, sort_most2least_liked, 
                                                                                                                pij_err, pij_err_RT, pij_sumi_mean, 
                                                                                                                pij_sumi_mean_RT, pij_sumi_median, 
                                                                                                                pij_sumi_median_RT, theta_err, 
                                                                                                                theta_err_RT, seed_value)
                                with open(loadpath, "wb") as f:
                                    pickle.dump((pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT), f)
                            
                            
                            # if not precomputed_errors:
                            #     # save updated file
                            #     out_file = "{}/params_out_combined_theta_hat.jsonl".format(trial_path)
                            #     with open(out_file, 'a') as f:         
                            #         writer = jsonlines.Writer(f)
                            #         writer.write(params_out)
                            # else:
                            #     reader.close()
                            #     ffop.close()
                                
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
                                        DIR_read = "{}/{}/".format(DIR_base, subdataset_name)
                                        path = pathlib.Path(DIR_read)  
                                        estimates_names = [file.name for file in pathlib.Path(path).iterdir() if file.is_file() and "_best" in file.name]
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
                            
                            X_true = np.asarray(theta_true[param_positions_dict["X"][0]:param_positions_dict["X"][1]]).reshape((d, K), order="F")
                            Z_true = np.asarray(theta_true[param_positions_dict["Z"][0]:param_positions_dict["Z"][1]]).reshape((d, J), order="F")  
                            alpha_true = np.asarray(theta_true[param_positions_dict["alpha"][0]:param_positions_dict["alpha"][1]])  
                            X_hat = np.asarray(params_out["X"]).reshape((d, K), order="F")
                            Z_hat = np.asarray(params_out["Z"]).reshape((d, J), order="F")
                            alpha_hat = np.asarray(params_out["alpha"])
                            beta_hat = np.asarray(params_out["beta"])
                            gamma_hat = np.asarray(params_out["gamma"])
                            loadpath = "{}/{}/pij_plottingdata_{}.pickle".format(res_path, trial, algo)
                            if pathlib.Path(loadpath).exists():
                                with open(loadpath, "rb") as f:
                                    pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                        pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = pickle.load(f)                  
                            else:
                                pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, \
                                pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT = get_polarisation_data(sum_over_users_sorted, X_true, Z_true, alpha_true, X_hat, 
                                                                                                            Z_hat, alpha_hat, beta_hat, 
                                                                                                            gamma_hat, K, sort_most2least_liked, 
                                                                                                            pij_err, pij_err_RT, pij_sumi_mean, 
                                                                                                            pij_sumi_mean_RT, pij_sumi_median, 
                                                                                                            pij_sumi_median_RT, theta_err, 
                                                                                                            theta_err_RT, seed_value)
                                with open(loadpath, "wb") as f:
                                    pickle.dump((pij_err, pij_err_RT, pij_sumi_mean, pij_sumi_mean_RT, pij_sumi_median, pij_sumi_median_RT, theta_err, theta_err_RT), f)
                            
                            if not precomputed_errors:
                                # save updated file
                                out_file = "{}/params_out_combined_theta_hat.jsonl".format(trial_path)
                                with open(out_file, 'a') as f:         
                                    writer = jsonlines.Writer(f)
                                    writer.write(params_out)
                            else:
                                pass
                                # reader.close()
                                # ffop.close()

                    # add plots per algorithm
                    if algo == "ca":
                        plotname = "CA"
                        sec_y = True
                    elif algo == "mle":
                        plotname = "D-MLE"
                        sec_y = True
                    elif algo == "icmd":
                        plotname = "ICM-D"
                        sec_y = False
                    elif algo == "icmp":
                        plotname = "ICM-P"
                        sec_y = False
                
                    algo_leadusers_boxplots = go.Figure()
                    pij_err_alltrials = np.stack(pij_err)
                    pij_err_alltrials_mean = np.mean(pij_err_alltrials, axis=0)
                    pij_err_alltrials_median = np.percentile(pij_err_alltrials, q=50, axis=0)
                    for jidx in range(pij_err_alltrials.shape[1]):
                        algo_leadusers_boxplots.add_trace(go.Box(
                                y=pij_err_alltrials[:, jidx], showlegend=False,
                                boxpoints='outliers', line=dict(color=colors[algo])                          
                            ))
                    algo_leadusers_boxplots.update_xaxes(showticklabels=False)
                    savename = "{}/pij_sumi_leadusers_err_algorithm_{}_K{}_J{}_sigmae_{}.html".format(dir_out, algo, K, J, str(sigma_e).replace(".", ""))
                    fix_plot_layout_and_save(algo_leadusers_boxplots, 
                                        savename, xaxis_title="Lead users rank (most to least liked)", 
                                        yaxis_title="Relative error", 
                                        title="", showgrid=False, showlegend=False, 
                                        print_png=True, print_html=False, 
                                        print_pdf=True) 
                    param_pijmean_fig.add_trace(go.Scatter(y=pij_err_alltrials_mean, x=np.arange(1, pij_err_alltrials.shape[1]),
                                                        name="{}-Average".format(algo), showlegend=True, opacity=0.5, line=dict(color=colors[algo])))
                    param_pijmean_fig.add_trace(go.Scatter(y=pij_err_alltrials_median, x=np.arange(1, pij_err_alltrials.shape[1]),
                                                        name="{}-Median".format(algo), showlegend=True, opacity=1, line=dict(color=colors[algo])))
                    
                    algo_leadusers_boxplots_Z = go.Figure()
                    err_alltrials = np.stack(theta_err["Z"])
                    for jidx in range(err_alltrials.shape[1]):
                        algo_leadusers_boxplots_Z.add_trace(go.Box(
                                y=err_alltrials[:, jidx], showlegend=False,
                                boxpoints='outliers', line=dict(color=colors[algo])                          
                            ))
                    algo_leadusers_boxplots_Z.update_xaxes(showticklabels=False)
                    savename = "{}/Z_sortedleadusers_err_algorithm_{}_K{}_J{}_sigmae_{}.html".format(dir_out, algo, K, J, str(sigma_e).replace(".", ""))
                    fix_plot_layout_and_save(algo_leadusers_boxplots_Z, 
                                        savename, xaxis_title="Lead users rank (most to least liked)", 
                                        yaxis_title="Lead user ideal points relative error", 
                                        title="", showgrid=False, showlegend=False, 
                                        print_png=True, print_html=False, 
                                        print_pdf=True)


                    # RT
                    algo_leadusers_boxplots = go.Figure()
                    pij_err_alltrials = np.stack(pij_err_RT)
                    pij_err_alltrials_mean = np.mean(pij_err_alltrials, axis=0)
                    pij_err_alltrials_median = np.percentile(pij_err_alltrials, q=50, axis=0)
                    for jidx in range(pij_err_alltrials.shape[1]):
                        algo_leadusers_boxplots.add_trace(go.Box(
                                y=pij_err_alltrials[:, jidx], showlegend=False,
                                boxpoints='outliers', line=dict(color=colors[algo])                          
                            ))
                    algo_leadusers_boxplots.update_xaxes(showticklabels=False)
                    savename = "{}/pij_sumi_leadusers_err_RT_algorithm_{}_K{}_J{}_sigmae_{}.html".format(dir_out, algo, K, J, str(sigma_e).replace(".", ""))
                    fix_plot_layout_and_save(algo_leadusers_boxplots, 
                                        savename, xaxis_title="Lead users rank (most to least liked)", 
                                        yaxis_title="Relative error (under rotation/scaling)", 
                                        title="", showgrid=False, showlegend=False, 
                                        print_png=True, print_html=False, 
                                        print_pdf=True) 
                    param_pijmean_fig_RT.add_trace(go.Scatter(y=pij_err_alltrials_mean, x=np.arange(1, pij_err_alltrials.shape[1]),
                                                        name="{}-Average".format(algo), showlegend=True, opacity=0.5, line=dict(color=colors[algo])))
                    param_pijmean_fig_RT.add_trace(go.Scatter(y=pij_err_alltrials_median, x=np.arange(1, pij_err_alltrials.shape[1]),
                                                        name="{}-Median".format(algo), showlegend=True, opacity=1, line=dict(color=colors[algo])))
                    
                    algo_leadusers_boxplots_Z = go.Figure()
                    err_alltrials = np.stack(theta_err_RT["Z"])
                    for jidx in range(err_alltrials.shape[1]):
                        algo_leadusers_boxplots_Z.add_trace(go.Box(
                                y=err_alltrials[:, jidx], showlegend=False,
                                boxpoints='outliers', line=dict(color=colors[algo])                          
                            ))
                    algo_leadusers_boxplots_Z.update_xaxes(showticklabels=False)
                    savename = "{}/Z_sortedleadusers_err_RT_algorithm_{}_K{}_J{}_sigmae_{}.html".format(dir_out, algo, K, J, str(sigma_e).replace(".", ""))
                    fix_plot_layout_and_save(algo_leadusers_boxplots_Z, 
                                        savename, xaxis_title="Lead users rank (most to least liked)", 
                                        yaxis_title="Lead user ideal points relative error (under rotation/scaling)", 
                                        title="", showgrid=False, showlegend=False, 
                                        print_png=True, print_html=False, 
                                        print_pdf=True)


                    if algo != "ca":
                        algo_leadusers_boxplots_alpha = go.Figure()
                        err_alltrials = np.stack(theta_err["alpha"])
                        for jidx in range(err_alltrials.shape[1]):
                            algo_leadusers_boxplots_alpha.add_trace(go.Box(
                                    y=err_alltrials[:, jidx], showlegend=False,
                                    boxpoints='outliers', line=dict(color=colors[algo])                          
                                ))
                        algo_leadusers_boxplots_alpha.update_xaxes(showticklabels=False)
                        savename = "{}/alpha_sortedleadusers_err_algorithm_{}_K{}_J{}_sigmae_{}.html".format(dir_out, algo, K, J, str(sigma_e).replace(".", ""))
                        fix_plot_layout_and_save(algo_leadusers_boxplots_alpha, 
                                            savename, xaxis_title="Lead users rank (most to least liked)", 
                                            yaxis_title="Lead user popularity relative error", 
                                            title="", showgrid=False, showlegend=False, 
                                            print_png=True, print_html=False, 
                                            print_pdf=True)
                savename = "{}/pij_sumi_sortedleadusers_err_allalgorithms_K{}_J{}_sigmae_{}.html".format(dir_out, K, J, str(sigma_e).replace(".", ""))
                param_pijmean_fig.update_layout(legend=dict(orientation="h"))
                fix_plot_layout_and_save(param_pijmean_fig, 
                                    savename, xaxis_title="Lead users rank (most to least liked)", 
                                    yaxis_title=r"$\text{Lead user total utility relative error }(p_{\cdot j})$", 
                                    title="", showgrid=False, showlegend=True, 
                                    print_png=True, print_html=True, 
                                    print_pdf=True)
                savename = "{}/pij_sumi_sortedleadusers_err_RT_allalgorithms_K{}_J{}_sigmae_{}.html".format(dir_out, K, J, str(sigma_e).replace(".", ""))
                param_pijmean_fig_RT.update_layout(legend=dict(orientation="h"))
                fix_plot_layout_and_save(param_pijmean_fig_RT, 
                                    savename, xaxis_title="Lead users rank (most to least liked)", 
                                    yaxis_title=r"$\text{Lead user total utility relative error under rotation/scaling }(p_{\cdot j})$", 
                                    title="", showgrid=False, showlegend=True, 
                                    print_png=True, print_html=True, 
                                    print_pdf=True)
                


                    


                    
                    

                    