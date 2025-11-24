import os 

os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["NUMBA_NUM_THREADS"] = "3"

import sys
import ipdb
import pathlib
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from idealpestimation.src.parallel_manager import jsonlines
from idealpestimation.src.icm_annealing_posteriorpower import icm_posterior_power_annealing
from idealpestimation.src.utils import pickle, \
                                        create_constraint_functions_icm, time, timedelta, \
                                                get_posterior_for_optimisation_vec,optimisation_dict2params,\
                                                    rank_and_plot_solutions, get_evaluation_grid, \
                                                            print_threadpool_info, clean_up_data_matrix
from idealpestimation.src.efficiency_monitor import Monitor


def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", 
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, temperature_rate=[0, 1], temperature_steps=[1e-3], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False,
        theta_true=None, percentage_parameter_change=1, min_sigma_e=None, fastrun=False,
        max_restarts=2, max_partial_restarts=2, max_halving=2, plot_online=False, seedint=1234):
        
        m = trialsmin

        if elementwise:
            if evaluate_posterior:                    
                DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_elementwise/".format(data_location, m)      
                if pathlib.Path("{}params_out_global_theta_hat.jsonl".format(DIR_out)).exists():
                    print("{}params_out_global_theta_hat.jsonl".format(DIR_out))              
                    return
            else:
                DIR_out = "{}/{}/estimation_ICM_differentiate_posterior_elementwise/".format(data_location, m)
        else:
            if evaluate_posterior:
                DIR_out = "{}/{}/estimation_ICM_evaluate_posterior_vector/".format(data_location, m)                    
            else:
                raise NotImplementedError("At the moment we only evaluate the posterior with vector parameters - minimising coordinate-wise is more efficient.")
        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)     
        
        # load data    
        with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
            Y = pickle.load(f)
        Y = Y.astype(np.int8).reshape((K, J), order="F")    

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
  
        args = (DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,                     
                parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)   
        
        # start efficiency monitoring - interval in seconds
        print_threadpool_info()
        monitor = Monitor(interval=2)
        monitor.start()            

        t_start = time.time()            
        thetas = icm_posterior_power_annealing(Y, param_positions_dict, args, temperature_rate=temperature_rate, 
                                            temperature_steps=temperature_steps, plot_online=plot_online, 
                                            percentage_parameter_change=percentage_parameter_change, fastrun=fastrun,
                                            data_annealing=False, annealing_prev=None, theta_part_annealing=None,
                                            max_restarts=max_restarts, max_partial_restarts=max_partial_restarts, 
                                            max_halving=max_halving, seedint=seedint)
        t_end = time.time()
        monitor.stop()
        wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
            avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
        efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                    avg_threads, max_threads, avg_processes, max_processes)
        elapsedtime = str(timedelta(seconds=t_end - t_start))   

        thetas_and_errors = []
        params_true = optimisation_dict2params(theta_true, param_positions_dict, J, K, d, parameter_names)            
        for solution in thetas:
            theta_curr = solution[0]
            params_hat = optimisation_dict2params(theta_curr, param_positions_dict, J, K, d, parameter_names)
            for param in ["X", "Z"]:
                if param == "X":                        
                    X_hat_vec = np.asarray(params_hat[param]).reshape((d*K,), order="F")       
                    X_true_vec = np.asarray(params_true[param]).reshape((d*K,), order="F")       
                    x_rel_err = (X_true_vec - X_hat_vec)/X_true_vec
                    x_sq_err = x_rel_err**2
                elif param == "Z":                               
                    Z_hat_vec = np.asarray(params_hat[param]).reshape((d*J,), order="F")         
                    Z_true_vec = np.asarray(params_true[param]).reshape((d*J,), order="F")       
                    z_rel_err = (Z_true_vec - Z_hat_vec)/Z_true_vec
                    z_sq_err = z_rel_err**2
            thetas_and_errors.append((theta_curr, None, None, np.mean(x_sq_err), np.mean(z_sq_err), None, None, np.mean(x_rel_err), np.mean(z_rel_err), solution[-1]))

        rank_and_plot_solutions(thetas_and_errors, elapsedtime, efficiency_measures, Y, J, K, d, parameter_names, 
                                dst_func, param_positions_dict, DIR_out, args, seedint=seedint, get_RT_error=False)


if __name__ == "__main__":

    parameter_vector_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])    

    seed_value = 8125 + parameter_vector_idx
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1

    dataspace = "/tmp/idealdata_expI/"         
    parameter_grid = pd.read_csv("/tmp/slurm_experimentI_icm_poster_upd.csv", header=None)
    parameter_vector = parameter_grid.iloc[parameter_vector_idx].values

    Mmin = int(parameter_vector[0])
    M = int(Mmin + 1)
    K = int(parameter_vector[1])
    J = int(parameter_vector[2])
    sigma_e_true = parameter_vector[3]
    total_running_processes = 1    

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes, elementwise, evaluate_posterior)
    # before halving annealing rate
    percentage_parameter_change = 0.2
    optimisation_method = "L-BFGS-B"
    dst_func = lambda x, y: np.sum((x-y)**2)
    penalty_weight_Z = 0.0
    constant_Z = 0.0
    retries = 20
    diff_iter = None
    disp = False
    # In parameter names keep the order fixed as is
    # full, with status quo
    # parameter_names = ["X", "Z", "Phi", "alpha", "beta", "gamma", "delta", "sigma_e"]
    # no status quo
    parameter_names = ["X", "Z", "alpha", "beta", "gamma", "sigma_e"]
    d = 2  
    # gridpoints_num = 30 # 30
    gridpoints_num = dict()
    gridpoints_num["X"] = 30
    gridpoints_num["Z"] = 30
    gridpoints_num["alpha"] = 80
    gridpoints_num["beta"] = 80
    gridpoints_num["gamma"] = 100
    gridpoints_num["sigma_e"] = 200
    prior_loc_x = np.zeros((d,))
    prior_scale_x = np.eye(d)
    prior_loc_z = np.zeros((d,))
    prior_scale_z = np.eye(d)
    prior_loc_phi = np.zeros((d,))
    prior_scale_phi = np.eye(d)
    prior_loc_alpha = 0
    prior_scale_alpha = 1    
    prior_loc_beta = 0
    prior_scale_beta = 1
    prior_loc_gamma = 0
    prior_scale_gamma = 1    
    prior_loc_delta = 0
    prior_scale_delta = 1        
    # a
    prior_loc_sigmae = 3
    # b
    prior_scale_sigmae = 0.5
    temperature_steps = [0, 1, 2, 5, 10]
    temperature_rate = [1e-3, 1e-2, 1e-1, 1]

    niter = 25 #25 # 15
    fastrun = True
    max_restarts = 2 #2
    max_partial_restarts = 4 # 2, 4
    max_halving = 2 # 2
    plot_online = False
    max_signal2noise_ratio = 25 # in dB   # max snr

    min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
    print(min_sigma_e)

    tol = 1e-6        
    data_location = "{}/data_K{}_J{}_sigmae{}/".format(dataspace, K, J, str(sigma_e_true).replace(".", ""))
    
    args = (None, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, None, tol,                     
            None, None, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
            gridpoints_num, diff_iter, disp, min_sigma_e, None)  

    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 3
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    main(J=J, K=K, d=d, total_running_processes=total_running_processes, 
        data_location=data_location, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, trialsmax=M, 
        penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=retries, 
        elementwise=elementwise, evaluate_posterior=evaluate_posterior, temperature_rate=temperature_rate, 
        temperature_steps=temperature_steps, L=niter, tol=tol, 
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
        prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, 
        prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma, 
        prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta,         
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
        gridpoints_num=gridpoints_num, diff_iter=diff_iter, disp=disp, theta_true=theta_true,
        percentage_parameter_change=percentage_parameter_change, min_sigma_e=min_sigma_e, fastrun=fastrun,
        max_restarts=max_restarts, max_partial_restarts=max_partial_restarts, max_halving=max_halving, plot_online=plot_online, seedint=seed_value)
    
