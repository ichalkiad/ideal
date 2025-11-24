import os 

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
import ipdb
import pathlib
import pickle
import numpy as np
import random
import math
import pandas as pd
from idealpestimation.src.parallel_manager import jsonlines
from idealpestimation.src.icm_annealing_posteriorpower import icm_posterior_power_annealing
from idealpestimation.src.utils import clean_up_data_matrix, time, timedelta, rank_and_plot_solutions, \
                                                            print_threadpool_info, \
                                                                optimisation_dict2params, \
                                                                    clean_up_data_matrix, \
                                                                        get_data_tempering_variance_combined_solution
from idealpestimation.src.efficiency_monitor import Monitor


def serial_worker(args):

    t0 = time.time()
    k_prev, s, Y_annealed, temperature_rate, temperature_steps, percentage_parameter_change,\
        fastrun, _, batchrows, theta_part_annealing, theta, elapsedtime, param_positions_dict,\
        DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,\
        parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x,\
        prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,\
        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae,\
        gridpoints_num, diff_iter, disp, min_sigma_e, theta_true, param_positions_dict_prev,\
        max_restarts, max_partial_restarts, max_halving, plot_online, seedint, subdataset_name = args
    
    # if np.allclose(s, 1):
    #     s = 1
    # DIR_out_icm = "{}/{}/".format(DIR_out, str(s).replace(".", ""))
    DIR_out_icm = "{}/{}/".format(DIR_out, subdataset_name)
    print(DIR_out_icm)
    pathlib.Path(DIR_out_icm).mkdir(parents=True, exist_ok=True)
    icm_args = (DIR_out_icm, total_running_processes, data_location, optimisation_method, parameter_names, J, K, d, dst_func, L, tol,\
                parameter_space_dim, m, penalty_weight_Z, constant_Z, retries, parallel, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x,\
                    prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha,\
                        prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae,\
                            gridpoints_num, diff_iter, disp, min_sigma_e, theta_true)

    thetas = icm_posterior_power_annealing(Y_annealed, param_positions_dict, icm_args,
                                temperature_rate=temperature_rate, temperature_steps=temperature_steps, 
                                plot_online=plot_online, percentage_parameter_change=percentage_parameter_change, 
                                fastrun=fastrun, data_annealing=True, annealing_prev=param_positions_dict_prev, 
                                theta_part_annealing=theta_part_annealing, max_restarts=max_restarts, 
                                max_partial_restarts=max_partial_restarts, max_halving=max_halving)
    elapsedtime = str(timedelta(seconds=time.time()-t0))   
    # get highest-likelihood solution and feed into icm_posterior_power_annealing to initialise theta for next iteration 
    # ensure indices of estimated theta segment are stored - not really needed for when batches are non-overlapping, except for Z, alpha, gamma, sigma_e
    print(len(thetas))
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
         
    theta_part_annealing = rank_and_plot_solutions(thetas_and_errors, elapsedtime, None, Y_annealed, J, batchrows, d, parameter_names, 
                                                dst_func, param_positions_dict, DIR_out_icm, icm_args, data_tempering=True, 
                                                row_start=k_prev, row_end=k_prev+batchrows, seedint=seedint, get_RT_error=False, plot_solutions=False)
    
    try:
        out_file = "{}/params_out_local_theta_hat_{}_{}.jsonl".format(DIR_out_icm, k_prev, k_prev+batchrows)
        params_out = dict()
        with jsonlines.open(out_file, 'r') as f:         
            for result in f.iter(type=dict, skip_invalid=True):
                for kparam in result.keys():
                    params_out[kparam] = result[kparam]
                    if isinstance(params_out[kparam], np.ndarray):
                        params_out[kparam] = params_out[kparam].tolist()
                break
        out_file = "{}/params_out_local_theta_hat_{}_{}_best.jsonl".format(DIR_out_icm, k_prev, k_prev+batchrows)
        with open(out_file, 'a') as f:         
            writer = jsonlines.Writer(f)
            writer.write(params_out)
    except:
        print("Did not converge: {}".format(DIR_out_icm))


def main(J=2, K=2, d=1, total_running_processes=1, data_location="/tmp/", batchsize=None,
        parallel=False, parameter_names={}, optimisation_method="L-BFGS-B", dst_func=lambda x:x**2, 
        parameter_space_dim=None, trialsmin=None, trialsmax=None, penalty_weight_Z=0.0, constant_Z=0.0, retries=10,
        elementwise=True, evaluate_posterior=True, tempering_rate=[0, 1], tempering_steps=[0.1], 
        L=20, tol=1e-6, prior_loc_x=0, prior_scale_x=1, 
        prior_loc_z=0, prior_scale_z=1, prior_loc_phi=0, prior_scale_phi=1, prior_loc_beta=0, prior_scale_beta=1, 
        prior_loc_alpha=0, prior_scale_alpha=1, prior_loc_gamma=0, prior_scale_gamma=1, prior_loc_delta=0, prior_scale_delta=1, 
        prior_loc_sigmae=0, prior_scale_sigmae=1,
        gridpoints_num=10, optimization_method="L-BFGS-B", diff_iter=None, disp=False,
        theta_true=None, percentage_parameter_change=1, min_sigma_e=None, fastrun=False,
        max_restarts=1, max_partial_restarts=1, max_halving=1, plot_online=False, seedint=None, data_start=0, data_end=0):
        
        theta_true_per_m = [] # in case some rows/columns are removed
        m = trialsmin
        
        if elementwise:
            if evaluate_posterior:                    
                DIR_out = "{}/{}/estimation_ICM_data_annealing_evaluate_posterior_elementwise/".format(data_location, m)                
            else:
                DIR_out = "{}/{}/estimation_ICM_data_annealing_differentiate_posterior_elementwise/".format(data_location, m)
        else:
            if evaluate_posterior:
                DIR_out = "{}/{}/estimation_ICM_data_annealing_evaluate_posterior_vector/".format(data_location, m)                
            else:
                raise NotImplementedError("At the moment we only evaluate the posterior with vector parameters - minimising coordinate-wise is more efficient.")
        pathlib.Path(DIR_out).mkdir(parents=True, exist_ok=True)        
        # load data    
        with open("{}/{}/Y.pickle".format(data_location, m), "rb") as f:
            Y = pickle.load(f)
        Y = Y.astype(np.int8).reshape((K, J), order="F")    

        param_positions_dict_init = dict()            
        k = 0
        for param in parameter_names:
            if param == "X":
                param_positions_dict_init[param] = (k, k + K*d)                       
                k += K*d    
            elif param in ["Z"]:
                param_positions_dict_init[param] = (k, k + J*d)                                
                k += J*d
            elif param in ["Phi"]:            
                param_positions_dict_init[param] = (k, k + J*d)                                
                k += J*d
            elif param == "beta":
                param_positions_dict_init[param] = (k, k + K)                                   
                k += K    
            elif param == "alpha":
                param_positions_dict_init[param] = (k, k + J)                                       
                k += J    
            elif param == "gamma":
                param_positions_dict_init[param] = (k, k + 1)                                
                k += 1
            elif param == "delta":
                param_positions_dict_init[param] = (k, k + 1)                                
                k += 1
            elif param == "sigma_e":
                param_positions_dict_init[param] = (k, k + 1)                                
                k += 1

        with jsonlines.open("{}/{}/synthetic_gen_parameters.jsonl".format(data_location, m), "r") as f:
            for result in f.iter(type=dict, skip_invalid=True):
                for param in parameter_names:
                    theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]] = result[param] 


        # Y, K, J, theta_true, param_positions_dict, parameter_space_dim, _, _ = clean_up_data_matrix(Y, K, J, d, theta_true, parameter_names, param_positions_dict)
        

        theta_true_per_m.append(theta_true)

        # min size for well-determined system of eq
        N = math.ceil(parameter_space_dim/J)
        theta_part_annealing = None
        param_positions_dict_prev = None
        
        if elementwise and not evaluate_posterior:
            raise NotImplementedError("Only elementwise and evaluate posterior functionality at the moment.")            
        else:
            assert elementwise and evaluate_posterior
            # assume non-overlapping data batches
            # for parallel data annealing: store partial solution, non-overlapping data batches
            theta = None
            elapsedtime = None

            print_threadpool_info()
            monitor = Monitor(interval=0.5)
            monitor.start()            

            t_start = time.time()          
                             
            k_theta_true = 0
            s = tempering_steps[0]
                        
            path = pathlib.Path("{}/{}/{}/".format(data_location, m, batchsize))                      
            
            subdataset_name = "dataset_{}_{}".format(data_start, data_end)     

            k_prev = data_start
            y_rows = data_end        
            Y_annealed = Y[k_prev:y_rows, :]            
            k = 0
            batchrows = y_rows - k_prev
            parameter_space_dim_theta = (batchrows+J)*d + J + batchrows + 2
            theta_true_partial_annealing = np.zeros((parameter_space_dim_theta,))
            param_positions_dict_partial_theta = dict()      

            DIR_out_icm = "{}/{}/".format(DIR_out, subdataset_name)
            out_file_exist_prev = "{}/params_out_local_theta_hat_{}_{}.jsonl".format(DIR_out_icm, k_prev, k_prev+batchrows)
            out_file_exist_best = "{}/params_out_local_theta_hat_{}_{}_best.jsonl".format(DIR_out_icm, k_prev, k_prev+batchrows)
            if pathlib.Path(out_file_exist_prev).exists() and pathlib.Path(out_file_exist_best).exists():
                print("Found: {}".format(out_file_exist_best))
                return      
            for param in parameter_names:
                if param == "X":
                    param_positions_dict_partial_theta[param] = (k, k + batchrows*d)
                    # X is always first in theta vector
                    theta_true_partial_annealing[k:k+batchrows*d] = theta_true[k_theta_true*d:k_theta_true*d+batchrows*d]            
                    k += batchrows*d    
                elif param in ["Z"]:
                    param_positions_dict_partial_theta[param] = (k, k + J*d)   
                    theta_true_partial_annealing[k:k+J*d] = theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]]             
                    k += J*d
                elif param in ["Phi"]:            
                    param_positions_dict_partial_theta[param] = (k, k + J*d)        
                    theta_true_partial_annealing[k:k+J*d] = theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]]                                                     
                    k += J*d
                elif param == "alpha":
                    param_positions_dict_partial_theta[param] = (k, k + J)       
                    theta_true_partial_annealing[k:k+J] = theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]]                     
                    k += J
                elif param == "beta":
                    param_positions_dict_partial_theta[param] = (k, k + batchrows)     
                    theta_true_partial_annealing[k:k+batchrows] = theta_true[param_positions_dict_init[param][0]+k_theta_true:param_positions_dict_init[param][0]+k_theta_true+batchrows]                             
                    k += batchrows    
                elif param == "gamma":
                    param_positions_dict_partial_theta[param] = (k, k + 1)     
                    theta_true_partial_annealing[k:k+1] = theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]] 
                    k += 1
                elif param == "delta":
                    param_positions_dict_partial_theta[param] = (k, k + 1)    
                    theta_true_partial_annealing[k:k+1] = theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]]                 
                    k += 1
                elif param == "sigma_e":
                    param_positions_dict_partial_theta[param] = (k, k + 1)  
                    theta_true_partial_annealing[k:k+1] = theta_true[param_positions_dict_init[param][0]:param_positions_dict_init[param][1]] 
                    k += 1
            

            # non-applicable, just for code compatibility
            temperature_steps_local = [0, 1, 2, 5, 10]
            temperature_rate_local = [1e-3, 1e-2, 1e-1, 1]

            worker_args = (k_prev, s, Y_annealed, temperature_rate_local, temperature_steps_local, percentage_parameter_change, 
                            fastrun, True, batchrows, theta_part_annealing, theta, elapsedtime, param_positions_dict_partial_theta, 
                            DIR_out, total_running_processes, data_location, optimisation_method, parameter_names, J, batchrows, d, dst_func, L, tol,                     
                            parameter_space_dim_theta, m, penalty_weight_Z, constant_Z, retries, False, elementwise, evaluate_posterior, prior_loc_x, prior_scale_x, 
                            prior_loc_z, prior_scale_z, prior_loc_phi, prior_scale_phi, prior_loc_beta, prior_scale_beta, prior_loc_alpha, prior_scale_alpha, 
                            prior_loc_gamma, prior_scale_gamma, prior_loc_delta, prior_scale_delta, prior_loc_sigmae, prior_scale_sigmae, 
                            gridpoints_num, diff_iter, disp, min_sigma_e, theta_true_partial_annealing, param_positions_dict_prev,
                            max_restarts, max_partial_restarts, max_halving, plot_online, seedint, subdataset_name)
            
            serial_worker(worker_args)                       
            
            print(k_theta_true, k_theta_true+batchrows) 
            k_theta_true += batchrows  
            param_positions_dict_prev = param_positions_dict_partial_theta
                           
            
            t_end = time.time()
            monitor.stop()
            wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes = monitor.report(t_end - t_start)
            efficiency_measures = (wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                                        avg_threads, max_threads, avg_processes, max_processes)
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
            elapsedtime = str(timedelta(seconds=t_end - t_start))       
        # get_data_tempering_variance_combined_solution(parameter_names, M, d, K, J, DIR_out, theta_true_per_m, param_positions_dict_init, topdir=data_location, seedint=seedint)


if __name__ == "__main__":

    # python idealpestimation/src/icm_annealing_data.py --trials 1 --K 30 --J 10 --sigmae 05 --elementwise --evaluate_posterior  --total_running_processes 5

    seed_value = 8125
    random.seed(seed_value)
    np.random.seed(seed_value)

    elementwise = True
    evaluate_posterior = True
    parallel = False
    total_running_processes = 1
    plot_online = False
    fastrun = True

    dataspace = "/tmp/idealdata_expI/"     
    parameter_vector_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])    
    parameter_grid = pd.read_csv("/tmp/slurm_experimentI_icm_data_upd.csv", header=None)
    parameter_vector = parameter_grid.iloc[parameter_vector_idx].values

    Mmin = int(parameter_vector[0])
    M = int(Mmin + 1)
    K = int(parameter_vector[1])
    J = int(parameter_vector[2])
    sigma_e_true = parameter_vector[3]
    batchsize = int(parameter_vector[4])
    data_start = int(parameter_vector[5])
    data_end = int(parameter_vector[6])

    print(parallel, Mmin, M, K, J, sigma_e_true, total_running_processes, elementwise, evaluate_posterior, data_start, data_end)
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
    parameter_names = ["X", "Z", "alpha", "beta", "gamma" , "sigma_e"]
    d = 2  
    # gridpoints_num = 50  #50,  80 for quite a few
    gridpoints_num = dict()
    gridpoints_num["X"] = 50
    gridpoints_num["Z"] = 50
    gridpoints_num["alpha"] = 100 #150
    gridpoints_num["beta"] = 100 # 150
    gridpoints_num["gamma"] = 150 # 300
    gridpoints_num["sigma_e"] = 250 #300
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
    tempering_steps = [0.1, 1]
    tempering_rate = [0.1]

    niter = 300 #100, 300
    max_restarts = 5 # 2,3 ,5
    max_partial_restarts = 5 #4 ,5, 10, 12
    max_halving = 2 # not applicable in data annealing
    max_signal2noise_ratio = 25 # in dB   # max snr

    min_sigma_e = (K*prior_scale_x[0, 0] + J*prior_scale_z[0, 0] + J*prior_scale_alpha + K*prior_scale_beta)/((K*J)*(10**(max_signal2noise_ratio/10)))
    print(min_sigma_e)

    tol = 1e-6        
    data_location = "{}/data_K{}_J{}_sigmae{}/".format(dataspace, K, J, str(sigma_e_true).replace(".", ""))
    total_running_processes = 1                 
    # full, with status quo
    # parameter_space_dim = (K+2*J)*d + J + K + 3
    # no status quo
    parameter_space_dim = (K+J)*d + J + K + 2
    theta_true = np.zeros((parameter_space_dim,))
    print("Parameter space dimensionality: {}".format(parameter_space_dim))
    main(J=J, K=K, d=d, total_running_processes=total_running_processes, 
        data_location=data_location, batchsize=batchsize, parallel=parallel, 
        parameter_names=parameter_names, optimisation_method=optimisation_method, 
        dst_func=dst_func, parameter_space_dim=parameter_space_dim, trialsmin=Mmin, trialsmax=M, 
        penalty_weight_Z=penalty_weight_Z, constant_Z=constant_Z, retries=retries, 
        elementwise=elementwise, evaluate_posterior=evaluate_posterior, tempering_rate=tempering_rate, 
        tempering_steps=tempering_steps, L=niter, tol=tol, 
        prior_loc_x=prior_loc_x, prior_scale_x=prior_scale_x, prior_loc_z=prior_loc_z, prior_scale_z=prior_scale_z, 
        prior_loc_phi=prior_loc_phi, prior_scale_phi=prior_scale_phi, prior_loc_beta=prior_loc_beta, prior_scale_beta=prior_scale_beta, 
        prior_loc_alpha=prior_loc_alpha, prior_scale_alpha=prior_scale_alpha, 
        prior_loc_gamma=prior_loc_gamma, prior_scale_gamma=prior_scale_gamma, 
        prior_loc_delta=prior_loc_delta, prior_scale_delta=prior_scale_delta,         
        prior_loc_sigmae=prior_loc_sigmae, prior_scale_sigmae=prior_scale_sigmae,
        gridpoints_num=gridpoints_num, diff_iter=diff_iter, disp=disp, theta_true=theta_true,
        percentage_parameter_change=percentage_parameter_change, min_sigma_e=min_sigma_e, fastrun=fastrun,
        max_restarts=max_restarts, max_partial_restarts=max_partial_restarts, max_halving=max_halving, 
        plot_online=plot_online, seedint=seed_value, data_start=data_start, data_end=data_end)
    
    
    
